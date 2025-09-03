import os
import sys
import time
import torch
import random
import math
import numpy as np
from pyDOE import lhs
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore", message="Attempting to run cuBLAS")

# Add the utils directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.snr import find_snr, find_phi, find_snr_adam
from utils.func import grad, tonp
from utils.mi import find_mi    
from utils.initialization import file_path, file_name, figs_folder, resid_folder, grads_file, grads_rms, grads_file2, grads_file0, mi_file

mpl.rcParams.update(mpl.rcParamsDefault)
np.set_printoptions(threshold=sys.maxsize)
plt.rcParams['figure.max_open_warning'] = 4
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
mpl.rcParams['font.size'] = 16
torch.backends.cudnn.benchmark = True

# File paths
os.makedirs(file_path+figs_folder, exist_ok=True)
os.makedirs(file_path+resid_folder, exist_ok=True)

if torch.cuda.is_available():
    """ Cuda support """
    print('cuda available')
    device = torch.device('cuda')
    
else:
    print('cuda not avail')
    device = torch.device('cpu')

def seed_torch(seed):
    """ Seed initialization """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.init()  # Initialize CUDA context
    # dummy = torch.tensor([0.0], device='cuda', requires_grad=True)
    # dummy.backward()
seed_torch(2341)
torch.cuda.empty_cache()

# Exact solution boundaries
xdim, tdim = 6.0, 6.0
nx, nt = (1001, 1001)
x = np.linspace(-1, 1, nx)
t = np.linspace(-1, 1, nt)
x0, t0 = np.meshgrid(x, t)
Exact0 = np.sin(xdim*math.pi*x0)*np.sin(tdim*math.pi*t0)
Exact0 = Exact0.T
x0, t0 = x[:,None], t[:,None]
lbc = torch.tensor([x0.min(), t0.min()]).to(torch.float32).to(device)
ubc = torch.tensor([x0.max(), t0.max()]).to(torch.float32).to(device)

plt.imshow(Exact0)
plt.show()

class DNN(torch.nn.Module):
    """ DNN Class """
    
    def __init__(self, layers):
        super(DNN, self).__init__()
        # Parameters
        self.depth = len(layers) - 1
        # Set up layer order dict
        self.activation = torch.nn.Tanh
        self.activations = None
        self.store_activations = False
        
        # Layers
        layer_list = list()
        for i in range(self.depth - 1):
            w_layer = torch.nn.Linear(layers[i], layers[i+1], bias=True)
            torch.nn.init.xavier_normal_(w_layer.weight)
            layer_list.append(('layer_%d' % i, w_layer))
            layer_list.append(('activation_%d' % i, self.activation()))

        w_layer = torch.nn.Linear(layers[-2], layers[-1], bias=True)
        torch.nn.init.xavier_normal_(w_layer.weight)
        layer_list.append(('layer_%d' % (self.depth - 1), w_layer))
        layerDict = OrderedDict(layer_list)
        # Deploy layers
        self.layers = torch.nn.Sequential(layerDict)
        
    def forward(self, x):
        if self.store_activations:
            activations = []
            out = x
            for layer in self.layers:
                out = layer(out)
                if isinstance(layer, self.activation):
                    activations.append(out.detach())  # Keep on GPU, detach from graph
            self.activations = activations
        else:
            out = self.layers(x)  # Fast path when not storing
        return out

class PhysicsInformedNN():
    """ PINN Class """
    
    def __init__(self, X_u, u, X_r, layers, lb, ub, dimx, dimt, savept=None):
        
        # Initialization
        self.rba = 1 # RBA weights
        self.snr = 0
        self.phi = 0
        self.adam_snr = 0
        self.ib = 0

        # Clear the file at the beginning of the training
        if self.snr == 1:
            with open(file_path+grads_file, 'w') as f:
                f.write('Epoch,Layer,Mean,Std,L2\n')
            with open(file_path+grads_rms, 'w') as f:
                f.write('Epoch,Layer,Mean,Std,L2\n')
            with open(file_path+grads_file2, 'w') as f:
                f.write('Epoch,Layer,Mean,Std,L2\n')
            with open(file_path+grads_file0, 'w') as f:
                f.write('Epoch,Layer,Mean,Std,L2\n')
        if self.adam_snr == 1:
            with open(file_path+'res_rms.txt', 'w') as f:
                f.write('Epoch,Layer,Mean,Std,L2\n')
        if self.ib > 0:
            with open(file_path+mi_file, 'w') as f:
                f.write('Epoch,Layer,mi_xt,saturation\n')

        self.iter = 0
        self.exec_time = 0
        self.batches = 100
        self.print_step = 100
        self.gradsave_step = 100
        self.savept = savept
        self.dimx, self.dimt = dimx, dimt
        self.dimx_, self.dimt_ = nx, nt  # prediction dimensions
        self.max_iters = 300000
        self.it = []; self.l2 = []; self.ll = []
        self.lrs, self.lrs2, self.iters = [], [], []
        self.mm, self.ss = [], []
        self.mi_xt, self.layer_saturation = [[] for _ in range(self.ib)], [[] for _ in range(self.ib)]
        self.mi_iters = []

        # Intermediate results
        self.Exact = Exact0
        X, T = np.meshgrid(x0, t0)
        X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
        self.xx = torch.tensor(X_star[:, 0:1], device=device, dtype=torch.float32)
        self.tt = torch.tensor(X_star[:, 1:2], device=device, dtype=torch.float32)
        
        # Data
        self.x_u = torch.tensor(X_u[:, 0:1], requires_grad=True, device=device, dtype=torch.float32)#.contiguous()
        self.t_u = torch.tensor(X_u[:, 1:2], requires_grad=True, device=device, dtype=torch.float32)#.contiguous()
        self.x_r = torch.tensor(X_r[:, 0:1], requires_grad=True, device=device, dtype=torch.float32)#.contiguous()
        self.t_r = torch.tensor(X_r[:, 1:2], requires_grad=True, device=device, dtype=torch.float32)#.contiguous()
        self.u = torch.tensor(u, device=device, dtype=torch.float32)
        self.N_r = tonp(self.x_r).size
        self.layers = layers
        self.dnn = DNN(layers).to(torch.float32).to(device)
        self.tn = tonp(self.t_r)[:, 0]
        self.xn = tonp(self.x_r)[:, 0]
        
        # RBA initialization
        self.init = 2
        self.eta = 0.001
        self.gamma = 0.999
        self.r_old = 0
        self.rsum = torch.zeros_like(self.x_r, requires_grad=False, device=device, dtype=torch.float32)

        # Optimizer (1st ord)
        self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.step_size = 5000
        
        # Number of bins along each axis (adjust as needed)
        num_bins_x = int(np.sqrt(self.batches))  # square root of total batches for equal division
        num_bins_t = num_bins_x
        # Calculate the range and bin size for x and t
        x_min, x_max = -1, 1
        t_min, t_max = 0, 1
        x_bin_size = (x_max - x_min) / num_bins_x
        t_bin_size = (t_max - t_min) / num_bins_t
        # Assign each point to a bin
        bin_indices = [[] for _ in range(self.batches)]
        for idx, (x, t) in enumerate(X_r):
            bin_x = int((x - x_min) / x_bin_size)
            bin_t = int((t - t_min) / t_bin_size)
            bin_index = bin_x * num_bins_t + bin_t # Calculate the bin index
            bin_indices[bin_index].append(idx)
        # Convert lists to tensors or arrays as needed
        self.bin_indices = [torch.tensor(bin_idx) for bin_idx in bin_indices]
        
        # Sort x and keep track of the indices
        sorted_t, sorted_indices = torch.sort(self.t_r[:, 0])
        bin_size = int(self.N_r/self.batches)
        # Create tensors of indices for each bin
        self.bin_indices2 = []
        for i in range(self.batches):
            start_index = i * bin_size
            end_index = min((i + 1) * bin_size, self.N_r)
            self.bin_indices2.append(tonp(sorted_indices[start_index:end_index]))

        self.training_path = [[tonp(param) for name, param in self.dnn.named_parameters() if "weight" in name or "bias" in name]]
        self.w = 1
        
        # Pre-compute all batch indices during initialization
        self.batch_indices = []
        batch_size = self.N_r // self.batches
        for i in range(self.batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, self.N_r)
            self.batch_indices.append(torch.arange(start_idx, end_idx, device=device))
        
    def net_u(self, x, t):
        """ Get the velocities """

        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_r(self, x, t, gp=0):
        """ Residual calculation """
        
        u = self.net_u(x, t)
        u_t = grad(u, t)
        u_tt = grad(u_t, t)
        u_x = grad(u, x)
        u_xx = grad(u_x, x)
        a1 = xdim
        a2 = tdim
        ksq = 1.0
        forcing = - (a1*torch.pi)**2*torch.sin(a1*torch.pi*x)*torch.sin(a2*torch.pi*t) - \
                    (a2*torch.pi)**2*torch.sin(a1*torch.pi*x)*torch.sin(a2*torch.pi*t) + \
                    ksq*torch.sin(a1*torch.pi*x)*torch.sin(a2*torch.pi*t)
        f = u_xx + u_tt + ksq*u - forcing
        return f, u_x

    def batch_loss(self, batch):
        """ Loss function
        """
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Indices
        self.idr = self.batch_indices[batch]

        # Predictions
        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, _ = self.net_r(self.x_r[self.idr], self.t_r[self.idr])
        if self.rba == True:
            ff = self.rsum[self.idr]
        else:
            ff = 1
        loss_r = torch.mean((ff*self.r_pred)**2)
        loss_u = torch.mean(((self.u_pred - self.u))**2)
        self.loss = loss_r + loss_u
        self.loss.backward()
        
    def batch_residuals(self, batch):
        """ Loss function
        """
        self.dnn.eval()
        # Indices
        self.idr = self.bin_indices[batch]
        
        if self.rba == True:
            ff = self.rsum[self.idr]
        else:
            ff = 1
            
        return np.mean(tonp(ff))

    def loss_func(self):
        """ Loss function """
        
        self.optimizer.zero_grad(set_to_none=True)

        # Predictions
        self.u_pred = self.net_u(self.x_u, self.t_u)
        self.r_pred, _ = self.net_r(self.x_r, self.t_r)

        # RBA
        eta = 1 if self.init == 2 and self.iter == 0 else self.eta
        r_norm = eta * (torch.abs(self.r_pred) - torch.min(torch.abs(self.r_pred))) / \
                    (torch.max(torch.abs(self.r_pred)) - torch.min(torch.abs(self.r_pred)))
        self.rsum = (self.rsum*self.gamma + r_norm).detach()

        if self.rba == True:
            loss_r = torch.mean((self.rsum*self.r_pred) ** 2)
        else:
            loss_r = torch.mean(self.r_pred**2)
        loss_u = torch.mean(((self.u_pred - self.u)) ** 2)

        # Loss calculation
        self.loss = loss_r + loss_u
        self.loss.backward()

        self.si = (self.ib > 0)
        if (self.iter+self.si) % self.print_step == 0:
            with torch.no_grad():
                # Grid prediction (for relative L2)
                res = self.net_u(self.xx, self.tt)
                sol = tonp(res)
                sol = np.reshape(sol, (self.dimt_, self.dimx_)).T

                if (self.iter % 200 == 0 and self.iter < 20000) or (self.iter % 500 == 0 and self.iter >= 20000):
                    plt.figure(1)
                    cmap = 'coolwarm' # RdYlBu_r'
                    ttd, xxd = tonp(self.tt), tonp(self.xx)
                    # tx = np.column_stack((ttd.min(), ttd.max(), xxd.min(), xxd.max()))
                    # np.save(file_path+figs_folder+'/r'+str(self.iter)+'.npy', tx)

                    # Prediction plot
                    plt.imshow(sol, extent=[ttd.min(), ttd.max(), xxd.min(), xxd.max()], origin='lower', cmap=cmap)
                    plt.gca().set_aspect(0.5)
                    plt.colorbar()
                    # plt.colorbar().set_ticks(np.linspace(sol.min()+.005, sol.max()-.005, 5).round(2))
                    plt.xticks([0, 0.25, 0.5, 0.75, 1])
                    plt.yticks([-1, -0.5, 0, 0.5, 1])
                    plt.savefig(file_path+figs_folder+'/training' + str(self.iter) + '.png', dpi=300)
                    plt.close()


                    zz = tonp(self.rsum)
                    tt = tonp(self.t_r)
                    xx = tonp(self.x_r)
#                     txz = np.column_stack((tt, xx, zz))
#                     np.save(file_path+resid_folder+'/xyz'+str(self.iter)+'.npy', txz)

                    # RBA plot
                    plt.figure(2)
                    h = plt.scatter(tt, xx, c=zz, s=8, cmap=cmap)
                    plt.xlim(0, 1)
                    plt.ylim(-1, 1)
                    plt.gca().set_aspect(0.5)
                    plt.colorbar()
                    # plt.colorbar().set_ticks(np.linspace(zz.min()+.005, zz.max()-.005, 5).round(2))
                    plt.xticks([0, 0.25, 0.5, 0.75, 1])
                    plt.yticks([-1, -0.5, 0, 0.5, 1])
                    plt.show()
                    plt.savefig(file_path+resid_folder+'/f' + str(self.iter) + '.png', dpi=300)
                    plt.clf()
                    plt.close()

                # L2 calculation
                self.l2_rel = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                l_inf = np.linalg.norm(self.Exact.flatten() - sol.flatten(), 2) / np.linalg.norm(self.Exact.flatten(), 2)
                print('Iter %d, Loss: %.3e, Rel_L2: %.3e, L_inf: %.3e, t/iter: %.1e' % 
                     (self.iter, self.loss.item(), self.l2_rel, l_inf, self.exec_time))
                print()
                
        # Optimizer step
        self.optimizer.step()

    def train(self):
        """ Train model """
        
        self.dnn.train()

        # Pre-allocate arrays for tracking
        self.it = np.zeros(self.max_iters//self.print_step, dtype=np.int32)
        self.l2 = np.zeros(self.max_iters//self.print_step, dtype=np.float32)
        self.ll = np.zeros(self.max_iters//self.print_step, dtype=np.float32)
        cnt = 0

        for self.iter in range(self.max_iters):

            # Store activations for MI
            self.dnn.store_activations = False
            store_acts = self.ib > 0 and (self.iter % 100 == 0 or self.iter in [10, 20, 40, 80])
            self.dnn.store_activations = store_acts

            start_time = time.time()
            self.loss_func()
            self.exec_time = time.time() - start_time

            # # Clear cache periodically
            # if epoch % 1000 == 0:
            #     torch.cuda.empty_cache()
                
            # Calculate gradient statistics
            if store_acts:
                find_mi(self, self.iter, self.ib, self.dnn, self.mi_iters, self.mi_xt, self.layer_saturation)
            if self.phi == 1:
                find_phi(self, self.iter, self.gradsave_step, 
                        self.batches, self.dnn, self.batch_residuals, 300)
            if self.snr == 1:
                find_snr(self, self.iter, self.gradsave_step, 
                        self.batches, self.dnn, self.batch_loss, 300)
            if self.adam_snr == 1:
                find_snr_adam(self, self.iter, self.optimizer, self.gradsave_step, 
                        self.dnn, self.lrs, self.lrs2, self.iters)

            if (self.iter+1) % self.step_size == 0:
                print('lr step')
                self.scheduler.step()
            # Store results
            if (self.iter+self.si) % self.print_step == 0:
                self.it[cnt] = self.iter
                self.l2[cnt] = self.l2_rel
                self.ll[cnt] = self.loss.item()
                cnt += 1

        # Write data
        # Stack them into a 2D array.
        d = np.column_stack((self.it, self.l2, self.ll))
        file = open(file_path + file_name, "w+")
        file.close()
        # Write to a txt file
        np.savetxt('losses.txt', d, fmt='%.10f %.10f %.10f')

        if self.savept != None:
            torch.save(self.dnn.state_dict(), str(self.savept)+".pt")
    
    def get_training_path(self):
        return self.training_path
    
    def predict(self, X):
        x = torch.tensor(X[:, 0:1]).to(device)
        t = torch.tensor(X[:, 1:2]).to(device)
        self.dnn.eval()
        u = self.net_u(x, t)
        u = tonp(u)
        return u

def main():
    # Collocation points
    dimx = nx
    dimt = nt
    N_u = 100
    N_r = 25600
    hidden = 128
    layers = [2] + [hidden]*6 + [1]

    # Definition
    Exact = Exact0.T
    tm = np.linspace(t0.min(), t0.max(), dimt)[:, None]
    xm = np.linspace(x0.min(), x0.max(), dimx)[:, None]
    X, T = np.meshgrid(xm, tm)
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))

    # Doman bounds
    lb = tonp(lbc)
    ub = tonp(ubc)
    xx1 = np.hstack((x0*0-1, t0))
    uu1 = x0*0

    # Top/bot boundaries
    xx2 = np.hstack((x0, t0*0+1))
    xx3 = np.hstack((x0, t0*0-1))
    xx4 = np.hstack((x0*0+1, t0))
    uu2 = uu1
    uu3 = uu1
    uu4 = uu1

    # Random choice
    idx0 = np.random.choice(dimt, N_u, replace=False)
    idx = np.random.choice(dimt, N_u, replace=False)
    idx2 = np.random.choice(dimt, N_u, replace=False)
    idx3 = np.random.choice(dimt, N_u, replace=False)

    u_train = np.vstack([uu1[idx0, :], uu2[idx, :], uu3[idx2, :], uu4[idx3, :]])
    X_u_train = np.vstack([xx1[idx0, :], xx2[idx, :], xx3[idx2, :], xx4[idx3, :]])

    # Collocation points
    X_r_train = lb + (ub-lb)*lhs(2, N_r)

    print('X_r shape:', X_r_train.shape)
    print('X_u shape:', X_u_train.shape)
    print('u shape:', u_train.shape)
                
    # plt.figure(1)
    # plt.title('Initial condition')
    # plt.plot(np.arange(uu1.shape[0]), uu1)
    # plt.show()

    # plt.figure(2)
    # plt.title('Collocation points')
    # plt.scatter(X_r_train[:, 1], X_r_train[:, 0], s=0.5)
    # plt.scatter(X_u_train[:, 1], X_u_train[:, 0], s=0.5, c='k')
    # plt.show()

    model = PhysicsInformedNN(X_u_train, u_train, X_r_train, layers, lb, ub, dimx, dimt=N_u, savept='weights')
    model.train()

if __name__ == "__main__":
    main()
