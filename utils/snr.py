from utils.func import tonp
import numpy as np
import torch
from utils.func import save_grad_stats0, save_grad_stats2, save_grad_rms, save_grad_stats

def find_phi(self, iter, gradsave_step, batches, dnn, batch_residuals, file_path):
    if iter % gradsave_step == 0 or (iter % 2 == 0 and iter < 300):
        resid_all_batches = []
        for i in range(batches):
            resid_all_batches.append(batch_residuals(i))
        grads_all_batches = np.array(resid_all_batches)
        m = np.mean(grads_all_batches)
        s = np.sqrt(np.mean(np.square(grads_all_batches)))
        with open(file_path+'res_rms.txt', 'a') as f:
            for name, param in dnn.named_parameters():
                if name[-6:] == 'weight':
                    ww = np.linalg.norm(tonp(param.data))
                    f.write(f'{iter},{name},{m},{s},{ww}\n')

def find_snr(self, iter, gradsave_step, batches, dnn, batch_loss, fitting_it):
    if iter % gradsave_step == 0 or (iter % 10 == 0 and iter < fitting_it):
        grads_all_batches = []
        for i in range(batches):
            j = 0
            batch_loss(i)
            for name, param in dnn.named_parameters():  # layer loop
                if param.grad is not None and name[-6:] == 'weight':
                    grads = tonp(param.grad.data)
                    if i == 0:
                        grads_all_batches.append(grads)
                    else:
                        grads_all_batches[j] = np.dstack((grads_all_batches[j], grads))
                    j += 1

        m = [np.mean(arr, axis=2) for arr in grads_all_batches]
        s = [np.std(arr, axis=2) for arr in grads_all_batches]
        save_grad_stats(dnn, iter, m, s)
        save_grad_stats2(dnn, iter, m, s)
        s = [np.sqrt(np.mean(np.square(arr), axis=2)) for arr in grads_all_batches]
        save_grad_rms(dnn, iter, m, s)

def find_snr_adam(self, iter, optimizer, gradsave_step, dnn, lrs, lrs2, iters):
    nm, den = [], []
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if not param.requires_grad:
                continue
            state = optimizer.state[param]
            if 'step' not in state or state['step'] == 0:
                continue
            if 'step' in state and state['step'] > 0:
                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                beta1, beta2 = param_group['betas']
                step = state['step']
                # Bias correction
                exp_avg_corr = exp_avg / (1 - beta1 ** step)
                exp_avg_sq_corr = exp_avg_sq / (1 - beta2 ** step)
                g, g2 = exp_avg_corr.view(-1), exp_avg_sq_corr.view(-1)
                # Compute corrected learning rate
                nm.append(g)
                den.append(g2.sqrt() + 1e-8)

    # Compute L2 norm
    nm, den = tonp(torch.cat(nm)), tonp(torch.cat(den))
    l2num, l2den = np.linalg.norm(nm), np.linalg.norm(den)
    l2_norm = l2num/l2den
    rlr = l2_norm
    l2_frac = np.abs(nm/den).mean() #np.linalg.norm(nm/den)
    if iter % gradsave_step == 0 or iter < 300:
        lrs.append(l2_norm)
        lrs2.append(l2_frac)
        iters.append(iter)
        save_grad_stats0(dnn, iter, l2num, l2den, l2_frac)
