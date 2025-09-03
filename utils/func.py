import torch
import numpy as np
from utils.initialization import file_path, grads_file, grads_rms, grads_file2, grads_file0

def grad(u, x):
    """ Get grad """
    gradient = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]
    return gradient

def grad_multi(u, inputs):
    """ Get gradients w.r.t. multiple inputs efficiently """
    gradients = torch.autograd.grad(
        u, inputs,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )
    return gradients

def tonp(tensor):
    """ Torch to Numpy """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))

def save_grad_stats(model, epoch, m, s):
    """ Grads Full net """
    with open(file_path+grads_file, 'a') as f:
        mean = np.linalg.norm(np.concatenate([arr.flatten() for arr in m]))
        std = np.linalg.norm(np.concatenate([arr.flatten() for arr in s]))
        for name, param in model.named_parameters():
            if name[-6:] == 'weight':
                ww = np.linalg.norm(tonp(param.data))
                f.write(f'{epoch},{name},{mean},{std},{ww}\n')

def save_grad_stats2(model, epoch, m, s):
    """ Grads Layer-wise """
    with open(file_path+grads_file2, 'a') as f:
        k = 0
        for name, param in model.named_parameters():
            if name[-6:] == 'weight':
                mean = np.linalg.norm(m[k])
                std = np.linalg.norm(s[k])
                ww = np.linalg.norm(tonp(param.data))
                f.write(f'{epoch},{name},{mean},{std},{ww}\n')
                k += 1
                
def save_grad_stats0(model, epoch, m, s, r):
    """ Save Adam grads """
    with open(file_path+grads_file0, 'a') as f:
        for name, _ in model.named_parameters():
            if name[-6:] == 'weight':
                f.write(f'{epoch},{name},{m},{s},{r}\n')

def save_grad_rms(model, epoch, m, s):
    """ Save RMS of grads """
    with open(file_path+grads_rms, 'a') as f:
        mean = np.linalg.norm(np.concatenate([arr.flatten() for arr in m]))
        std = np.linalg.norm(np.concatenate([arr.flatten() for arr in s]))
        for name, param in model.named_parameters():
            if name[-6:] == 'weight':
                ww = np.linalg.norm(tonp(param.data))
                f.write(f'{epoch},{name},{mean},{std},{ww}\n')
