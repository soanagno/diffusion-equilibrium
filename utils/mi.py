import numpy as np
from utils.func import tonp
import matplotlib.pyplot as plt
from utils.initialization import file_path, mi_file

def save_mi(epoch, mi_xt, sat):
    """ Mutual info """
    with open(file_path+mi_file, 'a') as f:
        for i in range(len(mi_xt)):
            f.write(f'{epoch},{i},{mi_xt[i][-1]},{sat[i][-1]}\n')

def av_diff(array):
    array = array[::10]
    rows = array.shape[0]
    cols = array.shape[1]
    k = 0
    
    for i in range(rows-1):
        j = np.arange(i+1, rows)
        a = np.abs(array[i]-array[j])
        a[a>0] = 1
        k += np.sum(a)
    return (np.sum(k))/((rows-1)*(rows)/2*cols)

def fast_av_diff(array):
    """Replace O(n²) with O(n log n) correlation"""
    array = array[::10]  # Sample
    # Correlation coefficient
    corr_matrix = np.corrcoef(array.T)
    return np.mean(1 - np.abs(corr_matrix))
    
def adaptive_bins(n_bins, values):
    valflat = values.flatten()
    valflat.sort()
    bsize  = int(len(valflat)/n_bins)
    excess = len(valflat) % n_bins
    bedges = np.cumsum([bsize+1 if i<excess else bsize for i in range(n_bins)])
    bins   = [(valflat[bedge-1]+valflat[bedge])/2 for bedge in bedges[:-1]]
    bins   = [valflat[0]-1] + bins + [valflat[-1]+1]
    return bins

def find_mi(self, iter, ib, dnn, mi_iters, mi_xt, layer_saturation):
    mi_iters.append(iter)
    act = dnn.activations

    for l in range(ib):
        an = tonp(act[l])
        N_r = an.shape[0]
        neurons = an.shape[1]
        
        # Get average bin width
        rg_sum, bins, eps = 0, 30, 1e-6
        ranges = np.abs(an.min(axis=0) - an.max(axis=0) - eps)
        rg_sum = ranges.sum()
        ww = rg_sum / neurons / bins
        digits_per_neuron = (ranges / ww).astype(int) + 1
        
        # Digitize T
        av_dgts = 0
        T_binned = np.zeros((N_r, neurons))
        for v in range(neurons):
            digits = digits_per_neuron[v]
            bin_edges = np.linspace(an[:, v].min(), an[:, v].max() + eps, num=digits)
            T_binned[:, v] = np.digitize(an[:, v], bins=bin_edges)
            uc = np.count_nonzero(T_binned[:, v] == digits-1) # upper saturated count
            lc = np.count_nonzero(T_binned[:, v] == 1) # lower saturated count
            av_dgts += (uc + lc)/N_r*100
            
        av_dgts /= neurons

        _, counts = np.unique(T_binned, axis=0, return_counts=True)
        dX = counts / sum(counts)
        MI_XT = -np.sum(dX * np.log2(dX+1e-24))
        mi_xt[l].append(MI_XT)
        layer_saturation[l].append(av_dgts)

    if iter % 10000 == 0 and iter > 1000:
        for l in range(ib):
            plt.figure(1)
            plt.plot(mi_iters, mi_xt[l])
            plt.title('MI_XT')
            plt.xscale('log')

            plt.figure(2)
            plt.plot(mi_iters, layer_saturation[l])
            plt.xscale('log')
            plt.title('Layer saturation')
        plt.show()
    save_mi(iter, mi_xt, layer_saturation)
