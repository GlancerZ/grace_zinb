import torch
import numba as nb
from numba import jit
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt

def filter_data(X, highly_genes=3000):

    X = np.ceil(X).astype(np.int)
    adata = sc.AnnData(X,dtype=np.float32)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4,min_disp=0.5, subset=True)
    if adata.X.shape[1] >= 3000:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4, min_disp=0.5, n_top_genes=highly_genes, subset=True)
    return adata

def input_data(data_location):
    adata = sc.read_h5ad(data_location)
    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    adata.raw = adata.copy()
    adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    
    true_label = np.array(adata.obs.cell_label)
    label_number = np.unique(true_label).max() - np.unique(true_label).min() + 1
    
    print('label_number:',label_number)
    print('Successfully preprocessed {} cells and {} genes.'.format(adata.n_obs, adata.n_vars))
    
    return adata

def cal_weights_via_CAN(X, num_neighbors, links=0):
    """
    Solve Problem: Clustering-with-Adaptive-Neighbors(CAN)
    :param X: d * n
    :param num_neighbors:
    :return:
    """
    size = X.shape[1]
    distances = distance(X, X)
    distances = torch.max(distances, torch.t(distances))
    sorted_distances, _ = distances.sort(dim=1)
    top_k = sorted_distances[:, num_neighbors]
    top_k = torch.t(top_k.repeat(size, 1)) + 10**-10

    sum_top_k = torch.sum(sorted_distances[:, 0:num_neighbors], dim=1)
    sum_top_k = torch.t(sum_top_k.repeat(size, 1))
    sorted_distances = None
    torch.cuda.empty_cache()
    T = top_k - distances
    distances = None
    torch.cuda.empty_cache()
    weights = torch.div(T, num_neighbors * top_k - sum_top_k)
    T = None
    top_k = None
    sum_top_k = None
    torch.cuda.empty_cache()
    weights = weights.relu().cpu()
    if links != 0:
        links = torch.Tensor(links).cuda()
        weights += torch.eye(size).cuda()
        weights += links
        weights /= weights.sum(dim=1).reshape([size, 1])
    torch.cuda.empty_cache()
    raw_weights = weights
    weights = (weights + weights.t()) / 2
    raw_weights = raw_weights.cuda()
    weights = weights.cuda()
    return weights, raw_weights

def distance(X, Y, square=True):
    """
    Compute Euclidean distances between two sets of samples
    Basic framework: pytorch
    :param X: d * n, where d is dimensions and n is number of data points in X
    :param Y: d * m, where m is number of data points in Y
    :param square: whether distances are squared, default value is True
    :return: n * m, distance matrix
    """
    n = X.shape[1]
    m = Y.shape[1]
    x = torch.norm(X, dim=0)
    x = x * x  # n * 1
    x = torch.t(x.repeat(m, 1))

    y = torch.norm(Y, dim=0)
    y = y * y  # m * 1
    y = y.repeat(n, 1)

    crossing_term = torch.t(X).matmul(Y)
    result = x + y - 2 * crossing_term
    # result = result.max(torch.zeros(result.shape).cuda())
    result = result.relu()
    if not square:
        result = torch.sqrt(result)
    # result = torch.max(result, result.t())
    return result

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def draw_fig(list,name,epoch):
    x1 = range(1, epoch+1)
    print(x1)
    y1 = list
    if name=="loss":
        plt.cla()
        plt.title('Train loss vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.savefig("./evaluation/Train_loss.png")
        plt.show()
        
    if name =="ari":
        plt.cla()
        plt.title('Train ari vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train ari', fontsize=20)
        plt.grid()
        plt.savefig("./evaluation/Train _accuracy.png")
        plt.show()
        
    elif name =="acc":
        plt.cla()
        plt.title('Train accuracy vs. epoch', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('epoch', fontsize=20)
        plt.ylabel('Train accuracy', fontsize=20)
        plt.grid()
        plt.savefig("./evaluation/Train _accuracy.png")
        plt.show()

@nb.jit(nopython=True)
def get_loc(mtx, sample_number):
    row, col = [], []
    for i in range(sample_number):
        for j in range(sample_number):
            if abs(corr[i,j]) > 0.7 :
                row.append(i)
                col.append(j)
    return row, col