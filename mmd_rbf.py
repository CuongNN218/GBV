import torch
import numpy as np
#from sklearn.utils import shuffle


def rbf_mmd2(X, Y, sigma_list=[1, 2, 5, 10], biased=True, device=torch.device('cuda')):
    """
    Computes squared MMD using a RBF kernel.
    
    Args:
        X, Y (Tensor): datasets that MMD is computed on
        sigma (float): lengthscale of the RBF kernel
        biased (bool): whether to compute a biased mean
        
    Return:
        MMD squared
    """

    if len(X.shape) > 2:
        X = X.view(len(X), -1)

    if len(Y.shape) > 2:
        Y = Y.view(len(Y), -1)

    X = X.to(device)
    Y = Y.to(device)
    
    XX = torch.matmul(X, X.T)
    XY = torch.matmul(X, Y.T)
    YY = torch.matmul(Y, Y.T)
    
    X_sqnorms = torch.diagonal(XX)
    Y_sqnorms = torch.diagonal(YY)
    
    assert len(sigma_list) > 0

    K_XYs, K_XXs, K_YYs = [], [], []
    for sigma in sigma_list:
        gamma = 1 / (2 * sigma**2)
        
        K_XY = torch.exp(-gamma * (
                -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        K_XX = torch.exp(-gamma * (
                -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
        K_YY = torch.exp(-gamma * (
                -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
        
        K_XXs.append(K_XX)
        K_XYs.append(K_XY)
        K_YYs.append(K_YY)

    K_XY = torch.stack(K_XYs).sum(dim=0)
    K_XX = torch.stack(K_XXs).sum(dim=0)
    K_YY = torch.stack(K_YYs).sum(dim=0)
    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return mmd2


def batched_rbf_mmd2(X, Y, sigma_list=[1, 2, 5, 10], biased=True, device=torch.device('cuda'), batch_size=128):
    print("Shape: ", X.shape, Y.shape) 

    X_batches = torch.split(X, batch_size)
    Y_batches = torch.split(Y, batch_size)
    print(f"Number of batches: {len(X_batches)}, {len(Y_batches)} at size {batch_size}")
    
    overall_mmd2 = 0
    for i, (x, y) in enumerate(zip(X_batches, Y_batches)):
        batch_mmd2 = rbf_mmd2(x, y, sigma_list=sigma_list, biased=True, device=device)
        overall_mmd2 += batch_mmd2 / batch_size

    return -1.0 * torch.sqrt(overall_mmd2)


def get_MMD_values_uneven(D_Xs, V_X, device=torch.device('cuda'), sample_size=None, squared=False, batch_size=1024, sigma_list = [1, 2, 5, 10],):
    results = []
    V_X = V_X.to(device)
    rand_indx = torch.randperm(len(V_X))
    permuted_V_X = V_X[rand_indx]

    for D_X in D_Xs:
        D_X = D_X[torch.randperm(len(D_X))]
        if sample_size is not None:
            permuted_V_X = permuted_V_X[:sample_size]
            D_X = D_X[:sample_size] 
        min_len = min(len(permuted_V_X), len(D_X))
        print(min_len)
        MMD2 = batched_rbf_mmd2(D_X[:min_len], permuted_V_X[:min_len], sigma_list, device=device, batch_size=batch_size)
        if squared:
            results.append(-MMD2.item())
        else:
            results.append(-torch.sqrt(max(1e-6, MMD2)).item())
    return results 
