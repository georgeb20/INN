import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_train(curve_raw_train):
    n_trace, n_exp = curve_raw_train.shape
    assert n_exp == 48, f"Expected 48 columns, got {n_exp}"
    
    # Identify which columns are ATT vs PHS
    att_idx = []
    phs_idx = []
    num_experiments = 6  # 3 frequencies × 2 spacings
    for i_exp in range(num_experiments):
        base = i_exp * 8
        # Indices for the 4 ATT columns within this 8-col block
        att_idx.extend([base + 0, base + 2, base + 4, base + 6])
        # Indices for the 4 PHS columns within this 8-col block
        phs_idx.extend([base + 1, base + 3, base + 5, base + 7])
    
    # 1) Extract all ATT => shape (n_trace, 24)
    curve_att_train = curve_raw_train[:, att_idx]
    # 2) Extract all PHS => shape (n_trace, 24)
    curve_phs_train = curve_raw_train[:, phs_idx]
    
    # 3) Convert PHS => cos, sin
    phs_radians_train = np.deg2rad(curve_phs_train)
    phs_cos_train = np.cos(phs_radians_train)  # (n_trace, 24)
    phs_sin_train = np.sin(phs_radians_train)  # (n_trace, 24)
    
    # 4) Standard-scale ATT
    scaler_att = StandardScaler()
    curve_att_train_scaled = scaler_att.fit_transform(curve_att_train)  # (n_trace, 24)
    
    # 5) Interleave experiment by experiment
    blocks = []  # each block => (n_trace, 12)
    for i_exp in range(num_experiments):
        sub_triplets = []
        for j in range(4):
            # Index in the scaled ATT array
            att_col = curve_att_train_scaled[:, i_exp*4 + j]     # shape (n_trace,)
            cos_col = phs_cos_train[:, i_exp*4 + j]              # shape (n_trace,)
            sin_col = phs_sin_train[:, i_exp*4 + j]              # shape (n_trace,)
            # Combine them side by side => (n_trace, 3)
            triplet = np.column_stack([att_col, cos_col, sin_col])
            sub_triplets.append(triplet)
        
        # Combine the 4 triplets horizontally => (n_trace, 12)
        block = np.hstack(sub_triplets)
        blocks.append(block)
    
    # Join all 6 experiment blocks => (n_trace, 72)
    curve_train_preprocessed = np.hstack(blocks)
    
    return curve_train_preprocessed, scaler_att

def preprocess_test(curve_raw_test, scaler_att):
    n_trace, n_exp = curve_raw_test.shape    
    att_idx = []
    phs_idx = []
    num_experiments = 6
    for i_exp in range(num_experiments):
        base = i_exp * 8
        att_idx.extend([base + 0, base + 2, base + 4, base + 6])
        phs_idx.extend([base + 1, base + 3, base + 5, base + 7])
    
    # 1) Extract all ATT => (n_trace, 24)
    curve_att_test = curve_raw_test[:, att_idx]
    # 2) Extract all PHS => (n_trace, 24)
    curve_phs_test = curve_raw_test[:, phs_idx]
    
    # 3) Convert PHS => cos, sin
    phs_radians_test = np.deg2rad(curve_phs_test)
    phs_cos_test = np.cos(phs_radians_test)
    phs_sin_test = np.sin(phs_radians_test)
    
    # 4) Scale ATT using the **train-fitted** scaler
    curve_att_test_scaled = scaler_att.transform(curve_att_test)  # (n_trace, 24)
    
    # 5) Interleave experiment by experiment
    blocks = []
    for i_exp in range(num_experiments):
        sub_triplets = []
        for j in range(4):
            att_col = curve_att_test_scaled[:, i_exp*4 + j]
            cos_col = phs_cos_test[:, i_exp*4 + j]
            sin_col = phs_sin_test[:, i_exp*4 + j]
            triplet = np.column_stack([att_col, cos_col, sin_col])
            sub_triplets.append(triplet)
        block = np.hstack(sub_triplets)  # (n_trace, 12)
        blocks.append(block)
    
    curve_test_preprocessed = np.hstack(blocks)  # (n_trace, 72)
    
    return curve_test_preprocessed

def boundary_loss(x_inv, x_true):
    # x_inv, x_true shape => (batch, x_dim)
    # 1D derivative
    grad_inv = x_inv[:, 1:] - x_inv[:, :-1]
    grad_true = x_true[:, 1:] - x_true[:, :-1]
    # L1 difference in the discrete gradient
    return torch.mean(torch.abs(grad_inv - grad_true))

def split_data(Data, N_train, N_test):
    # Check if N_train and N_test together exceed the number of rows in Data
    if N_train + N_test > len(Data):
        raise ValueError("N_train + N_test exceeds the total number of rows in Data. Please choose smaller values.")
    
    # Splitting the data
    train_data = Data[:N_train]
    test_data = Data[-N_test:]
    
    return train_data, test_data

def get_dims(model, data):
    n_data = model.shape[0]  # dataset size
    x_dim = model.shape[1]  # length of resistivity model
    y_dim = data.shape[1]  # length of EM signal
    e_dim = y_dim  # length of noise variable
    z_dim = x_dim  # length of latent variable
    tot_dim = x_dim + e_dim # length of input data
    return n_data, x_dim, y_dim, e_dim, z_dim, tot_dim

def gaussian_kernel(source,target,kernel_mul = 2, kernel_num = 5, fix_sigma = None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source,target],dim = 0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)),\
                                      int(total.size(0)),\
                                      int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)),\
                                      int(total.size(0)),\
                                      int(total.size(1)))
    L2_distance = ((total0-total1) **2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data)/(n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num//2)
    bandwidth_list = [bandwidth * (kernel_mul **i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance/bandwidth_temp) for\
                 bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd(source,target,kernel_mul = 2, kernel_num = 5, fix_sigma = None):
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source,target,
                             kernel_mul = kernel_mul,
                             kernel_num = kernel_num,
                             fix_sigma = fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX+YY-XY-YX)
    return loss

def model_misfit(true_log_model, inverted_log_model, type=None):
    if type == "RMSPE":
        model_log_misfit = np.sqrt(np.mean(np.square((inverted_log_model - true_log_model)/true_log_model), axis=1)) * 100  # Shape (n_trace,)
    elif type == "Rnorm":
        model_log_misfit = np.linalg.norm((inverted_log_model - true_log_model), ord=2, axis=1) / np.linalg.norm(true_log_model, ord=2, axis=1) * 100
    elif type == "MSE":
        model_log_misfit = np.mean(np.square(inverted_log_model - true_log_model), axis=1)
    elif type == "2norm":
        model_log_misfit = np.linalg.norm((inverted_log_model - true_log_model), ord=2, axis=0)  # Shape (n_trace,)
    elif type == "r2norm":
        difference = true_log_model - inverted_log_model
        
        # Compute the R2-norm of the difference along axis 0
        misfit = np.linalg.norm(difference, ord=2, axis=0)
        
        # Compute the R2-norm of the true model along axis 0
        true_model_norm = np.linalg.norm(true_log_model, ord=2, axis=0)
        
        # Avoid division by zero by adding a small epsilon where true_model_norm is zero
        epsilon = 1e-10
        true_model_norm = np.maximum(true_model_norm, epsilon)
        
        # Compute the misfit percentage
        model_log_misfit = (misfit / true_model_norm) * 100
    else:
        raise TypeError
    return model_log_misfit

import numpy as np

def l2_norm(true_matrix, predicted_matrix, *, relative=True, eps=1e-12):
    """
    Compute the L2 (Frobenius) misfit between `predicted_matrix` and `true_matrix`.

    Parameters
    ----------
    true_matrix : np.ndarray
        Ground-truth resistivity model (2-D or 3-D array).
    predicted_matrix : np.ndarray
        Inverted/estimated resistivity model, same shape as `true_matrix`.
    relative : bool, default True
        • True  → return relative misfit in percent  
        • False → return absolute Frobenius norm (same units as input)
    eps : float, default 1e-12
        Small stabiliser to avoid division by zero.

    Returns
    -------
    float
        Scalar misfit (percent if `relative=True`, otherwise in Ω·m, Ω·ft, etc.).
    """
    # Difference between models
    diff = predicted_matrix - true_matrix

    # Frobenius norms  (np.linalg.norm(..., 'fro')  ≡ √Σ_ij a_ij² )
    diff_norm  = np.linalg.norm(diff,        ord='fro')
    if not relative:
        return diff_norm                    # Absolute misfit

    true_norm  = np.linalg.norm(true_matrix, ord='fro')
    return 100.0 * diff_norm / (true_norm + eps)

