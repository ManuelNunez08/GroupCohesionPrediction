from denseweight import DenseWeight
import numpy as np
import torch

def calculate_bin_statistics(weights, targets, num_bins=4):
    """
    Calculate the product N_i * N_w for each bin and return the standard deviation.
    """
    
    # Adjust the bin edges to ensure a more even distribution
    bin_edges = np.linspace(min(targets), max(targets), num_bins + 1)
    # Use np.digitize to assign each target to a bin
    bin_indices = np.digitize(targets, bin_edges, right=True)
    
    # Merge the first two bins together
    bin_indices[bin_indices == 0] = 1
    
    bin_stats = []
    for i in range(0, num_bins + 1):
        bin_mask = bin_indices == i
        N_i = np.sum(bin_mask)  # Number of observations in the bin
        N_w = np.sum(weights[bin_mask])  # Sum of weights in the bin
        
        if N_i > 0 and N_w > 0:
            bin_stats.append(N_i * N_w)

    # Return the standard deviation of Ni * Nw across bins
    return np.std(bin_stats) if bin_stats else float('inf')


def find_optimal_alpha(train_val_targets, data_train_val, alpha_range=(0, 2), num_bins=4):
    """
    Perform a search over the range of alpha values and minimize the standard deviation of Ni * Nw across bins.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], 100)  
    best_alpha = None
    best_std = float('inf')
    
    for alpha in alphas:
        # Fit the DenseWeight model with the current alpha
        dw = DenseWeight(alpha=alpha)
        train_val_weights = dw.fit(train_val_targets)
        
        # Assign the weights to each data point
        for i, data in enumerate(data_train_val):
            data.weight = torch.tensor([train_val_weights[i]], dtype=torch.float)
        
        # Calculate the standard deviation of Ni * Nw across bins
        std_deviation = calculate_bin_statistics(train_val_weights, train_val_targets, num_bins=num_bins)
        
        if std_deviation < best_std:
            best_std = std_deviation
            best_alpha = alpha
    
    # print(f"Best alpha: {best_alpha} with minimum std of Ni * Nw: {best_std}")
    return best_alpha
