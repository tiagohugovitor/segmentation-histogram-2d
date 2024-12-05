import numpy as np

def minimum_cross_entropy_2d(image, hist_2d):
    rows, cols = hist_2d.shape
    min_entropy = float('inf')
    optimal_s = 0
    optimal_t = 0
    eps = 1e-10
    
    if hist_2d.sum() != 1.0:
        hist_2d = hist_2d / hist_2d.sum()
    
    for s in range(1, rows-1):
        for t in range(1, cols-1):
            P0 = np.sum(hist_2d[:s, :t])
            P1 = np.sum(hist_2d[s:, :t])
            
            if P0 < eps or P1 < eps:
                continue
                
            i_coords, j_coords = np.meshgrid(np.arange(s), np.arange(t), indexing='ij')
            m0i = np.sum(i_coords * hist_2d[:s, :t]) / P0
            m0j = np.sum(j_coords * hist_2d[:s, :t]) / P0
            
            i_coords, j_coords = np.meshgrid(np.arange(s, rows), np.arange(t), indexing='ij')
            m1i = np.sum((i_coords-s) * hist_2d[s:, :t]) / P1
            m1j = np.sum(j_coords * hist_2d[s:, :t]) / P1
            
            cross_entropy = 0
            
            valid_mask = hist_2d[:s, :t] > eps
            if np.any(valid_mask):
                i_vals = np.arange(s)[:, np.newaxis]
                j_vals = np.arange(t)[np.newaxis, :]
                
                cross_entropy += np.sum(
                    i_vals * hist_2d[:s, :t] * np.log((i_vals + eps) / (m0i + eps)) +
                    j_vals * hist_2d[:s, :t] * np.log((j_vals + eps) / (m0j + eps))
                )
            
            valid_mask = hist_2d[s:, :t] > eps
            if np.any(valid_mask):
                i_vals = np.arange(s, rows)[:, np.newaxis]
                j_vals = np.arange(t)[np.newaxis, :]
                
                cross_entropy += np.sum(
                    i_vals * hist_2d[s:, :t] * np.log((i_vals + eps) / (m1i + eps)) +
                    j_vals * hist_2d[s:, :t] * np.log((j_vals + eps) / (m1j + eps))
                )
            
            if cross_entropy < min_entropy:
                min_entropy = cross_entropy
                optimal_s = s
                optimal_t = t
    
    binary_image = np.zeros_like(image)
    binary_image[image > optimal_s] = 255
    
    return (optimal_s, optimal_t), binary_image

def calculate_kapur_t(hist_2d):
    L = 256
    max_entropy = -np.inf
    optimal_t = 0
    eps = 1e-10

    for t in range(1, L-1):
        P_low = np.sum(hist_2d[:, :t])
        P_high = np.sum(hist_2d[:, t:])
    
        if P_low < eps or P_high < eps:
            continue
    
        prob_low = hist_2d[:, :t] / P_low
        prob_high = hist_2d[:, t:] / P_high
    
        H_low = -np.sum(prob_low * np.log(prob_low + eps))
        H_high = -np.sum(prob_high * np.log(prob_high + eps))

        total_entropy = P_low * H_low + P_high * H_high
    
        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_t = t
        
    return optimal_t

def proposed_method_shared_T(image, hist_2d):
    t = calculate_kapur_t(hist_2d)

    min_relative_entropy = np.inf
    optimal_s = 0
    L = 256
    eps = 1e-10

    I, J = np.meshgrid(np.arange(L), np.arange(L))
    I = I.T
    J = J.T

    for s in range(1, L-1):
        P0 = np.sum(hist_2d[:s, :t])
        P1 = np.sum(hist_2d[s:, :t])
    
        if P0 < eps or P1 < eps:
            continue
    
        m0i = np.sum(I[:s, :t] * hist_2d[:s, :t]) / P0
        m0j = np.sum(J[:s, :t] * hist_2d[:s, :t]) / P0
    
        m1i = np.sum(I[s:, :t] * hist_2d[s:, :t]) / P1
        m1j = np.sum(J[s:, :t] * hist_2d[s:, :t]) / P1
    
        D = 0
    
        valid0 = hist_2d[:s, :t] > eps
        if np.any(valid0):
            D += np.sum(
                I[:s, :t][valid0] * hist_2d[:s, :t][valid0] * np.log((I[:s, :t][valid0] + eps) / (m0i + eps)) +
                J[:s, :t][valid0] * hist_2d[:s, :t][valid0] * np.log((J[:s, :t][valid0] + eps) / (m0j + eps))
            )
    
        valid1 = hist_2d[s:, :t] > eps
        if np.any(valid1):
            D += np.sum(
                I[s:, :t][valid1] * hist_2d[s:, :t][valid1] * np.log((I[s:, :t][valid1] + eps) / (m1i + eps)) +
                J[s:, :t][valid1] * hist_2d[s:, :t][valid1] * np.log((J[s:, :t][valid1] + eps) / (m1j + eps))
            )
    
        if D < min_relative_entropy and not np.isnan(D):
            min_relative_entropy = D
            optimal_s = s

    binary = np.zeros_like(image)
    binary[image > optimal_s] = 255

    return (optimal_s, t), binary


def li_threshold_2d(image, hist_2d, max_iter=100, tol=1e-5):
    optimal_thresh = (0, 0)
    L = 256
    eps = 1e-10
    
    J, I = np.meshgrid(np.arange(L), np.arange(L))

    s = L // 2 
    t = L // 2
    for _ in range(max_iter):
        P0 = np.sum(hist_2d[:s, :t])
        P1 = np.sum(hist_2d[s:, :t])
        
        if P0 < eps or P1 < eps:
            continue
        
        m0_i = np.sum(I[:s, :t] * hist_2d[:s, :t]) / P0
        m0_j = np.sum(J[:s, :t] * hist_2d[:s, :t]) / P0

        m1_i = np.sum(I[s:, :t] * hist_2d[s:, :t]) / P1
        m1_j = np.sum(J[s:, :t] * hist_2d[s:, :t]) / P1

        new_s = (m0_i * P0 + m1_i * P1) / (P0 + P1)
        new_t = (m0_j * P0 + m1_j * P1) / (P0 + P1)

        if abs(new_s - s) < tol and abs(new_t - t) < tol:
            break

        s, t = int(new_s), int(new_t)
        optimal_thresh = (s, t)

    binary = np.zeros_like(image)
    binary[image > optimal_thresh[0]] = 255
    
    return optimal_thresh, binary