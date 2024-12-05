import numpy as np
import cv2

def otsu(image):
    thresh, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh, binary

def otsu_gllre(image, hist_2d):
    max_variance = -np.inf
    optimal_thresh = (0, 0)
    L = 256
    eps = 1e-10
    
    J, I = np.meshgrid(np.arange(L), np.arange(L))
    
    for s in range(1, L-1):
        for t in range(1, L-1):
            P0 = np.sum(hist_2d[:s, :t])
            P1 = np.sum(hist_2d[s:, :t])
            
            if P0 < eps or P1 < eps:
                continue
                
            m0i = np.sum(I[:s, :t] * hist_2d[:s, :t]) / P0
            m0j = np.sum(J[:s, :t] * hist_2d[:s, :t]) / P0
            
            m1i = np.sum(I[s:, :t] * hist_2d[s:, :t]) / P1
            m1j = np.sum(J[s:, :t] * hist_2d[s:, :t]) / P1
            
            variance = P0 * P1 * ((m1i - m0i)**2 + (m1j - m0j)**2)
            
            if variance > max_variance:
                max_variance = variance
                optimal_thresh = (s, t)
    
    binary = np.zeros_like(image)
    binary[image > optimal_thresh[0]] = 255
    
    return optimal_thresh, binary

def kapur(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel() / image.size

    max_entropy = -np.inf
    optimal_threshold = 0
    L = 256
    eps = 1e-10

    for threshold in range(1, L-1):
        p0 = hist[:threshold].sum()
        p1 = hist[threshold:].sum()

        if p0 < eps or p1 < eps:
            continue

        h0 = hist[:threshold] / p0
        h0_entropy = -np.sum(h0 * np.log(h0 + eps))

        h1 = hist[threshold:] / p1
        h1_entropy = -np.sum(h1 * np.log(h1 + eps))

        total_entropy = h0_entropy + h1_entropy

        if total_entropy > max_entropy:
            max_entropy = total_entropy
            optimal_threshold = threshold

    binary = np.zeros_like(image)
    binary[image > optimal_threshold] = 255

    return optimal_threshold, binary

def kapur_gllre(image, hist_2d):
    max_entropy = -np.inf
    optimal_thresh = (0, 0)
    L = 256
    eps = 1e-10
    
    for s in range(1, L-1):
        for t in range(1, L-1):
            P0 = np.sum(hist_2d[:s, :t])
            P1 = np.sum(hist_2d[s:, :t])
            
            if P0 < eps or P1 < eps:
                continue
            
            region0 = hist_2d[:s, :t] / P0
            region1 = hist_2d[s:, :t] / P1
            
            H0 = -np.sum(region0 * np.log(region0 + eps))
            H1 = -np.sum(region1 * np.log(region1 + eps))
             
            total_entropy = H0 + H1
            
            if total_entropy > max_entropy:
                max_entropy = total_entropy
                optimal_thresh = (s, t)
    
    binary = np.zeros_like(image)
    binary[image > optimal_thresh[0]] = 255
    
    return optimal_thresh, binary

def proposed_method(image, hist_2d):
    min_relative_entropy = np.inf
    optimal_thresh = (0, 0)
    L = 256
    eps = 1e-10
    
    J, I = np.meshgrid(np.arange(L), np.arange(L))

    for s in range(1, L-1):
        for t in range(1, L-1):
            P0 = np.sum(hist_2d[:s, :t])
            P1 = np.sum(hist_2d[s:, :t])
            
            if P0 < eps or P1 < eps:
                continue
                
            m0i = np.sum(I[:s, :t] * hist_2d[:s, :t]) / P0
            m0j = np.sum(J[:s, :t] * hist_2d[:s, :t]) / P0
            
            m1i = np.sum(I[s:, :t] * hist_2d[s:, :t]) / P1
            m1j = np.sum(J[s:, :t] * hist_2d[s:, :t]) / P1
            
            D0 = np.sum(I[:s, :t] * hist_2d[:s, :t] * np.log((I[:s, :t] + eps)/(m0i + eps)) + 
                       J[:s, :t] * hist_2d[:s, :t] * np.log((J[:s, :t] + eps)/(m0j + eps)))
            
            D1 = np.sum(I[s:, :t] * hist_2d[s:, :t] * np.log((I[s:, :t] + eps)/(m1i + eps)) + 
                       J[s:, :t] * hist_2d[s:, :t] * np.log((J[s:, :t] + eps)/(m1j + eps)))
            
            D = D0 + D1
            
            if D < min_relative_entropy:
                min_relative_entropy = D
                optimal_thresh = (s, t)
    
    binary = np.zeros_like(image)
    binary[image > optimal_thresh[0]] = 255
    
    return optimal_thresh, binary
