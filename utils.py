import numpy as np
import cv2

def calculate_lre(image, window_size=3):
    image = image.astype(np.float32)
    padding = window_size // 2
    padded = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_REFLECT)
    lre = np.zeros_like(image, dtype=np.float32)
    eps = 1e-10
    
    for i in range(padding, padded.shape[0] - padding):
        for j in range(padding, padded.shape[1] - padding):
            window = padded[i-padding:i+padding+1, j-padding:j+padding+1]
            mean_val = np.mean(window)
            
            if mean_val > eps:
                rel_entropy = np.sum(window * np.abs(np.log(window / (mean_val + eps) + eps)))
                lre[i-padding, j-padding] = rel_entropy
    
    lre_min = np.min(lre)
    lre_max = np.max(lre)
    
    lre = ((lre - lre_min) / (lre_max - lre_min) * 255).astype(np.uint8)
    
    return lre

def calculate_2d_histogram(image, lre):
    hist_2d = np.zeros((256, 256), dtype=np.float32)
    
    for i, j in zip(image.ravel(), lre.ravel()):
        hist_2d[i, j] += 1
    
    hist_2d = hist_2d / (image.size)
    
    return hist_2d

def calculate_me(segmented, ground_truth):
    seg_binary = segmented > 0
    gt_binary = ground_truth > 0
    
    bo = ~gt_binary
    fo = gt_binary
    bt = ~seg_binary
    ft = seg_binary
    
    bo_bt = np.logical_and(bo, bt)
    fo_ft = np.logical_and(fo, ft)
    
    me = 1.0 - (np.sum(bo_bt) + np.sum(fo_ft)) / (np.sum(bo) + np.sum(fo))
    
    return me