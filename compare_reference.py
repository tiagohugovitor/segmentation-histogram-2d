import cv2
import matplotlib.pyplot as plt
from adittional_methods import li_threshold_2d, minimum_cross_entropy_2d, proposed_method_shared_T
from thresholding import otsu, otsu_gllre, kapur, kapur_gllre, proposed_method
from utils import calculate_2d_histogram, calculate_lre

def test_segmentation_methods(image_path='Eight.png'):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    methods = {
        'Otsu': otsu,
        'Otsu-GLLRE': otsu_gllre,
        'Kapur': kapur,
        'Kapur-GLLRE': kapur_gllre,
        'Proposed': proposed_method,
        'li_threshold': li_threshold_2d,
        'minimum_cross_entropy': minimum_cross_entropy_2d,
        'proposed_method_shared_T': proposed_method_shared_T,
    }
    
    results = {}
    
    lre = calculate_lre(image)
    hist_2d = calculate_2d_histogram(image, lre)

    for name, method in methods.items():

        if name in ['Otsu-GLLRE', 'Kapur-GLLRE', 'Proposed', 'li_threshold', 'minimum_cross_entropy', 'proposed_method_shared_T']:
            threshold, segmented = method(image, hist_2d)
        else:
            threshold, segmented = method(image)
            
        results[name] = {
            'threshold': threshold,
            'segmented': segmented
        }
        print(f"{name} threshold: {threshold}")
    
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    for idx, (name, result) in enumerate(results.items(), 2):
        plt.subplot(3, 3, idx)
        plt.imshow(result['segmented'], cmap='gray')
        if isinstance(result['threshold'], tuple):
            plt.title(f'{name}\nThreshold: {result["threshold"][0]},{result["threshold"][1]}')
        else:
            plt.title(f'{name}\nThreshold: {result["threshold"]}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

test_segmentation_methods()