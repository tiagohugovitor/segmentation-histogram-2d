import cv2
import pandas as pd
from utils import calculate_me, calculate_lre, calculate_2d_histogram
from adittional_methods import li_threshold_2d, minimum_cross_entropy_2d, proposed_method_shared_T
from thresholding import otsu, kapur, otsu_gllre, kapur_gllre, proposed_method
from pathlib import Path

def process_dataset(base_path="dataset-weizmann"):
    count = 0
    results_dir = Path("results")
    method_dirs = {
        'otsu': results_dir / "otsu",
        'kapur': results_dir / "kapur",
        'otsu_gllre': results_dir / "otsu_gllre",
        'kapur_gllre': results_dir / "kapur_gllre",
        'proposed': results_dir / "proposed",
        'li_threshold': results_dir / "li_threshold_2d",
        'minimum_cross_entropy': results_dir / "minimum_cross_entropy_2d",
        'proposed_method_shared_T': results_dir / "proposed_method_shared_T"
    }
    
    for dir_path in method_dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    results = []

    columns = [
        'Image',
        'Otsu_Threshold', 'Kapur_Threshold', 'Otsu_GLLRE_Threshold',
        'Kapur_GLLRE_Threshold', 'Proposed_Threshold',
        'Otsu_ME', 'Kapur_ME', 'Otsu_GLLRE_ME', 'Kapur_GLLRE_ME', 'Proposed_ME',
        'li_Threshold', 'minimum_cross_entropy_Threshold', 'proposed_method_shared_T_Threshold',
        'li_ME', 'minimum_cross_entropy_ME', 'proposed_method_shared_T_ME'
    ]
    
    excel_path = results_dir / "segmentation_results_experimental_.xlsx"
        
    original_path = Path(base_path) / "Original"
    segmented_path = Path(base_path) / "Segmented"
    
    for img_path in original_path.glob("*.png"):
        count += 1
        print(f"\nProcessing {img_path.name} - {count}")
        
        original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        ground_truth = cv2.imread(str(segmented_path / img_path.name), cv2.IMREAD_GRAYSCALE)
        
        if original is None or ground_truth is None:
            print(f"Warning: Could not read image pair: {img_path}")
            continue
            
        _, ground_truth = cv2.threshold(ground_truth, 127, 255, cv2.THRESH_BINARY)
        
        lre = calculate_lre(original)
        hist_2d = calculate_2d_histogram(original, lre)
        
        image_results = {
            'Image': img_path.name
        }
        
        li_thresh, li_seg = li_threshold_2d(original, hist_2d)
        image_results.update({
            'li_Threshold': li_thresh,
            'li_ME': calculate_me(li_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['li_threshold'] / img_path.name), li_seg)
        
        minimum_cross_entropy_thresh, minimum_cross_entropy_seg = minimum_cross_entropy_2d(original, hist_2d)
        image_results.update({
            'minimum_cross_entropy_Threshold': minimum_cross_entropy_thresh,
            'minimum_cross_entropy_ME': calculate_me(minimum_cross_entropy_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['minimum_cross_entropy'] / img_path.name), minimum_cross_entropy_seg)
        
        proposed_method_shared_T_Threshold, proposed_method_shared_T_seg = proposed_method_shared_T(original, hist_2d)
        image_results.update({
            'proposed_method_shared_T_Threshold': proposed_method_shared_T_Threshold,
            'proposed_method_shared_T_ME': calculate_me(proposed_method_shared_T_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['proposed_method_shared_T'] / img_path.name), proposed_method_shared_T_seg)
        
        otsu_thresh, otsu_seg = otsu(original)
        image_results.update({
            'Otsu_Threshold': otsu_thresh,
            'Otsu_ME': calculate_me(otsu_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['otsu'] / img_path.name), otsu_seg)
        
        kapur_thresh, kapur_seg = kapur(original)
        image_results.update({
            'Kapur_Threshold': kapur_thresh,
            'Kapur_ME': calculate_me(kapur_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['kapur'] / img_path.name), kapur_seg)
        
        otsu_gllre_thresh, otsu_gllre_seg = otsu_gllre(original, hist_2d)
        image_results.update({
            'Otsu_GLLRE_Threshold': f"{otsu_gllre_thresh}",
            'Otsu_GLLRE_ME': calculate_me(otsu_gllre_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['otsu_gllre'] / img_path.name), otsu_gllre_seg)
        
        kapur_gllre_thresh, kapur_gllre_seg = kapur_gllre(original, hist_2d)
        image_results.update({
            'Kapur_GLLRE_Threshold': f"{kapur_gllre_thresh}",
            'Kapur_GLLRE_ME': calculate_me(kapur_gllre_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['kapur_gllre'] / img_path.name), kapur_gllre_seg)
        
        proposed_thresh, proposed_seg = proposed_method(original, hist_2d)
        image_results.update({
            'Proposed_Threshold': f"{proposed_thresh}",
            'Proposed_ME': calculate_me(proposed_seg, ground_truth)
        })
        cv2.imwrite(str(method_dirs['proposed'] / img_path.name), proposed_seg)
        
        results.append(image_results)
        
        df = pd.DataFrame(results)
        df = df[columns]
        df.to_excel(excel_path, sheet_name='Results', index=False)
        
        print(f"Results for {img_path.name} - {count}:")
        print(f"  Li Threshold:             Threshold={li_thresh}, ME={image_results['li_ME']:.4f}")
        print(f"  Minimum Cross Entropy:    Threshold={minimum_cross_entropy_thresh}, ME={image_results['minimum_cross_entropy_ME']:.4f}")
        print(f"  Proposed Shared Threshold: Threshold={proposed_method_shared_T_Threshold}, ME={image_results['proposed_method_shared_T_ME']:.4f}")

    df = pd.DataFrame(results)
    print("\nAverage Misclassification Error (ME) for each method:")
    print(f"Li Threshold:             {df['li_ME'].mean():.4f}")
    print(f"Minimum Cross Entropy:    {df['minimum_cross_entropy_ME'].mean():.4f}")
    print(f"Proposed Shared Threshold: {df['proposed_method_shared_T_ME'].mean():.4f}")
    
    print(f"\nResults saved to {excel_path}")

process_dataset()
