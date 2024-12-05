import pandas as pd
import numpy as np

def analyze_results(excel_path="results/segmentation_results.xlsx"):
    df = pd.read_excel(excel_path)
    
    me_columns = [col for col in df.columns if col.endswith('_ME')]
    methods = [col.replace('_ME', '') for col in me_columns]
    
    stats = {
        'Method': [],
        'Mean ME': [],
        'Std Dev': [],
        'Best Count (Duplicated)': [],
        'Best Count (Unique)': []
    }
    
    min_values = df[me_columns].min(axis=1)
    
    best_counts_duplicated = {}
    for col in me_columns:
        best_counts_duplicated[col] = (df[col] == min_values).sum()
    
    best_methods_unique = df[me_columns].idxmin(axis=1)
    best_counts_unique = {col: (best_methods_unique == col).sum() for col in me_columns}
    
    for method, col in zip(methods, me_columns):
        stats['Method'].append(method)
        stats['Mean ME'].append(df[col].mean())
        stats['Std Dev'].append(df[col].std())
        stats['Best Count (Duplicated)'].append(best_counts_duplicated[col])
        stats['Best Count (Unique)'].append(best_counts_unique[col])
    
    results_df = pd.DataFrame(stats)
    
    results_df = results_df.sort_values('Mean ME')
    
    print("\nSegmentation Methods Analysis:")
    print("==============================")
    print("\nStatistics for each method:")
    print(results_df.to_string(index=False, float_format=lambda x: '{:.4f}'.format(x)))
    
    print("\nTotal images analyzed:", len(df))
    
    return results_df

analyze_results()