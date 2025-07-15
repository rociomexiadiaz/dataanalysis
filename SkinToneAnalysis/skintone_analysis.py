import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add the parent directory to the Python path to import dataframe module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

def parse_all_log_files(log_directory='./'):
    """Parse all log files and combine into a single dataset"""
    
    log_files = [
        'combined_all.txt',
        'combined_minus_dermie.txt', 
        'combined_minus_fitz.txt',
        'combined_minus_india.txt',
        'combined_minus_pad.txt',
        'combined_minus_scin.txt'
    ]
    
    # Map filenames to dataset names
    dataset_mapping = {
        'combined_all.txt': 'all',
        'combined_minus_dermie.txt': 'minus_dermie',
        'combined_minus_fitz.txt': 'minus_fitz',
        'combined_minus_india.txt': 'minus_india',
        'combined_minus_pad.txt': 'minus_pad',
        'combined_minus_scin.txt': 'minus_scin'
    }
    
    combined_data = {
        'AverageSensitivities': [],
        'ConditionSensitivities': [],
        'SkinToneAccuracies': [],
        'StratifiedSensitivities': [],
        'MisclassifiedCounts': [],
        'MisclassificationDetails': []
    }
    
    for log_file in log_files:
        file_path = os.path.join(log_directory, log_file)
        
        if not os.path.exists(file_path):
            print(f"Warning: {file_path} not found, skipping...")
            continue
            
        print(f"Parsing {log_file}...")
        
        # Parse the individual log file
        df_dict = parse_combined_log(file_path)
        
        # Add dataset info to each dataframe and append to combined data
        for key in combined_data.keys():
            if key in df_dict and not df_dict[key].empty:
                # Update the Datasets column to reflect the actual dataset combination
                df_copy = df_dict[key].copy()
                # Use just the filename, not the full path for mapping
                df_copy['Datasets'] = dataset_mapping[log_file]
                combined_data[key].append(df_copy)
    
    # Combine all dataframes
    final_data = {}
    for key in combined_data.keys():
        if combined_data[key]:
            final_data[key] = pd.concat(combined_data[key], ignore_index=True)
        else:
            final_data[key] = pd.DataFrame()
    
    return final_data

def plot_architecture_analysis(df_dict, save_path='./'):
    """Plot architecture analysis charts with models on x-axis and skin tone as grouped bars"""

    fst_color_map = {
        1: '#F5D5A0',
        2: '#E4B589',
        3: '#D1A479',
        4: '#C0874F',
        5: '#A56635',
        6: '#4C2C27'
    }

    stratified_df = df_dict['StratifiedSensitivities']
    stratified_df = stratified_df[~stratified_df['Model'].str.contains('FairDisco', case=False, na=False)]

    # Replace model names
    stratified_df['Model'] = stratified_df['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE'
    })


    all_data = stratified_df[
        (stratified_df['Datasets'] == 'all') &
        (stratified_df['Metric'] == 'Top-1 Sensitivity')
    ]

    if all_data.empty:
        print("No data found for all datasets.")
        return

    grouped = all_data.groupby(['Model', 'Skin Tone'])['Value'].agg(['mean', 'std']).reset_index()
    models = grouped['Model'].unique()
    skin_tones = [1,2,3,4,5,6]
    x = np.arange(len(models))
    width = 0.8 / len(skin_tones)

    plt.figure(figsize=(14, 8))
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for model in models:
            row = grouped[(grouped['Model'] == model) & (grouped['Skin Tone'] == tone)]
            values.append(row['mean'].values[0] if not row.empty else 0)
            errors.append(row['std'].values[0] if not row.empty else 0)
        plt.bar(x + i * width, values, width, yerr=errors, capsize=4,
                label=f'Skin Tone {tone}', color=fst_color_map[tone])

    plt.xticks(x + width * (len(skin_tones) - 1) / 2, models, rotation=45, ha='right')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Max - Min) per Model
    pivot = grouped.pivot(index='Skin Tone', columns='Model', values='mean')
    disparity_maxmin = pivot.max(axis=0) - pivot.min(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_maxmin.index, disparity_maxmin.values, color='#76b5c5')
    plt.ylabel('Skin Tone Disparity (Max - Min)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_disparity_maxmin_per_model.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Std) per Model
    disparity_std = pivot.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_std.index, disparity_std.values, color='#76b5c5')
    plt.ylabel('Skin Tone Disparity (Std)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_disparity_std_per_model.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Architecture analysis and disparity plots saved!")

def plot_dataset_analysis(df_dict, save_path='./'):
    """Plot dataset analysis charts with datasets on x-axis and skin tone as grouped bars"""

    fst_color_map = {
        1: '#F5D5A0',
        2: '#E4B589',
        3: '#D1A479',
        4: '#C0874F',
        5: '#A56635',
        6: '#4C2C27'
    }

    stratified_df = df_dict['StratifiedSensitivities']
    baseline_data = stratified_df[
        (stratified_df['Model'].str.contains('baseline', case=False, na=False)) &
        (stratified_df['Metric'] == 'Top-1 Sensitivity')
    ]

    if baseline_data.empty:
        print("No baseline data found.")
        return

    grouped = baseline_data.groupby(['Datasets', 'Skin Tone'])['Value'].agg(['mean', 'std']).reset_index()
    datasets = grouped['Datasets'].unique()
    skin_tones = [1,2,3,4,5,6]
    x = np.arange(len(datasets))
    width = 0.8 / len(skin_tones)

    plt.figure(figsize=(16, 8))
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for dataset in datasets:
            row = grouped[(grouped['Datasets'] == dataset) & (grouped['Skin Tone'] == tone)]
            values.append(row['mean'].values[0] if not row.empty else 0)
            errors.append(row['std'].values[0] if not row.empty else 0)
        plt.bar(x + i * width, values, width, yerr=errors, capsize=4,
                label=f'Skin Tone {tone}', color=fst_color_map[tone])

    plt.xticks(x + width * (len(skin_tones) - 1) / 2, datasets, rotation=45, ha='right')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Max - Min) per Dataset
    pivot = grouped.pivot(index='Skin Tone', columns='Datasets', values='mean')
    disparity_maxmin = pivot.max(axis=0) - pivot.min(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_maxmin.index, disparity_maxmin.values, color='#76b5c5')
    plt.ylabel('Skin Tone Disparity (Max - Min)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_disparity_maxmin_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Std) per Dataset
    disparity_std = pivot.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_std.index, disparity_std.values, color='#76b5c5')
    plt.ylabel('Skin Tone Disparity (Std)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_disparity_std_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Dataset analysis and disparity plots saved!")



def plot_comparison_analysis(df_dict, save_path='./'):
    """Plot comparison analysis between architecture and dataset improvements"""
    
    # Use your StratifiedSensitivities dataframe
    stratified_df = df_dict['StratifiedSensitivities']
    
    # Get control (baseline with all datasets)
    control_data = stratified_df[
        (stratified_df['Model'].str.contains('baseline', case=False, na=False)) &
        (stratified_df['Datasets'] == 'all') &
        (stratified_df['Metric'] == 'Top-1 Sensitivity')
    ]
    
    if control_data.empty:
        print("No control data found. Cannot create comparison analysis.")
        print("Available models:", stratified_df['Model'].unique())
        print("Available datasets:", stratified_df['Datasets'].unique())
        return
    
    control_grouped = control_data.groupby('Skin Tone')['Value'].mean()
    
    # Get best architecture improvement (highest sensitivity with all datasets)
    all_datasets_data = stratified_df[
        (stratified_df['Datasets'] == 'all') &
        (stratified_df['Metric'] == 'Top-1 Sensitivity')
    ]
    
    arch_grouped = all_datasets_data.groupby(['Model', 'Skin Tone'])['Value'].mean().reset_index()
    arch_pivot = arch_grouped.pivot(index='Skin Tone', columns='Model', values='Value')
    
    # Find best architecture for each skin tone
    best_arch_improvement = {}
    for tone in control_grouped.index:
        if tone in arch_pivot.index:
            best_arch_val = arch_pivot.loc[tone].max()
            best_arch_improvement[tone] = best_arch_val - control_grouped[tone]
        else:
            best_arch_improvement[tone] = 0
    
    # Get biggest dataset drop (baseline with different datasets)
    baseline_data = stratified_df[
        (stratified_df['Model'].str.contains('baseline', case=False, na=False)) &
        (stratified_df['Metric'] == 'Top-1 Sensitivity')
    ]
    
    dataset_grouped = baseline_data.groupby(['Datasets', 'Skin Tone'])['Value'].mean().reset_index()
    dataset_pivot = dataset_grouped.pivot(index='Skin Tone', columns='Datasets', values='Value')
    
    # Find biggest dataset drop for each skin tone
    biggest_dataset_drop = {}
    for tone in control_grouped.index:
        if tone in dataset_pivot.index:
            min_dataset_val = dataset_pivot.loc[tone].min()
            biggest_dataset_drop[tone] = control_grouped[tone] - min_dataset_val
        else:
            biggest_dataset_drop[tone] = 0
    
    # Create comparison plot (3x6 bars)
    skin_tones = sorted(list(control_grouped.index))
    control_vals = [control_grouped[tone] for tone in skin_tones]
    arch_improvements = [best_arch_improvement[tone] for tone in skin_tones]
    dataset_drops = [biggest_dataset_drop[tone] for tone in skin_tones]
    
    plt.figure(figsize=(12, 8))
    x = np.arange(len(skin_tones))
    width = 0.25
    
    plt.bar(x - width, control_vals, width, label='Control (Baseline)', color='#1f77b4')
    plt.bar(x, arch_improvements, width, label='Best Architecture Improvement', color='#ff7f0e')
    plt.bar(x + width, dataset_drops, width, label='Biggest Dataset Drop', color='#2ca02c')
    
    plt.xlabel('Skin Tone')
    plt.ylabel('Sensitivity (%)')
    plt.title('Architecture vs Dataset Improvement Comparison')
    plt.xticks(x, skin_tones)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison analysis plot saved!")

def main():
    """Main function to run all analyses using multiple log files"""
    
    # Set up matplotlib for better plots
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Directory containing log files (use raw string or forward slashes for Windows)
    log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
    save_path = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneAnalysis"
    
    print("Parsing all log files...")
    df_dict = parse_all_log_files(log_directory)
    
    print("Data parsing complete!")
    print(f"Available dataframes: {list(df_dict.keys())}")
    print(f"StratifiedSensitivities shape: {df_dict['StratifiedSensitivities'].shape}")
    
    # Print some sample data to verify
    print("\nDatasets found:")
    print(df_dict['StratifiedSensitivities']['Datasets'].unique())
    print("\nModels found:")
    print(df_dict['StratifiedSensitivities']['Model'].unique())
    print("\nSkin tones found:")
    print(sorted(df_dict['StratifiedSensitivities']['Skin Tone'].unique()))
    
    # Generate all plots using your combined dataframes
    print("\nGenerating architecture analysis plots...")
    plot_architecture_analysis(df_dict, save_path)
    
    print("\nGenerating dataset analysis plots...")
    plot_dataset_analysis(df_dict, save_path)
    
    print("\nGenerating comparison analysis plots...")
    plot_comparison_analysis(df_dict, save_path)
    
    print("\nAll plots saved successfully!")

if __name__ == "__main__":
    main()