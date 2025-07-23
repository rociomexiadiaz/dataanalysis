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
        'MisclassificationDetails': [],
        'F1Scores': []  # NEW: Include F1 scores
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

def plot_architecture_f1_analysis(df_dict, save_path='./'):
    """Plot architecture analysis charts with models on x-axis and skin tone as grouped bars - F1 VERSION"""

    fst_color_map = {
        1: '#F5D5A0',
        2: '#E4B589',
        3: '#D1A479',
        4: '#C0874F',
        5: '#A56635',
        6: '#4C2C27'
    }

    # Use F1Scores dataframe instead of StratifiedSensitivities
    f1_df = df_dict['F1Scores']
    f1_df = f1_df[~f1_df['Model'].str.contains('FairDisco', case=False, na=False)]

    # Replace model names
    f1_df['Model'] = f1_df['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE'
    })

    # Filter for stratified F1 scores (condition-specific, by skin tone) for 'all' datasets
    all_data = f1_df[
        (f1_df['Datasets'] == 'all') &
        (f1_df['Metric'] == 'F1 Score (Stratified)') &
        (f1_df['Condition'] != 'Overall') &  # Exclude overall F1 scores
        (pd.notna(f1_df['Skin_Tone']))  # Only stratified data has Skin_Tone
    ]

    if all_data.empty:
        print("No F1 stratified data found for all datasets.")
        return

    # Calculate average F1 score across all conditions for each model-skin tone combination
    grouped = all_data.groupby(['Model', 'Skin_Tone'])['Value'].agg(['mean', 'std']).reset_index()
    models = grouped['Model'].unique()
    skin_tones = [1,2,3,4,5,6]
    x = np.arange(len(models))
    width = 0.8 / len(skin_tones)

    plt.figure(figsize=(14, 8))
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for model in models:
            row = grouped[(grouped['Model'] == model) & (grouped['Skin_Tone'] == tone)]
            values.append(row['mean'].values[0] if not row.empty else 0)
            errors.append(row['std'].values[0] if not row.empty else 0)
        plt.bar(x + i * width, values, width, yerr=errors, capsize=4,
                label=f'Skin Tone {tone}', color=fst_color_map[tone])

    plt.xticks(x + width * (len(skin_tones) - 1) / 2, models, rotation=45, ha='right')
    plt.ylabel('Average F1 Score (%)')
    plt.title('Architecture Analysis: F1 Score by Model and Skin Tone\n(Average across all conditions)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_f1_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Max - Min) per Model for F1 scores
    pivot = grouped.pivot(index='Skin_Tone', columns='Model', values='mean')
    disparity_maxmin = pivot.max(axis=0) - pivot.min(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_maxmin.index, disparity_maxmin.values, color='#76b5c5')
    plt.ylabel('Skin Tone F1 Disparity (Max - Min)')
    plt.title('F1 Score Disparity by Architecture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_f1_disparity_maxmin_per_model.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Std) per Model for F1 scores
    disparity_std = pivot.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_std.index, disparity_std.values, color='#76b5c5')
    plt.ylabel('Skin Tone F1 Disparity (Std)')
    plt.title('F1 Score Standard Deviation by Architecture')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/architecture_f1_disparity_std_per_model.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Architecture F1 analysis and disparity plots saved!")

def plot_dataset_f1_analysis(df_dict, save_path='./'):
    """Plot dataset analysis charts with datasets on x-axis and skin tone as grouped bars - F1 VERSION"""

    fst_color_map = {
        1: '#F5D5A0',
        2: '#E4B589',
        3: '#D1A479',
        4: '#C0874F',
        5: '#A56635',
        6: '#4C2C27'
    }

    # Use F1Scores dataframe for baseline model only
    f1_df = df_dict['F1Scores']
    baseline_data = f1_df[
        (f1_df['Model'].str.contains('baseline', case=False, na=False)) &
        (f1_df['Metric'] == 'F1 Score (Stratified)') &
        (f1_df['Condition'] != 'Overall') &  # Exclude overall F1 scores
        (pd.notna(f1_df['Skin_Tone']))  # Only stratified data has Skin_Tone
    ]

    if baseline_data.empty:
        print("No baseline F1 stratified data found.")
        return

    # Calculate average F1 score across all conditions for each dataset-skin tone combination
    grouped = baseline_data.groupby(['Datasets', 'Skin_Tone'])['Value'].agg(['mean', 'std']).reset_index()
    datasets = grouped['Datasets'].unique()
    skin_tones = [1,2,3,4,5,6]
    x = np.arange(len(datasets))
    width = 0.8 / len(skin_tones)

    plt.figure(figsize=(16, 8))
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for dataset in datasets:
            row = grouped[(grouped['Datasets'] == dataset) & (grouped['Skin_Tone'] == tone)]
            values.append(row['mean'].values[0] if not row.empty else 0)
            errors.append(row['std'].values[0] if not row.empty else 0)
        plt.bar(x + i * width, values, width, yerr=errors, capsize=4,
                label=f'Skin Tone {tone}', color=fst_color_map[tone])

    plt.xticks(x + width * (len(skin_tones) - 1) / 2, datasets, rotation=45, ha='right')
    plt.ylabel('Average F1 Score (%)')
    plt.title('Dataset Analysis: F1 Score by Dataset and Skin Tone\n(Baseline model, average across all conditions)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_f1_score.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Max - Min) per Dataset for F1 scores
    pivot = grouped.pivot(index='Skin_Tone', columns='Datasets', values='mean')
    disparity_maxmin = pivot.max(axis=0) - pivot.min(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_maxmin.index, disparity_maxmin.values, color='#76b5c5')
    plt.ylabel('Skin Tone F1 Disparity (Max - Min)')
    plt.title('F1 Score Disparity by Dataset Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_f1_disparity_maxmin_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Disparity (Std) per Dataset for F1 scores
    disparity_std = pivot.std(axis=0)
    plt.figure(figsize=(10, 6))
    plt.bar(disparity_std.index, disparity_std.values, color='#76b5c5')
    plt.ylabel('Skin Tone F1 Disparity (Std)')
    plt.title('F1 Score Standard Deviation by Dataset Configuration')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{save_path}/dataset_f1_disparity_std_per_dataset.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Dataset F1 analysis and disparity plots saved!")

def plot_comparison_f1_analysis(df_dict, save_path='./'):
    """Plot comparison analysis between architecture and dataset improvements - F1 VERSION"""
    
    # Use F1Scores dataframe
    f1_df = df_dict['F1Scores']
    
    # Get control (baseline with all datasets) - average F1 across all conditions by skin tone
    control_data = f1_df[
        (f1_df['Model'].str.contains('baseline', case=False, na=False)) &
        (f1_df['Datasets'] == 'all') &
        (f1_df['Metric'] == 'F1 Score (Stratified)') &
        (f1_df['Condition'] != 'Overall') &
        (pd.notna(f1_df['Skin_Tone']))
    ]
    
    if control_data.empty:
        print("No control F1 data found. Cannot create comparison analysis.")
        print("Available models:", f1_df['Model'].unique())
        print("Available datasets:", f1_df['Datasets'].unique())
        return
    
    control_grouped = control_data.groupby('Skin_Tone')['Value'].mean()
    
    # Get best architecture improvement (highest F1 with all datasets)
    all_datasets_data = f1_df[
        (f1_df['Datasets'] == 'all') &
        (f1_df['Metric'] == 'F1 Score (Stratified)') &
        (f1_df['Condition'] != 'Overall') &
        (pd.notna(f1_df['Skin_Tone']))
    ]
    
    arch_grouped = all_datasets_data.groupby(['Model', 'Skin_Tone'])['Value'].mean().reset_index()
    arch_pivot = arch_grouped.pivot(index='Skin_Tone', columns='Model', values='Value')
    
    # Find best architecture for each skin tone
    best_arch_improvement = {}
    for tone in control_grouped.index:
        if tone in arch_pivot.index:
            best_arch_val = arch_pivot.loc[tone].max()
            best_arch_improvement[tone] = best_arch_val - control_grouped[tone]
        else:
            best_arch_improvement[tone] = 0
    
    # Get biggest dataset drop (baseline with different datasets)
    baseline_data = f1_df[
        (f1_df['Model'].str.contains('baseline', case=False, na=False)) &
        (f1_df['Metric'] == 'F1 Score (Stratified)') &
        (f1_df['Condition'] != 'Overall') &
        (pd.notna(f1_df['Skin_Tone']))
    ]
    
    dataset_grouped = baseline_data.groupby(['Datasets', 'Skin_Tone'])['Value'].mean().reset_index()
    dataset_pivot = dataset_grouped.pivot(index='Skin_Tone', columns='Datasets', values='Value')
    
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
    plt.ylabel('F1 Score (%)')
    plt.title('Architecture vs Dataset Improvement Comparison - F1 Scores\n(Average across all conditions)')
    plt.xticks(x, [f'FST {int(tone)}' for tone in skin_tones])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_path}/comparison_f1_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison F1 analysis plot saved!")

def main():
    """Main function to run all F1 analyses using multiple log files"""
    
    # Set up matplotlib for better plots
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Directory containing log files (use raw string or forward slashes for Windows)
    log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
    save_path = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneAnalysis"
    
    print("=== Skin Tone Analysis - F1 Score Version ===")
    print("Parsing all log files...")
    df_dict = parse_all_log_files(log_directory)
    
    print("Data parsing complete!")
    print(f"Available dataframes: {list(df_dict.keys())}")
    
    # Check F1Scores data
    if 'F1Scores' in df_dict and not df_dict['F1Scores'].empty:
        print(f"F1Scores shape: {df_dict['F1Scores'].shape}")
        
        # Print some sample data to verify
        f1_data = df_dict['F1Scores']
        print("\nDatasets found in F1 data:")
        print(f1_data['Datasets'].unique())
        print("\nModels found in F1 data:")
        print(f1_data['Model'].unique())
        print("\nMetrics found in F1 data:")
        print(f1_data['Metric'].unique())
        
        # Check for stratified data
        stratified_f1 = f1_data[
            (f1_data['Metric'] == 'F1 Score (Stratified)') & 
            (pd.notna(f1_data['Skin_Tone']))
        ]
        print(f"\nStratified F1 data shape: {stratified_f1.shape}")
        if not stratified_f1.empty:
            print("Skin tones found in stratified F1 data:")
            print(sorted(stratified_f1['Skin_Tone'].unique()))
    else:
        print("❌ No F1Scores data found!")
        return
    
    # Generate all F1 plots using your combined dataframes
    print("\nGenerating architecture F1 analysis plots...")
    plot_architecture_f1_analysis(df_dict, save_path)
    
    print("\nGenerating dataset F1 analysis plots...")
    plot_dataset_f1_analysis(df_dict, save_path)
    
    print("\nGenerating comparison F1 analysis plots...")
    plot_comparison_f1_analysis(df_dict, save_path)
    
    print(f"\n✅ All F1 plots saved to: {save_path}")
    print("Generated F1 score files:")
    print("- architecture_f1_score.png")
    print("- architecture_f1_disparity_maxmin_per_model.png") 
    print("- architecture_f1_disparity_std_per_model.png")
    print("- dataset_f1_score.png")
    print("- dataset_f1_disparity_maxmin_per_dataset.png")
    print("- dataset_f1_disparity_std_per_dataset.png")
    print("- comparison_f1_analysis.png")

if __name__ == "__main__":
    main()