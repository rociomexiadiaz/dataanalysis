import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import glob

# Add parent directory for import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

# Configuration
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneAnalysis"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Data source file for baseline dataset comparison
DATASET_COMPARISON_FILE = 'combined_baseline_datasets.txt'

# FST color mapping
fst_color_map = {
    1.0: '#F5D5A0',
    2.0: '#E4B589',
    3.0: '#D1A479',
    4.0: '#C0874F',
    5.0: '#A56635',
    6.0: '#4C2C27'
}

# ================================================================
# MODIFIED UTILITY FUNCTIONS FROM overall2.py
# ================================================================

def find_log_files(log_directory, base_pattern='combined_all'):
    """
    Find all log files matching the base pattern with optional suffixes
    e.g., combined_all.txt, combined_all_0.txt, combined_all_1.txt, etc.
    """
    pattern = os.path.join(log_directory, f"{base_pattern}*.txt")
    files = glob.glob(pattern)
    files.sort()
    
    print(f"Found {len(files)} log files matching '{base_pattern}*':")
    for f in files:
        print(f"  {os.path.basename(f)}")
    
    return files

def parse_multiple_logs(log_files, parser_func, source_name, split_offset=0):
    """
    Parse multiple log files using a specified parser function and combine results.
    Adds 'Split' and 'Source' identifiers.
    """
    all_data = []
    
    for i, log_file in enumerate(log_files):
        print(f"\nParsing {os.path.basename(log_file)}...")
        
        try:
            df_dict = parser_func(log_file)
            
            # Add split and source identifier to each dataframe
            processed_dict = {}
            for key, df in df_dict.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['Split'] = i + split_offset
                    df_copy['LogFile'] = os.path.basename(log_file)
                    df_copy['Source'] = source_name
                    processed_dict[key] = df_copy
                else:
                    processed_dict[key] = df
            
            all_data.append(processed_dict)
            print(f"  Successfully parsed {len([k for k, v in processed_dict.items() if not v.empty])} data types")
            
        except Exception as e:
            print(f"  Error parsing {log_file}: {e}")
            continue
    
    return all_data

def combine_data_with_stats(all_data):
    """
    Combine data from multiple splits and compute statistics.
    """
    combined_results = {}
    
    data_types = set()
    for data_dict in all_data:
        data_types.update(data_dict.keys())
    
    for data_type in data_types:
        dfs = []
        for data_dict in all_data:
            if data_type in data_dict and not data_dict[data_type].empty:
                dfs.append(data_dict[data_type])
        
        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            combined_results[data_type] = combined_df
        else:
            combined_results[data_type] = pd.DataFrame()
    
    return combined_results

# ================================================================
# ORIGINAL PARSING FUNCTIONS (MODIFIED TO USE VALUE/METRIC COLUMNS)
# ================================================================

def parse_stratified_f1_scores_from_log(filepath):
    """Parse stratified F1 scores per skin tone from log files"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    if 'Dataset Combination:' in content:
        model_blocks = re.split(r'(?=Python Filename:)', content)
        f1_scores = []
        for block in model_blocks:
            if not block.strip(): continue
            dataset_match = re.search(r'Dataset Combination:\s*(\S+)', block)
            if not dataset_match: continue
            dataset_config = dataset_match.group(1)
            stratified_f1_pattern = r'Skin Tone:\s*([\d.]+)\n.*?Per Condition:\n(.*?)(?=\nSkin Tone:|\n\nNumber|\n===|$)'
            stratified_f1_matches = re.findall(stratified_f1_pattern, block, re.DOTALL)
            for skin_tone, conditions_text in stratified_f1_matches:
                condition_f1_matches = re.findall(r'(\w+):\s*([\d.]+)%', conditions_text)
                for condition, f1_score in condition_f1_matches:
                    f1_scores.append({
                        'Model': 'Baseline',
                        'Dataset_Config': dataset_config,
                        'Skin Tone': float(skin_tone),
                        'Condition': condition,
                        'Metric': 'F1 Score',
                        'Value': float(f1_score)
                    })
        return pd.DataFrame(f1_scores)
    else:
        model_blocks = re.split(r'(?=Python Filename:)', content)
        f1_scores = []
        for block in model_blocks:
            if not block.strip(): continue
            model_match = re.search(r'Python Filename:\s*(\S+)', block)
            if not model_match: continue
            model = model_match.group(1).replace('.py', '').replace('train_', '')
            if 'fairdisco' in model.lower(): continue
            stratified_f1_pattern = r'Skin Tone:\s*([\d.]+)\n.*?Per Condition:\n(.*?)(?=\nSkin Tone:|\n\nNumber|\n===|$)'
            stratified_f1_matches = re.findall(stratified_f1_pattern, block, re.DOTALL)
            for skin_tone, conditions_text in stratified_f1_matches:
                condition_f1_matches = re.findall(r'(\w+):\s*([\d.]+)%', conditions_text)
                for condition, f1_score in condition_f1_matches:
                    f1_scores.append({
                        'Model': model,
                        'Dataset_Config': 'All Datasets',
                        'Skin Tone': float(skin_tone),
                        'Condition': condition,
                        'Metric': 'F1 Score',
                        'Value': float(f1_score)
                    })
        return pd.DataFrame(f1_scores)

def parse_stratified_sensitivity_with_dataset_config(filepath):
    """Parse stratified sensitivity data from baseline datasets file"""
    with open(filepath, 'r') as file:
        content = file.read()
    model_blocks = re.split(r'(?=Python Filename:)', content)
    sensitivity_data = []
    for block in model_blocks:
        if not block.strip(): continue
        dataset_match = re.search(r'Dataset Combination:\s*(\S+)', block)
        if not dataset_match: continue
        dataset_config = dataset_match.group(1)
        skin_tone_sections = re.findall(r'Skin Tone:\s*([\d.]+)\n(.*?)(?=\nSkin Tone:|\n\n===|\nNumber|$)', block, re.DOTALL)
        for skin_tone, section_content in skin_tone_sections:
            condition_matches = re.findall(r'Condition:\s*(.+?), Top-1 Sensitivity:\s*([\d.]+)%', section_content)
            for condition, sensitivity in condition_matches:
                sensitivity_data.append({
                    'Model': 'Baseline',
                    'Dataset_Config': dataset_config,
                    'Skin Tone': float(skin_tone),
                    'Condition': condition,
                    'Metric': 'Top-1 Sensitivity',
                    'Value': float(sensitivity)
                })
    return pd.DataFrame(sensitivity_data)

def parse_baseline_datasets_log(filepath):
    """
    A wrapper to parse both sensitivity and F1 data from the
    combined_baseline_datasets.txt file.
    """
    sensitivity_df = parse_stratified_sensitivity_with_dataset_config(filepath)
    f1_df = parse_stratified_f1_scores_from_log(filepath)
    return {
        'StratifiedSensitivities': sensitivity_df,
        'F1Scores': f1_df
    }

def clean_model_names(df, column='Model'):
    """Clean and standardize model names"""
    model_mapping = {
        'Baseline': 'Baseline', 'VAE': 'VAE', 'TABE': 'TABE',
        'train_Baseline': 'Baseline', 'train_VAE': 'VAE', 'train_TABE': 'TABE'
    }
    df[column] = df[column].replace(model_mapping)
    return df

# ================================================================
# REFACRORED load_skin_tone_data()
# ================================================================

def load_skin_tone_data():
    """Load stratified sensitivity and F1 score data from multiple log files."""

    print("=== Loading and Combining Data ===")

    # Find and parse all model comparison files
    model_comparison_files = find_log_files(log_directory, 'combined_all')
    if not model_comparison_files:
        print("ERROR: No 'combined_all*.txt' files found!")
        # Return empty dataframes in case of no files
        return pd.DataFrame(), pd.DataFrame()
    
    # We will use parse_combined_log to get all dataframes, then
    # manually filter for the ones we care about.
    all_models_data_dicts = parse_multiple_logs(model_comparison_files, parse_combined_log, 'all_models')
    
    # Parse the F1 and Sensitivity data from these logs
    model_sensitivity_dfs = [d['StratifiedSensitivities'] for d in all_models_data_dicts if 'StratifiedSensitivities' in d and not d['StratifiedSensitivities'].empty]
    model_f1_dfs = [d['F1Scores'] for d in all_models_data_dicts if 'F1Scores' in d and not d['F1Scores'].empty]
    
    model_sensitivity_data = pd.concat(model_sensitivity_dfs, ignore_index=True) if model_sensitivity_dfs else pd.DataFrame()
    model_f1_data = pd.concat(model_f1_dfs, ignore_index=True) if model_f1_dfs else pd.DataFrame()

    if not model_sensitivity_data.empty:
        model_sensitivity_data['Dataset_Config'] = 'All Datasets'
    if not model_f1_data.empty:
        model_f1_data['Dataset_Config'] = 'All Datasets'

    print(f"Loaded {len(model_comparison_files)} model comparison splits.")
    
    # Load dataset comparison data from a single file
    dataset_comparison_path = os.path.join(log_directory, DATASET_COMPARISON_FILE)
    if not os.path.exists(dataset_comparison_path):
        print(f"WARNING: {dataset_comparison_path} not found. Skipping dataset comparison.")
        dataset_data_dicts = []
    else:
        dataset_data_dicts = parse_multiple_logs(
            [dataset_comparison_path], parse_baseline_datasets_log, 
            'baseline_datasets', split_offset=len(model_comparison_files)
        )
    
    dataset_sensitivity_dfs = [d['StratifiedSensitivities'] for d in dataset_data_dicts if 'StratifiedSensitivities' in d and not d['StratifiedSensitivities'].empty]
    dataset_f1_dfs = [d['F1Scores'] for d in dataset_data_dicts if 'F1Scores' in d and not d['F1Scores'].empty]
    
    dataset_sensitivity_data = pd.concat(dataset_sensitivity_dfs, ignore_index=True) if dataset_sensitivity_dfs else pd.DataFrame()
    dataset_f1_data = pd.concat(dataset_f1_dfs, ignore_index=True) if dataset_f1_dfs else pd.DataFrame()

    print(f"Loaded {len(dataset_data_dicts)} dataset comparison splits.")

    # Combine model and dataset data
    combined_sensitivity = pd.concat([model_sensitivity_data, dataset_sensitivity_data], ignore_index=True)
    combined_f1 = pd.concat([model_f1_data, dataset_f1_data], ignore_index=True)
    
    # Clean up and normalize
    combined_sensitivity = combined_sensitivity[combined_sensitivity['Model'] != 'train_FairDisCo'].copy()
    combined_f1 = combined_f1[combined_f1['Model'] != 'train_FairDisCo'].copy()
    
    if len(combined_sensitivity) > 0: combined_sensitivity = clean_model_names(combined_sensitivity)
    if len(combined_f1) > 0: combined_f1 = clean_model_names(combined_f1)

    print(f"Final data summary:")
    print(f"  Total stratified sensitivity records: {len(combined_sensitivity)}")
    print(f"  Total stratified F1 records: {len(combined_f1)}")
    print(f"  Total unique splits: {combined_sensitivity['Split'].nunique() if 'Split' in combined_sensitivity.columns else 1}")

    return combined_sensitivity, combined_f1

# ================================================================
# MODIFIED PLOTTING FUNCTIONS
# ================================================================

def create_model_skin_tone_charts(sensitivity_data, f1_data, save_path):
    """Create charts showing performance by model across skin tones with disparity subplots"""
    
    sens_model_data = sensitivity_data[sensitivity_data['Dataset_Config'] == 'All Datasets'].copy()
    f1_model_data = f1_data[f1_data['Dataset_Config'] == 'All Datasets'].copy()
    
    if sens_model_data.empty and f1_model_data.empty:
        print("No model comparison data found for plotting.")
        return

    # 1. F1 Score by Model across Skin Tones with Disparity
    print("Creating F1 score by model across skin tones charts...")
    if not f1_model_data.empty:
        f1_metrics = f1_model_data['Metric'].unique()
        for metric in f1_metrics:
            if 'F1' in metric:
                print(f"  - Plotting for metric: {metric}")
                f1_plot_data = f1_model_data[f1_model_data['Metric'] == metric].copy()
                
                has_splits = 'Split' in f1_plot_data.columns and f1_plot_data['Split'].nunique() > 1
                
                grouped_f1 = f1_plot_data.groupby(['Model', 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
                grouped_f1['stderr'] = grouped_f1['std'] / np.sqrt(grouped_f1['count'])
                grouped_f1['stderr'] = grouped_f1['stderr'].fillna(0)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                models = sorted(grouped_f1['Model'].unique())
                skin_tones = sorted(grouped_f1['Skin Tone'].unique())
                x = np.arange(len(models))
                width = 0.8 / len(skin_tones)

                for i, fst in enumerate(skin_tones):
                    values = []
                    errors = []
                    for model in models:
                        row = grouped_f1[(grouped_f1['Model'] == model) & (grouped_f1['Skin Tone'] == fst)]
                        if not row.empty:
                            values.append(row['mean'].values[0])
                            errors.append(row['stderr'].values[0] if has_splits else 0)
                        else:
                            values.append(0)
                            errors.append(0)
                    
                    bars = ax1.bar(x + i * width, values, width,
                                   yerr=errors if has_splits else None,
                                   capsize=4 if has_splits else 0,
                                   label=f'FST {fst:.0f}', color=fst_color_map[fst], alpha=0.8, edgecolor='black', linewidth=0.5)

                    for bar, val, err in zip(bars, values, errors):
                        height = bar.get_height()
                        xpos = bar.get_x() + bar.get_width() / 2
                        ypos = height + err + 1
                        if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                            ax1.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

                ax1.set_xlabel('Model', fontweight='bold')
                ax1.set_ylabel(f'{metric} (%)', fontweight='bold')
                title_suffix = f" (across {f1_plot_data['Split'].nunique()} splits)" if has_splits else " (single split)"
                ax1.set_title(f'{metric} by Model and Skin Tone{title_suffix}', fontweight='bold')
                ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
                ax1.set_xticklabels(models, rotation=45, ha='right')
                ax1.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                # Disparity analysis subplot
                disparity_data = []
                for model in models:
                    model_data = f1_plot_data[f1_plot_data['Model'] == model]
                    if not model_data.empty and has_splits:
                        split_disparities = []
                        for split in model_data['Split'].unique():
                            split_data = model_data[model_data['Split'] == split]
                            if not split_data.empty:
                                tone_values = split_data.groupby('Skin Tone')['Value'].mean()
                                if len(tone_values) > 1:
                                    split_disparities.append(tone_values.max() - tone_values.min())
                        if split_disparities:
                            disparity_data.append({
                                'Model': model,
                                'mean': np.mean(split_disparities),
                                'std': np.std(split_disparities) if len(split_disparities) > 1 else 0
                            })
                
                disparity_df = pd.DataFrame(disparity_data)

                x_disp = np.arange(len(models))
                bars_disp = ax2.bar(x_disp, disparity_df['mean'], 
                                  yerr=disparity_df['std'] if has_splits else None,
                                  capsize=4 if has_splits else 0,
                                  color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

                for bar, val, err in zip(bars_disp, disparity_df['mean'], disparity_df['std']):
                    height = bar.get_height()
                    xpos = bar.get_x() + bar.get_width() / 2
                    ypos = height + err + 0.1
                    if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                        ax2.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

                ax2.set_xlabel('Model', fontweight='bold')
                ax2.set_ylabel('Skin Tone Disparity (Max-Min)', fontweight='bold')
                ax2.set_title(f'Disparity Analysis by Model{title_suffix}', fontweight='bold')
                ax2.set_xticks(x_disp)
                ax2.set_xticklabels(models, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = f"f1_score_by_model_skin_with_disparity_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: {filename}")

    # 2. Sensitivity by Model across Skin Tones with Disparity
    print("\nCreating sensitivity by model across skin tones chart...")
    if not sens_model_data.empty:
        has_splits = 'Split' in sens_model_data.columns and sens_model_data['Split'].nunique() > 1
        
        grouped_sens = sens_model_data.groupby(['Model', 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
        grouped_sens['stderr'] = grouped_sens['std'] / np.sqrt(grouped_sens['count'])
        grouped_sens['stderr'] = grouped_sens['stderr'].fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        models = sorted(grouped_sens['Model'].unique())
        skin_tones = sorted(grouped_sens['Skin Tone'].unique())
        x = np.arange(len(models))
        width = 0.8 / len(skin_tones)

        for i, fst in enumerate(skin_tones):
            values = []
            errors = []
            for model in models:
                row = grouped_sens[(grouped_sens['Model'] == model) & (grouped_sens['Skin Tone'] == fst)]
                if not row.empty:
                    values.append(row['mean'].values[0])
                    errors.append(row['stderr'].values[0] if has_splits else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax1.bar(x + i * width, values, width,
                           yerr=errors if has_splits else None,
                           capsize=4 if has_splits else 0,
                           label=f'FST {fst:.0f}', color=fst_color_map[fst], alpha=0.8, edgecolor='black', linewidth=0.5)

            for bar, val, err in zip(bars, values, errors):
                height = bar.get_height()
                xpos = bar.get_x() + bar.get_width() / 2
                ypos = height + err + 1
                if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                    ax1.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

        ax1.set_xlabel('Model', fontweight='bold')
        ax1.set_ylabel('Top-1 Sensitivity (%)', fontweight='bold')
        title_suffix = f" (across {sens_model_data['Split'].nunique()} splits)" if has_splits else " (single split)"
        ax1.set_title(f'Sensitivity by Model and Skin Tone{title_suffix}', fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Disparity analysis subplot
        disparity_data = []
        for model in models:
            model_data = sens_model_data[sens_model_data['Model'] == model]
            if not model_data.empty and has_splits:
                split_disparities = []
                for split in model_data['Split'].unique():
                    split_data = model_data[model_data['Split'] == split]
                    if not split_data.empty:
                        tone_values = split_data.groupby('Skin Tone')['Value'].mean()
                        if len(tone_values) > 1:
                            split_disparities.append(tone_values.max() - tone_values.min())
                if split_disparities:
                    disparity_data.append({
                        'Model': model,
                        'mean': np.mean(split_disparities),
                        'std': np.std(split_disparities) if len(split_disparities) > 1 else 0
                    })
        
        disparity_df = pd.DataFrame(disparity_data)

        x_disp = np.arange(len(models))
        bars_disp = ax2.bar(x_disp, disparity_df['mean'], 
                          yerr=disparity_df['std'] if has_splits else None,
                          capsize=4 if has_splits else 0,
                          color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, val, err in zip(bars_disp, disparity_df['mean'], disparity_df['std']):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + err + 0.1
            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax2.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

        ax2.set_xlabel('Model', fontweight='bold')
        ax2.set_ylabel('Skin Tone Disparity (Max-Min)', fontweight='bold')
        ax2.set_title(f'Disparity Analysis by Model{title_suffix}', fontweight='bold')
        ax2.set_xticks(x_disp)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sensitivity_by_model_skin_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: sensitivity_by_model_skin_with_disparity.png")

def create_dataset_skin_tone_charts(sensitivity_data, f1_data, save_path):
    """Create charts showing performance by dataset configuration across skin tones with disparity subplots"""
    
    sens_dataset_data = sensitivity_data[sensitivity_data['Dataset_Config'] != 'All Datasets'].copy()
    f1_dataset_data = f1_data[f1_data['Dataset_Config'] != 'All Datasets'].copy()
    
    if sens_dataset_data.empty and f1_dataset_data.empty:
        print("No dataset comparison data found for plotting.")
        return

    # 1. F1 Score by Dataset across Skin Tones with Disparity
    print("\nCreating F1 score by dataset across skin tones charts...")
    if not f1_dataset_data.empty:
        f1_metrics = f1_dataset_data['Metric'].unique()
        for metric in f1_metrics:
            if 'F1' in metric:
                print(f"  - Plotting for metric: {metric}")
                f1_plot_data = f1_dataset_data[f1_dataset_data['Metric'] == metric].copy()
                
                has_splits = 'Split' in f1_plot_data.columns and f1_plot_data['Split'].nunique() > 1
                
                grouped_f1 = f1_plot_data.groupby(['Dataset_Config', 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
                grouped_f1['stderr'] = grouped_f1['std'] / np.sqrt(grouped_f1['count'])
                grouped_f1['stderr'] = grouped_f1['stderr'].fillna(0)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                
                datasets = sorted(grouped_f1['Dataset_Config'].unique())
                skin_tones = sorted(grouped_f1['Skin Tone'].unique())
                x = np.arange(len(datasets))
                width = 0.8 / len(skin_tones)

                for i, fst in enumerate(skin_tones):
                    values = []
                    errors = []
                    for dataset in datasets:
                        row = grouped_f1[(grouped_f1['Dataset_Config'] == dataset) & (grouped_f1['Skin Tone'] == fst)]
                        if not row.empty:
                            values.append(row['mean'].values[0])
                            errors.append(row['stderr'].values[0] if has_splits else 0)
                        else:
                            values.append(0)
                            errors.append(0)
                    
                    bars = ax1.bar(x + i * width, values, width,
                                   yerr=errors if has_splits else None,
                                   capsize=4 if has_splits else 0,
                                   label=f'FST {fst:.0f}', color=fst_color_map[fst], alpha=0.8, edgecolor='black', linewidth=0.5)

                    for bar, val, err in zip(bars, values, errors):
                        height = bar.get_height()
                        xpos = bar.get_x() + bar.get_width() / 2
                        ypos = height + err + 1
                        if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                            ax1.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

                ax1.set_xlabel('Dataset Configuration', fontweight='bold')
                ax1.set_ylabel(f'{metric} (%)', fontweight='bold')
                title_suffix = f" (across {f1_plot_data['Split'].nunique()} splits)" if has_splits else " (single split)"
                ax1.set_title(f'{metric} by Dataset and Skin Tone{title_suffix}', fontweight='bold')
                ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
                ax1.set_xticklabels(datasets, rotation=45, ha='right')
                ax1.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
                ax1.grid(True, alpha=0.3)
                
                # Disparity analysis subplot
                disparity_data = []
                for dataset in datasets:
                    dataset_data = f1_plot_data[f1_plot_data['Dataset_Config'] == dataset]
                    if not dataset_data.empty and has_splits:
                        split_disparities = []
                        for split in dataset_data['Split'].unique():
                            split_data = dataset_data[dataset_data['Split'] == split]
                            if not split_data.empty:
                                tone_values = split_data.groupby('Skin Tone')['Value'].mean()
                                if len(tone_values) > 1:
                                    split_disparities.append(tone_values.max() - tone_values.min())
                        if split_disparities:
                            disparity_data.append({
                                'Dataset_Config': dataset,
                                'mean': np.mean(split_disparities),
                                'std': np.std(split_disparities) if len(split_disparities) > 1 else 0
                            })
                
                disparity_df = pd.DataFrame(disparity_data)

                x_disp = np.arange(len(datasets))
                bars_disp = ax2.bar(x_disp, disparity_df['mean'], 
                                  yerr=disparity_df['std'] if has_splits else None,
                                  capsize=4 if has_splits else 0,
                                  color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

                for bar, val, err in zip(bars_disp, disparity_df['mean'], disparity_df['std']):
                    height = bar.get_height()
                    xpos = bar.get_x() + bar.get_width() / 2
                    ypos = height + err + 0.1
                    if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                        ax2.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

                ax2.set_xlabel('Dataset Configuration', fontweight='bold')
                ax2.set_ylabel('Skin Tone Disparity (Max-Min)', fontweight='bold')
                ax2.set_title(f'Disparity Analysis by Dataset{title_suffix}', fontweight='bold')
                ax2.set_xticks(x_disp)
                ax2.set_xticklabels(datasets, rotation=45, ha='right')
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                filename = f"f1_score_by_dataset_skin_with_disparity_{metric.replace(' ', '_').replace('(', '').replace(')', '')}.png"
                plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: {filename}")

    # 2. Sensitivity by Dataset across Skin Tones with Disparity
    print("\nCreating sensitivity by dataset across skin tones chart...")
    if not sens_dataset_data.empty:
        has_splits = 'Split' in sens_dataset_data.columns and sens_dataset_data['Split'].nunique() > 1
        
        grouped_sens = sens_dataset_data.groupby(['Dataset_Config', 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
        grouped_sens['stderr'] = grouped_sens['std'] / np.sqrt(grouped_sens['count'])
        grouped_sens['stderr'] = grouped_sens['stderr'].fillna(0)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        datasets = sorted(grouped_sens['Dataset_Config'].unique())
        skin_tones = sorted(grouped_sens['Skin Tone'].unique())
        x = np.arange(len(datasets))
        width = 0.8 / len(skin_tones)

        for i, fst in enumerate(skin_tones):
            values = []
            errors = []
            for dataset in datasets:
                row = grouped_sens[(grouped_sens['Dataset_Config'] == dataset) & (grouped_sens['Skin Tone'] == fst)]
                if not row.empty:
                    values.append(row['mean'].values[0])
                    errors.append(row['stderr'].values[0] if has_splits else 0)
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax1.bar(x + i * width, values, width,
                           yerr=errors if has_splits else None,
                           capsize=4 if has_splits else 0,
                           label=f'FST {fst:.0f}', color=fst_color_map[fst], alpha=0.8, edgecolor='black', linewidth=0.5)

            for bar, val, err in zip(bars, values, errors):
                height = bar.get_height()
                xpos = bar.get_x() + bar.get_width() / 2
                ypos = height + err + 1
                if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                    ax1.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

        ax1.set_xlabel('Dataset Configuration', fontweight='bold')
        ax1.set_ylabel('Top-1 Sensitivity (%)', fontweight='bold')
        title_suffix = f" (across {sens_dataset_data['Split'].nunique()} splits)" if has_splits else " (single split)"
        ax1.set_title(f'Sensitivity by Dataset and Skin Tone{title_suffix}', fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Disparity analysis subplot
        disparity_data = []
        for dataset in datasets:
            dataset_data = sens_dataset_data[sens_dataset_data['Dataset_Config'] == dataset]
            if not dataset_data.empty and has_splits:
                split_disparities = []
                for split in dataset_data['Split'].unique():
                    split_data = dataset_data[dataset_data['Split'] == split]
                    if not split_data.empty:
                        tone_values = split_data.groupby('Skin Tone')['Value'].mean()
                        if len(tone_values) > 1:
                            split_disparities.append(tone_values.max() - tone_values.min())
                if split_disparities:
                    disparity_data.append({
                        'Dataset_Config': dataset,
                        'mean': np.mean(split_disparities),
                        'std': np.std(split_disparities) if len(split_disparities) > 1 else 0
                    })
        
        disparity_df = pd.DataFrame(disparity_data)

        x_disp = np.arange(len(datasets))
        bars_disp = ax2.bar(x_disp, disparity_df['mean'], 
                          yerr=disparity_df['std'] if has_splits else None,
                          capsize=4 if has_splits else 0,
                          color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, val, err in zip(bars_disp, disparity_df['mean'], disparity_df['std']):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + err + 0.1
            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax2.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

        ax2.set_xlabel('Dataset Configuration', fontweight='bold')
        ax2.set_ylabel('Skin Tone Disparity (Max-Min)', fontweight='bold')
        title_suffix = f" (across {sens_dataset_data['Split'].nunique()} splits)" if has_splits else " (single split)"
        ax2.set_title(f'Disparity Analysis by Dataset{title_suffix}', fontweight='bold')
        ax2.set_xticks(x_disp)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sensitivity_by_dataset_skin_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: sensitivity_by_dataset_skin_with_disparity.png")

def main():
    """Main function to run the analysis and generate plots"""
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    sensitivity_data, f1_data = load_skin_tone_data()
    
    if not sensitivity_data.empty or not f1_data.empty:
        create_model_skin_tone_charts(sensitivity_data, f1_data, output_directory)
        create_dataset_skin_tone_charts(sensitivity_data, f1_data, output_directory)
    else:
        print("No data found. Exiting.")
    
    print("\n=== Analysis Complete ===")
    print(f"Plots saved to: {output_directory}")

if __name__ == "__main__":
    main()