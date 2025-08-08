import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re
import glob

# Add the parent directory to the Python path to import dataframe module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

# ================================================================
# MODIFIED FUNCTIONS FROM overall2.py
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
    This function has been generalized to combine any number of log types.
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
# ORIGINAL FUNCTIONS FROM skin_tone_condition copy.py
# (with modifications for multiple splits)
# ================================================================

def parse_baseline_datasets_log(filepath):
    """Parse combined_baseline_datasets.txt and extract Dataset Combination field"""
    with open(filepath, 'r') as file:
        content = file.read()

    model_blocks = re.split(r'(?=Python Filename:)', content)

    avg_sensitivities = []
    condition_sensitivities = []
    skin_tone_accuracies = []
    stratified_sensitivities = []
    f1_scores = []

    for block in model_blocks:
        if not block.strip():
            continue

        model_match = re.search(r'Python Filename:\s*(\S+)', block)
        if not model_match: continue
        model = model_match.group(1).replace('.py', '')
        
        dataset_combination_match = re.search(r'Dataset Combination:\s*(.+)', block)
        if not dataset_combination_match: continue
        dataset_combination = dataset_combination_match.group(1).strip()
        
        # --- Parse data (no change here, just use the extracted fields) ---
        for k in ['Top-1', 'Top-3', 'Top-5']:
            match = re.search(rf'Average {k} Sensitivity:\s*([\d.]+)%', block)
            if match:
                avg_sensitivities.append({'Model': model, 'Datasets': dataset_combination, 'Metric': f'{k} Sensitivity', 'Value': float(match.group(1))})
        
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                condition_sensitivities.append({'Model': model, 'Datasets': dataset_combination, 'Condition': condition, 'Metric': f'{k} Sensitivity', 'Value': float(value)})
        
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for tone_match in re.findall(rf'Skin Tone:\s*([\d.]+), {k} Accuracy:\s*([\d.]+)%', block):
                tone, value = tone_match
                skin_tone_accuracies.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': float(tone), 'Metric': f'{k} Accuracy', 'Value': float(value)})

        stratified_blocks = re.findall(r'Skin Tone:\s*([\d.]+)\n((?:\s+Condition:.*\n?)+)', block)
        for tone, section in stratified_blocks:
            matches = re.findall(r'Condition:\s*(.+?), (Top-[135]) Sensitivity:\s*([\d.]+)%', section)
            for condition, k, value in matches:
                stratified_sensitivities.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': float(tone), 'Condition': condition, 'Metric': f'{k} Sensitivity', 'Value': float(value)})

        # Parse F1 Score data (no change)
        for f1_type in ['Macro', 'Weighted']:
            match = re.search(rf'Overall F1 Score \({f1_type}\):\s*([\d.]+)%', block)
            if match:
                f1_scores.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': None, 'Condition': 'Overall', 'Metric': f'F1 Score ({f1_type})', 'Value': float(match.group(1))})

        f1_section = re.search(r'=== F1 SCORE ===(.+?)(?==== |$)', block, re.DOTALL)
        if f1_section:
            f1_content = f1_section.group(1)
            condition_matches = re.findall(r'^\s*([^:]+):\s*([\d.]+)%', f1_content, re.MULTILINE)
            for condition, value in condition_matches:
                condition = condition.strip()
                f1_scores.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': None, 'Condition': condition, 'Metric': 'F1 Score', 'Value': float(value)})

        stratified_f1_section = re.search(r'=== STRATIFIED F1 SCORE ===(.+?)(?==== |$)', block, re.DOTALL)
        if stratified_f1_section:
            stratified_content = stratified_f1_section.group(1)
            skin_tone_blocks = re.findall(r'Skin Tone:\s*([\d.]+)(.*?)(?=Skin Tone:|$)', stratified_content, re.DOTALL)
            for tone, tone_content in skin_tone_blocks:
                tone_val = float(tone)
                for f1_type in ['Macro', 'Weighted']:
                    match = re.search(rf'Overall F1 Score \({f1_type}\):\s*([\d.]+)%', tone_content)
                    if match:
                        f1_scores.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': tone_val, 'Condition': 'Overall', 'Metric': f'F1 Score ({f1_type})', 'Value': float(match.group(1))})
                
                per_condition_section = re.search(r'Per Condition:(.*?)(?=\n\nSkin Tone:|$)', tone_content, re.DOTALL)
                if per_condition_section:
                    condition_content = per_condition_section.group(1)
                    condition_matches = re.findall(r'^\s*([^:]+):\s*([\d.]+)%', condition_content, re.MULTILINE)
                    for condition, value in condition_matches:
                        condition = condition.strip()
                        f1_scores.append({'Model': model, 'Datasets': dataset_combination, 'Skin Tone': tone_val, 'Condition': condition, 'Metric': 'F1 Score', 'Value': float(value)})

    return {
        'AverageSensitivities': pd.DataFrame(avg_sensitivities),
        'ConditionSensitivities': pd.DataFrame(condition_sensitivities),
        'SkinToneAccuracies': pd.DataFrame(skin_tone_accuracies),
        'StratifiedSensitivities': pd.DataFrame(stratified_sensitivities),
        'F1Scores': pd.DataFrame(f1_scores)
    }

# Fitzpatrick Skin Tone color mapping
fst_color_map = {
    1: '#F5D5A0', 2: '#E4B589', 3: '#D1A479', 4: '#C0874F', 5: '#A56635', 6: '#4C2C27'
}

def create_skin_tone_plot_with_disparity(data, x_col, title, ylabel, save_path, condition_name, plot_type):
    """
    Create a skin tone plot with disparity metrics.
    Now calculates mean and standard error across splits for error bars.
    """
    if data.empty:
        print(f"No data available for {title}")
        return

    data = data[~data['Model'].str.contains('FairDisCo', case=False, na=False)]
    if data.empty:
        print(f"No data available for {title} after filtering")
        return

    # Check if we have multiple splits
    has_splits = 'Split' in data.columns and data['Split'].nunique() > 1
    if not has_splits:
        print("WARNING: No split information found or only one split. Plots will not have error bars.")

    # Group data and calculate statistics across splits
    grouped = data.groupby([x_col, 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
    grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['stderr'] = grouped['stderr'].fillna(0)

    x_values = sorted(grouped[x_col].unique())
    skin_tones = [1, 2, 3, 4, 5, 6]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Main plot
    x = np.arange(len(x_values))
    width = 0.8 / len(skin_tones)

    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for x_val in x_values:
            row = grouped[(grouped[x_col] == x_val) & (grouped['Skin Tone'] == tone)]
            if not row.empty:
                values.append(row['mean'].values[0])
                if has_splits:
                    errors.append(row['stderr'].values[0])
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)

        bars = ax1.bar(x + i * width, values, width,
                       yerr=errors if has_splits else None,
                       capsize=4 if has_splits else 0,
                       label=f'FST {tone}', color=fst_color_map[tone], alpha=0.8, edgecolor='black', linewidth=0.5)

        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + (err if has_splits else 0) + 1.0

            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax1.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

    ax1.set_xlabel(x_col.replace('_', ' ').title(), fontweight='bold')
    ax1.set_ylabel(ylabel, fontweight='bold')
    title_suffix = f" (across {data['Split'].nunique()} splits)" if has_splits else " (single split)"
    ax1.set_title(f'{title} - {condition_name}{title_suffix}', fontweight='bold')
    ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
    ax1.set_xticklabels(x_values, rotation=45, ha='right')
    ax1.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Disparity analysis subplot - now plots standard deviation across splits
    disparity_data = []
    for x_val in x_values:
        x_val_data = data[data[x_col] == x_val]
        if not x_val_data.empty and 'Split' in x_val_data.columns:
            split_disparities = []
            for split in x_val_data['Split'].unique():
                split_data = x_val_data[x_val_data['Split'] == split]
                if not split_data.empty:
                    tone_values = split_data.groupby('Skin Tone')['Value'].mean()
                    if len(tone_values) > 1:
                        split_disparities.append(tone_values.max() - tone_values.min())
            if split_disparities:
                disparity_data.append({
                    x_col: x_val,
                    'mean': np.mean(split_disparities),
                    'std': np.std(split_disparities) if len(split_disparities) > 1 else 0
                })

    disparity_df = pd.DataFrame(disparity_data)

    x_disp = np.arange(len(x_values))
    bars = ax2.bar(x_disp, disparity_df['mean'],
            yerr=disparity_df['std'] if has_splits else None,
            capsize=4 if has_splits else 0,
            color='lightcoral', alpha=0.8, edgecolor='black', linewidth=0.5)

    for bar, val, err in zip(bars, disparity_df['mean'], disparity_df['std']):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + (err if has_splits else 0) + 0.1
            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax2.text(xpos, ypos, f'{val:.1f}', ha='center', va='bottom', fontsize=9, color='black')

    ax2.set_xlabel(x_col.replace('_', ' ').title(), fontweight='bold')
    ax2.set_ylabel('Skin Tone Disparity (Max-Min)', fontweight='bold')
    ax2.set_title(f'Disparity Analysis - {condition_name}{title_suffix}', fontweight='bold')
    ax2.set_xticks(x_disp)
    ax2.set_xticklabels(x_values, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"{condition_name.lower().replace(' ', '_')}_{plot_type}"
    plt.savefig(f'{save_path}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {filename}")


def generate_condition_plots(df_dict, save_path='./'):
    """Generate the 4 required plots for each condition"""
    
    for df_name in df_dict.keys():
        if 'Model' in df_dict[df_name].columns:
            df_dict[df_name]['Model'] = df_dict[df_name]['Model'].replace({
                'train_Baseline': 'Baseline',
                'train_VAE': 'VAE',
                'train_TABE': 'TABE'
            })
    
    f1_data = df_dict['F1Scores']
    stratified_data = df_dict['StratifiedSensitivities']
    
    if f1_data.empty and stratified_data.empty:
        print("No F1 or stratified sensitivity data found!")
        return
    
    conditions = set()
    if not f1_data.empty: conditions.update(f1_data['Condition'].unique())
    if not stratified_data.empty: conditions.update(stratified_data['Condition'].unique())
    
    conditions = [c for c in conditions if c != 'Overall']
    print(f"Found conditions: {conditions}")
    
    for condition in conditions:
        print(f"\nGenerating plots for condition: {condition}")
        
        # 1. F1 Score by Model
        f1_model_data = f1_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'all_models' and "
            f"Metric == 'F1 Score' and "
            f"`Skin Tone`.notna()"
        ).copy()
        
        if not f1_model_data.empty:
            create_skin_tone_plot_with_disparity(
                f1_model_data, 'Model', 
                'F1 Score by Model', 'F1 Score (%)', 
                save_path, condition, 'f1_score_by_model_skin_with_disparity.png'
            )
        else:
            print(f"No F1 model data found for {condition}")
        
        # 2. Sensitivity by Model
        sens_model_data = stratified_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'all_models' and "
            f"Metric == 'Top-1 Sensitivity'"
        ).copy() if not stratified_data.empty else pd.DataFrame()
        
        if not sens_model_data.empty:
            create_skin_tone_plot_with_disparity(
                sens_model_data, 'Model', 
                'Sensitivity by Model', 'Sensitivity (%)', 
                save_path, condition, 'sensitivity_by_model_skin_with_disparity.png'
            )
        else:
            print(f"No model sensitivity data found for {condition}")

# ================================================================
# Main function with new logic
# ================================================================

def main():
    """Main function to handle multiple log files and generate plots with error bars"""
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
    save_path = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneConditionAnalysis"
    
    os.makedirs(save_path, exist_ok=True)
    
    print("=== Multi-Split Condition and Skin Tone Analysis ===")
    
    # 1. Parse all 'combined_all' log files
    combined_all_files = find_log_files(log_directory, 'combined_all')
    if not combined_all_files:
        print("No 'combined_all' log files found!")
        return

    all_models_data = parse_multiple_logs(combined_all_files, parse_combined_log, 'all_models')
    
    # 2. Parse the single 'combined_baseline_datasets.txt' file
    baseline_file_path = os.path.join(log_directory, 'combined_baseline_datasets.txt')
    if os.path.exists(baseline_file_path):
        print("\nParsing single 'combined_baseline_datasets.txt' file...")
        # Since this is a single file, it's a single "split" with a source identifier
        baseline_datasets_data = parse_multiple_logs([baseline_file_path], parse_baseline_datasets_log, 'baseline_datasets', split_offset=len(combined_all_files))
    else:
        print("No 'combined_baseline_datasets.txt' found. Skipping dataset comparison plots.")
        baseline_datasets_data = []

    # 3. Combine all parsed data
    print("\nCombining data from all sources and splits...")
    combined_data = combine_data_with_stats(all_models_data + baseline_datasets_data)
    
    print(f"Combined data types: {list(combined_data.keys())}")
    
    # 4. Generate the condition-specific plots
    generate_condition_plots(combined_data, save_path)

    # Re-run the dataset comparison plots, which were removed in the new
    # `generate_condition_plots` for clarity. This ensures all four original
    # plots are generated.
    print("\nGenerating dataset-specific plots...")
    generate_dataset_plots(combined_data, save_path)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Plots saved to: {save_path}")

def generate_dataset_plots(df_dict, save_path):
    """
    Generate plots comparing different datasets for the baseline model.
    This is a new helper function to separate the logic.
    """
    f1_data = df_dict['F1Scores']
    stratified_data = df_dict['StratifiedSensitivities']
    
    conditions = set()
    if not f1_data.empty: conditions.update(f1_data['Condition'].unique())
    if not stratified_data.empty: conditions.update(stratified_data['Condition'].unique())
    
    conditions = [c for c in conditions if c != 'Overall']

    for condition in conditions:
        print(f"\nGenerating dataset plots for condition: {condition}")
        
        # 1. F1 Score by Dataset
        f1_dataset_data = f1_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'baseline_datasets' and "
            f"Metric == 'F1 Score' and "
            f"`Skin Tone`.notna()"
        ).copy()
        
        if not f1_dataset_data.empty:
            create_skin_tone_plot_with_disparity(
                f1_dataset_data, 'Datasets', 
                'F1 Score by Dataset', 'F1 Score (%)', 
                save_path, condition, 'f1_score_by_dataset_skin_with_disparity.png'
            )
        else:
            print(f"No F1 dataset data found for {condition}")
        
        # 2. Sensitivity by Dataset
        sens_dataset_data = stratified_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'baseline_datasets' and "
            f"Metric == 'Top-1 Sensitivity'"
        ).copy() if not stratified_data.empty else pd.DataFrame()
        
        if not sens_dataset_data.empty:
            create_skin_tone_plot_with_disparity(
                sens_dataset_data, 'Datasets', 
                'Sensitivity by Dataset', 'Sensitivity (%)', 
                save_path, condition, 'sensitivity_by_dataset_skin_with_disparity.png'
            )
        else:
            print(f"No dataset sensitivity data found for {condition}")

if __name__ == "__main__":
    main()