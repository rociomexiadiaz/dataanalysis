import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import glob
import re

# Add the parent directory to the Python path to import dataframe module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

def find_log_files(log_directory, base_pattern='combined_all'):
    """
    Find all log files matching the base pattern with optional suffixes
    e.g., combined_all.txt, combined_all_0.txt, combined_all_1.txt, etc.
    """
    # Look for files with pattern: base_pattern[_suffix].txt
    pattern = os.path.join(log_directory, f"{base_pattern}*.txt")
    files = glob.glob(pattern)
    
    # Sort files to ensure consistent ordering
    files.sort()
    
    print(f"Found {len(files)} log files matching '{base_pattern}*':")
    for f in files:
        print(f"  {os.path.basename(f)}")
    
    return files

def parse_multiple_logs(log_files):
    """
    Parse multiple log files and combine results with split information
    """
    all_data = []
    
    for i, log_file in enumerate(log_files):
        print(f"\nParsing {os.path.basename(log_file)}...")
        
        try:
            df_dict = parse_combined_log(log_file)
            
            # Add split identifier to each dataframe
            processed_dict = {}
            for key, df in df_dict.items():
                if not df.empty:
                    df_copy = df.copy()
                    df_copy['Split'] = i
                    df_copy['LogFile'] = os.path.basename(log_file)
                    processed_dict[key] = df_copy
                else:
                    processed_dict[key] = df  # Keep empty dataframe as is
            
            all_data.append(processed_dict)
            print(f"  Successfully parsed {len([k for k, v in processed_dict.items() if not v.empty])} data types")
            
        except Exception as e:
            print(f"  Error parsing {log_file}: {e}")
            continue
    
    return all_data

def combine_data_with_stats(all_data):
    """
    Combine data from multiple splits and compute statistics
    """
    combined_results = {}
    
    # Get all data types from first successful parse
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

def plot_balanced_accuracy_overall(combined_data, output_directory):
    """
    Plot overall balanced accuracy across models with error bars from multiple splits
    """
    # Look in the correct dataframe for balanced accuracy
    if 'BalancedAccuraciesBySkinTone' not in combined_data or combined_data['BalancedAccuraciesBySkinTone'].empty:
        print("WARNING: No balanced accuracy data found in BalancedAccuraciesBySkinTone")
        return False
    
    balanced_acc_df = combined_data['BalancedAccuraciesBySkinTone']
    print(f"DEBUG: Found BalancedAccuraciesBySkinTone with {len(balanced_acc_df)} records")
    print(f"DEBUG: Columns: {list(balanced_acc_df.columns)}")
    
    if 'Metric' in balanced_acc_df.columns:
        print(f"DEBUG: Available metrics in BalancedAccuraciesBySkinTone: {balanced_acc_df['Metric'].unique()}")
    
    # Clean up model names
    balanced_acc_df = balanced_acc_df.copy()
    balanced_acc_df['Model'] = balanced_acc_df['Model'].str.replace('train_', '')
    #balanced_acc_df = balanced_acc_df[~balanced_acc_df['Model'].str.contains('FairDisco', case=False, na=False)]
    
    print(f"DEBUG: Available models: {balanced_acc_df['Model'].unique()}")
    
    # Check if Split column exists and count unique splits properly
    has_splits = 'Split' in balanced_acc_df.columns
    if not has_splits:
        balanced_acc_df['Split'] = 0  # Add dummy split column
        n_splits = 1
        print("DEBUG: No split column found, using single split")
    else:
        n_splits = balanced_acc_df['Split'].nunique()
        print(f"DEBUG: Found {n_splits} splits: {sorted(balanced_acc_df['Split'].unique())}")
    
    # Compute overall accuracy (mean across all skin tones per model per split)
    overall_data = balanced_acc_df[balanced_acc_df['Skin Tone'] == 'Overall']
    overall_by_split = overall_data[['Model', 'Split', 'Value']]
    #overall_by_split = balanced_acc_df.groupby(['Model', 'Split'])['Value'].mean().reset_index()
    print(f"DEBUG: Computed overall performance for {len(overall_by_split)} model-split combinations")
    
    # Group by model and compute statistics across splits
    stats = overall_by_split.groupby('Model')['Value'].agg(['mean', 'std', 'count']).reset_index()
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
    stats['stderr'] = stats['stderr'].fillna(0)
    
    print(f"DEBUG: Final stats shape: {stats.shape}")
    print(f"DEBUG: Models in stats: {stats['Model'].tolist()}")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(stats))
    
    if n_splits > 1:
        bars = plt.bar(x_pos, stats['mean'], yerr=stats['stderr'],
                       capsize=5, alpha=0.8, edgecolor='black', linewidth=1, color='steelblue')
        error_info = f"across {n_splits} splits"
    else:
        bars = plt.bar(x_pos, stats['mean'], 
                       alpha=0.8, edgecolor='black', linewidth=1, color='steelblue')
        error_info = "(single split)"
       
    # Customize plot
    plt.xlabel('Model Architecture', fontsize=14, fontweight='bold')
    plt.ylabel('Overall Balanced Accuracy (%)', fontsize=14, fontweight='bold')
    plt.title(f'Overall Balanced Accuracy Comparison {error_info}', fontsize=16, fontweight='bold')
    plt.xticks(x_pos, stats['Model'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, mean_val) in enumerate(zip(bars, stats['mean'])):
        if n_splits > 1:
            std_val = stats.iloc[i]['std']
            stderr_val = stats.iloc[i]['stderr']
            label_text = f'{mean_val:.1f}±{std_val:.1f}'
            y_pos = mean_val + stderr_val + 1
        else:
            label_text = f'{mean_val:.1f}'
            y_pos = mean_val + 1
        
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                label_text, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_directory, 'balanced_accuracy_overall.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return True

def plot_balanced_accuracy_by_skin_tone(combined_data, output_directory):
    """
    Plot balanced accuracy by skin tone across models with error bars from multiple splits
    """
    # Look in the correct dataframe for balanced accuracy
    if 'BalancedAccuraciesBySkinTone' not in combined_data or combined_data['BalancedAccuraciesBySkinTone'].empty:
        print("WARNING: No balanced accuracy data found in BalancedAccuraciesBySkinTone")
        return False
    
    balanced_acc_df = combined_data['BalancedAccuraciesBySkinTone']
    print(f"DEBUG: Found BalancedAccuraciesBySkinTone with {len(balanced_acc_df)} records for skin tone plot")
    
    # Clean up model names
    balanced_acc_df = balanced_acc_df.copy()
    balanced_acc_df['Model'] = balanced_acc_df['Model'].str.replace('train_', '')
    #balanced_acc_df = balanced_acc_df[~balanced_acc_df['Model'].str.contains('FairDisco', case=False, na=False)]
    
    # Check if Split column exists and count splits properly
    has_splits = 'Split' in balanced_acc_df.columns
    if not has_splits:
        balanced_acc_df['Split'] = 0
        n_splits = 1
    else:
        n_splits = balanced_acc_df['Split'].nunique()
        print(f"DEBUG: Found {n_splits} splits for skin tone plot")
    
    # Skin tone color mapping
    fst_color_map = {
        1: '#F5D5A0',
        2: '#E4B589', 
        3: '#D1A479',
        4: '#C0874F',
        5: '#A56635',
        6: '#4C2C27'
    }
    
    # Group by model and skin tone, compute statistics across splits
    grouped = balanced_acc_df.groupby(['Model', 'Skin Tone'])['Value'].agg(['mean', 'std', 'count']).reset_index()
    grouped['stderr'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['stderr'] = grouped['stderr'].fillna(0)
    
    print(f"DEBUG: Grouped data shape: {grouped.shape}")
    grouped = grouped[grouped['Skin Tone']!= 'Overall']
    print(f"DEBUG: Available skin tones: {sorted(grouped['Skin Tone'].unique())}")
    
    models = sorted(grouped['Model'].unique())
    skin_tones = [1, 2, 3, 4, 5, 6]
    x = np.arange(len(models))
    width = 0.8 / len(skin_tones)
    
    print(f"DEBUG: Will plot {len(models)} models: {models}")
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for model in models:
            row = grouped[(grouped['Model'] == model) & (grouped['Skin Tone'] == tone)]
            if not row.empty:
                values.append(row['mean'].values[0])
                if n_splits > 1:
                    errors.append(row['stderr'].values[0])
                else:
                    errors.append(0)
            else:
                values.append(0)
                errors.append(0)
        
        # Show error bars only if we have multiple splits
        if n_splits > 1 and any(err > 0 for err in errors):
            bars = plt.bar(x + i * width, values, width, yerr=errors, capsize=4,
                    label=f'FST {tone}', color=fst_color_map[tone], alpha=0.8, edgecolor='black', linewidth=0.5)
        else:
            bars = plt.bar(x + i * width, values, width,
                    label=f'FST {tone}', color=fst_color_map[tone], alpha=0.8, edgecolor='black', linewidth=0.5)
            
        # Add value labels on bars
        for bar, acc in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=6, fontweight='bold')
    
    
    # Customize plot
    plt.xticks(x + width * (len(skin_tones) - 1) / 2, models, rotation=45, ha='right')
    plt.ylabel('Balanced Accuracy (%)', fontsize=14, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=14, fontweight='bold')
    
    if n_splits > 1:
        title_suffix = f"across {n_splits} splits"
    else:
        title_suffix = "(single split)"
    
    plt.title(f'Balanced Accuracy by Skin Tone {title_suffix}', fontsize=16, fontweight='bold')
    plt.legend(title='Fitzpatrick Skin Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(output_directory, 'balanced_accuracy_by_skin_tone.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")
    return True

def plot_condition_sensitivity_with_errors(combined_data, output_directory):
    """
    Plot condition-specific sensitivity with error bars
    """
    if 'ConditionSensitivities' not in combined_data or combined_data['ConditionSensitivities'].empty:
        print("No condition sensitivity data found for plotting")
        return
    
    cond_df = combined_data['ConditionSensitivities']
    
    # Focus on Top-1 Sensitivity for baseline model
    baseline_data = cond_df[
        (cond_df['Metric'] == 'Top-1 Sensitivity') & 
        (cond_df['Model'].str.contains('Baseline', case=False, na=False))
    ]
    
    if baseline_data.empty:
        print("No baseline condition data found")
        return
    
    # Group by condition and compute statistics
    stats = baseline_data.groupby('Condition')['Value'].agg(['mean', 'std', 'count']).reset_index()
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
    stats = stats.sort_values('mean', ascending=True)
    
    # Create plot
    plt.figure(figsize=(14, 8))
    
    x_pos = np.arange(len(stats))
    bars = plt.barh(x_pos, stats['mean'], xerr=stats['stderr'], 
                    capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize plot
    plt.ylabel('Medical Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Top-1 Sensitivity (%)', fontsize=14, fontweight='bold')
    plt.title('Baseline Model: Condition-Specific Sensitivity Across Multiple Splits', 
              fontsize=16, fontweight='bold')
    plt.yticks(x_pos, stats['Condition'])
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, mean_val, std_val) in enumerate(zip(bars, stats['mean'], stats['std'])):
        plt.text(mean_val + stats.iloc[i]['stderr'] + 1, bar.get_y() + bar.get_height()/2.,
                f'{mean_val:.1f}±{std_val:.1f}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'condition_sensitivity_baseline.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: condition_sensitivity_baseline.png")

def plot_skin_tone_disparity_analysis(combined_data, output_directory):
    """
    Plot skin tone disparity analysis showing fairness across models
    """
    if 'StratifiedSensitivities' not in combined_data or combined_data['StratifiedSensitivities'].empty:
        print("No stratified data for disparity analysis")
        return
    
    stratified_df = combined_data['StratifiedSensitivities']
    
    # Look for accuracy data first, fall back to sensitivity
    accuracy_data = stratified_df[stratified_df['Metric'] == 'Top-1 Accuracy']
    if accuracy_data.empty:
        accuracy_data = stratified_df[stratified_df['Metric'] == 'Top-1 Sensitivity']
        metric_name = 'Sensitivity'
    else:
        metric_name = 'Accuracy'
    
    if accuracy_data.empty:
        print("No accuracy or sensitivity data found for disparity analysis")
        return
    
    # Clean model names
    accuracy_data = accuracy_data.copy()
    accuracy_data['Model'] = accuracy_data['Model'].str.replace('train_', '')
    
    # Check if Split column exists
    has_splits = 'Split' in accuracy_data.columns
    if not has_splits:
        print("No split information found, computing disparity without split-based error bars")
        # Add a dummy split column
        accuracy_data['Split'] = 0
    
    # Compute disparity for each model-split combination
    disparity_data = []
    
    for model in accuracy_data['Model'].unique():
        model_data = accuracy_data[accuracy_data['Model'] == model]
        
        for split in model_data['Split'].unique():
            split_data = model_data[model_data['Split'] == split]
            tone_means = split_data.groupby('Skin Tone')['Value'].mean()
            
            if len(tone_means) > 1:
                max_perf = tone_means.max()
                min_perf = tone_means.min()
                disparity = max_perf - min_perf
                std_dev = tone_means.std()
                
                disparity_data.append({
                    'Model': model,
                    'Split': split,
                    'Max_Performance': max_perf,
                    'Min_Performance': min_perf,
                    'Disparity': disparity,
                    'Std_Dev': std_dev
                })
    
    if not disparity_data:
        print("No disparity data could be computed")
        return
    
    disparity_df = pd.DataFrame(disparity_data)
    
    # Group by model and compute statistics
    disparity_stats = disparity_df.groupby('Model')['Disparity'].agg(['mean', 'std', 'count']).reset_index()
    disparity_stats['stderr'] = disparity_stats['std'] / np.sqrt(disparity_stats['count'])
    disparity_stats['stderr'] = disparity_stats['stderr'].fillna(0)
    
    # Create disparity plot
    plt.figure(figsize=(10, 6))
    
    x_pos = np.arange(len(disparity_stats))
    
    # Only show error bars if we have multiple splits
    if has_splits and disparity_stats['count'].max() > 1:
        bars = plt.bar(x_pos, disparity_stats['mean'], yerr=disparity_stats['stderr'],
                       capsize=4, alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1)
        error_info = "with error bars from multiple splits"
    else:
        bars = plt.bar(x_pos, disparity_stats['mean'], 
                       alpha=0.8, color='lightcoral', edgecolor='black', linewidth=1)
        error_info = "(single split - no error bars)"
    
    plt.xlabel('Model Architecture', fontsize=12, fontweight='bold')
    plt.ylabel(f'Skin Tone Disparity ({metric_name} %)', fontsize=12, fontweight='bold')
    plt.title(f'Skin Tone Fairness: Performance Disparity Across Models\n(Lower is more fair) {error_info}', 
              fontsize=14, fontweight='bold')
    plt.xticks(x_pos, disparity_stats['Model'], rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, mean_val) in enumerate(zip(bars, disparity_stats['mean'])):
        if has_splits and disparity_stats.iloc[i]['count'] > 1:
            std_val = disparity_stats.iloc[i]['std']
            stderr_val = disparity_stats.iloc[i]['stderr']
            label_text = f'{mean_val:.1f}±{std_val:.1f}'
            y_pos = mean_val + stderr_val + 0.5
        else:
            label_text = f'{mean_val:.1f}'
            y_pos = mean_val + 0.5
            
        plt.text(bar.get_x() + bar.get_width()/2., y_pos,
                label_text, ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'skin_tone_disparity_analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: skin_tone_disparity_analysis.png")

def plot_model_comparison_with_errors(combined_data, output_directory):
    """
    Compare all models across conditions with error bars
    """
    if 'ConditionSensitivities' not in combined_data or combined_data['ConditionSensitivities'].empty:
        print("No condition sensitivity data found for model comparison")
        return
    
    cond_df = combined_data['ConditionSensitivities']
    
    # Focus on Top-1 Sensitivity for baseline model
    top1_data = cond_df[cond_df['Metric'] == 'Top-1 Sensitivity'].copy()
    top1_data['Model'] = top1_data['Model'].str.replace('train_', '')
    
    if top1_data.empty:
        print("No Top-1 sensitivity data found for model comparison")
        return
    
    # Check if Split column exists
    has_splits = 'Split' in top1_data.columns
    if not has_splits:
        top1_data['Split'] = 0  # Add dummy split column
    
    # Group by model and condition, compute statistics
    stats = top1_data.groupby(['Model', 'Condition'])['Value'].agg(['mean', 'std', 'count']).reset_index()
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
    stats['stderr'] = stats['stderr'].fillna(0)
    
    # Get unique models and conditions
    models = sorted(stats['Model'].unique())
    conditions = sorted(stats['Condition'].unique())
    
    # Create subplot for each condition
    n_conditions = len(conditions)
    n_cols = 3
    n_rows = (n_conditions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        
        # Get data for this condition
        cond_stats = stats[stats['Condition'] == condition]
        
        if not cond_stats.empty:
            x_pos = np.arange(len(models))
            values = []
            errors = []
            
            for model in models:
                model_data = cond_stats[cond_stats['Model'] == model]
                if not model_data.empty:
                    values.append(model_data['mean'].iloc[0])
                    # Only add error bars if we have multiple splits
                    if has_splits and model_data['count'].iloc[0] > 1:
                        errors.append(model_data['stderr'].iloc[0])
                    else:
                        errors.append(0)
                else:
                    values.append(0)
                    errors.append(0)
            
            # Only show error bars if we have error data
            if has_splits and any(err > 0 for err in errors):
                bars = ax.bar(x_pos, values, yerr=errors, capsize=4, alpha=0.8)
            else:
                bars = ax.bar(x_pos, values, alpha=0.8)
            
            ax.set_title(f'{condition.title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Sensitivity (%)', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
                if val > 0:
                    y_pos = val + err + 1 if err > 0 else val + 1
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)
    
    # Update title based on whether we have splits
    if has_splits and stats['count'].max() > 1:
        title_suffix = "Multiple Splits"
    else:
        title_suffix = "Single Split"
        
    plt.suptitle(f'Model Comparison: Condition-Specific Performance Across {title_suffix}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'model_comparison_by_condition.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison_by_condition.png")
    """
    Compare all models across conditions with error bars
    """
    if 'ConditionSensitivities' not in combined_data or combined_data['ConditionSensitivities'].empty:
        print("No condition sensitivity data found for model comparison")
        return
    
    cond_df = combined_data['ConditionSensitivities']
    
    # Focus on Top-1 Sensitivity
    top1_data = cond_df[cond_df['Metric'] == 'Top-1 Sensitivity'].copy()
    top1_data['Model'] = top1_data['Model'].str.replace('train_', '')
    
    if top1_data.empty:
        return
    
    # Group by model and condition, compute statistics
    stats = top1_data.groupby(['Model', 'Condition'])['Value'].agg(['mean', 'std', 'count']).reset_index()
    stats['stderr'] = stats['std'] / np.sqrt(stats['count'])
    
    # Get unique models and conditions
    models = sorted(stats['Model'].unique())
    conditions = sorted(stats['Condition'].unique())
    
    # Create subplot for each condition
    n_conditions = len(conditions)
    n_cols = 3
    n_rows = (n_conditions + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, condition in enumerate(conditions):
        ax = axes[i]
        
        # Get data for this condition
        cond_stats = stats[stats['Condition'] == condition]
        
        if not cond_stats.empty:
            x_pos = np.arange(len(models))
            values = []
            errors = []
            
            for model in models:
                model_data = cond_stats[cond_stats['Model'] == model]
                if not model_data.empty:
                    values.append(model_data['mean'].iloc[0])
                    errors.append(model_data['stderr'].iloc[0])
                else:
                    values.append(0)
                    errors.append(0)
            
            bars = ax.bar(x_pos, values, yerr=errors, capsize=4, alpha=0.8)
            ax.set_title(f'{condition.title()}', fontsize=12, fontweight='bold')
            ax.set_ylabel('Sensitivity (%)', fontsize=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for j, (bar, val, err) in enumerate(zip(bars, values, errors)):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., val + err + 1,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Hide unused subplots
    for i in range(len(conditions), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Model Comparison: Condition-Specific Performance Across Multiple Splits', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'model_comparison_by_condition.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: model_comparison_by_condition.png")

def generate_summary_statistics(combined_data, output_directory):
    """
    Generate summary statistics table for balanced accuracy by skin tone
    """
    if 'StratifiedSensitivities' not in combined_data or combined_data['StratifiedSensitivities'].empty:
        print("No stratified data available for summary statistics")
        return
    
    stratified_df = combined_data['StratifiedSensitivities']
    
    # Look for accuracy data first, fall back to sensitivity
    accuracy_data = stratified_df[stratified_df['Metric'] == 'Top-1 Accuracy']
    if accuracy_data.empty:
        accuracy_data = stratified_df[stratified_df['Metric'] == 'Top-1 Sensitivity']
        metric_name = 'Sensitivity'
    else:
        metric_name = 'Accuracy'
    
    if accuracy_data.empty:
        print("No accuracy or sensitivity data available for summary statistics")
        return
    
    # Clean model names
    accuracy_data = accuracy_data.copy()
    accuracy_data['Model'] = accuracy_data['Model'].str.replace('train_', '')
    
    # Check if Split column exists
    has_splits = 'Split' in accuracy_data.columns
    if not has_splits:
        print("No split information found, computing statistics without split variance")
        accuracy_data['Split'] = 0  # Add dummy split column
    
    # Overall summary (averaged across all skin tones)
    overall_summary = accuracy_data.groupby(['Model', 'Split'])['Value'].mean().reset_index()
    summary_stats = overall_summary.groupby('Model')['Value'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    
    summary_stats.columns = ['Model', 'N_Splits', f'Mean_{metric_name}', f'Std_{metric_name}', 
                           f'Min_{metric_name}', f'Max_{metric_name}']
    summary_stats['Range'] = summary_stats[f'Max_{metric_name}'] - summary_stats[f'Min_{metric_name}']
    summary_stats['CV'] = (summary_stats[f'Std_{metric_name}'] / summary_stats[f'Mean_{metric_name}']) * 100
    
    # Handle case where there's only one split (std will be 0 or NaN)
    if not has_splits or summary_stats['N_Splits'].max() <= 1:
        summary_stats['CV'] = 0.0
        summary_stats[f'Std_{metric_name}'] = 0.0
    
    # Skin tone disparity summary
    disparity_summary = []
    for model in accuracy_data['Model'].unique():
        model_data = accuracy_data[accuracy_data['Model'] == model]
        # Compute disparity for each split, then average
        split_disparities = []
        for split in model_data['Split'].unique():
            split_data = model_data[model_data['Split'] == split]
            tone_means = split_data.groupby('Skin Tone')['Value'].mean()
            if len(tone_means) > 1:
                disparity = tone_means.max() - tone_means.min()
                split_disparities.append(disparity)
        
        if split_disparities:
            disparity_summary.append({
                'Model': model,
                'Mean_Disparity': np.mean(split_disparities),
                'Std_Disparity': np.std(split_disparities) if len(split_disparities) > 1 else 0.0,
                'Max_Disparity': np.max(split_disparities),
                'Min_Disparity': np.min(split_disparities)
            })
    
    disparity_df = pd.DataFrame(disparity_summary)
    
    # Round to 2 decimal places
    numeric_columns = [col for col in summary_stats.columns if col not in ['Model', 'N_Splits']]
    for col in numeric_columns:
        summary_stats[col] = summary_stats[col].round(2)
    
    if not disparity_df.empty:
        for col in ['Mean_Disparity', 'Std_Disparity', 'Max_Disparity', 'Min_Disparity']:
            disparity_df[col] = disparity_df[col].round(2)
    
    # Save to CSV
    #summary_stats.to_csv(os.path.join(output_directory, 'summary_statistics.csv'), index=False)
    #if not disparity_df.empty:
        #disparity_df.to_csv(os.path.join(output_directory, 'skin_tone_disparity_summary.csv'), index=False)
    
    print("✓ Saved: summary_statistics.csv")
    if not disparity_df.empty:
        print("✓ Saved: skin_tone_disparity_summary.csv")
    
    # Print summaries
    n_splits = summary_stats['N_Splits'].max()
    split_info = f"across {n_splits} splits" if has_splits and n_splits > 1 else "(single split)"
    
    print(f"\n=== {metric_name} Summary Statistics {split_info} ===")
    print("Overall Performance (averaged across all skin tones):")
    print(summary_stats.to_string(index=False))
    
    if not disparity_df.empty:
        print(f"\nSkin Tone Disparity Summary (FST Max - Min per split):")
        print(disparity_df.to_string(index=False))

def main():
    """
    Main function to generate balanced accuracy analysis with multiple train-test splits
    Produces:
    - balanced_accuracy_overall.png: Overall balanced accuracy per model
    - balanced_accuracy_by_skin_tone.png: Balanced accuracy by FST per model
    """
    # Configuration
    log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
    output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Overall"
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    print("=== Overall Analysis with Multiple Train-Test Splits ===")
    
    # Find all relevant log files
    log_files = find_log_files(log_directory, 'combined_all')
    
    if not log_files:
        print("No log files found! Please ensure you have files like:")
        print("  - combined_all.txt")
        print("  - combined_all_0.txt") 
        print("  - combined_all_1.txt")
        print("  - etc.")
        return
    
    # Parse all log files
    all_data = parse_multiple_logs(log_files)
    
    if not all_data:
        print("No data could be parsed from log files!")
        return
    
    # Combine data and compute statistics
    print("\nCombining data from all splits...")
    combined_data = combine_data_with_stats(all_data)
    
    print(f"Combined data types: {list(combined_data.keys())}")
    
    # Generate the two specific plots requested
    print("\nGenerating balanced accuracy plots...")
    
    plot1_success = plot_balanced_accuracy_overall(combined_data, output_directory)
    plot2_success = plot_balanced_accuracy_by_skin_tone(combined_data, output_directory)
    
    if not plot1_success and not plot2_success:
        print("ERROR: Both plots failed to generate!")
        print("Available data types:", list(combined_data.keys()))
        if 'StratifiedSensitivities' in combined_data:
            print("StratifiedSensitivities columns:", list(combined_data['StratifiedSensitivities'].columns))
            print("Available metrics:", combined_data['StratifiedSensitivities']['Metric'].unique())
    elif not plot1_success:
        print("WARNING: Overall balanced accuracy plot failed to generate")
    elif not plot2_success:
        print("WARNING: Skin tone balanced accuracy plot failed to generate")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_statistics(combined_data, output_directory)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Results saved to: {output_directory}")
    print(f"Generated plots:")
    print(f"  - balanced_accuracy_overall.png: Overall balanced accuracy per model")
    print(f"  - balanced_accuracy_by_skin_tone.png: Balanced accuracy by FST per model") 
    if len(log_files) > 1:
        print(f"  - Error bars show variance across {len(log_files)} train-test splits")
    else:
        print(f"  - Single train-test split (no error bars)")

if __name__ == "__main__":
    main()