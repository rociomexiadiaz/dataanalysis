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

# Configuration - Third folder structure
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Overall"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Fitzpatrick Skin Tone color mapping
fst_color_map = {
    1.0: '#F5D5A0',
    2.0: '#E4B589',
    3.0: '#D1A479',
    4.0: '#C0874F',
    5.0: '#A56635',
    6.0: '#4C2C27'
}

def parse_all_log_files(log_directory):
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
        'BalancedAccuraciesBySkinTone': [],  # NEW
        'F1Scores': []  # NEW
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
                df_copy = df_dict[key].copy()
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

def create_overall_performance_comparison(df_dict, output_directory):
    """
    Create overall performance comparison across models with Top-1, Top-3, and Top-5 metrics
    """
    print("\n=== Creating Overall Performance Comparison ===")
    
    # Use condition sensitivities data
    if 'ConditionSensitivities' not in df_dict or df_dict['ConditionSensitivities'].empty:
        print("✗ No condition sensitivity data found for overall performance comparison")
        return
    
    condition_data = df_dict['ConditionSensitivities']
    
    # Filter for all Top-K sensitivities and 'all' dataset
    topk_data = condition_data[
        (condition_data['Metric'].isin(['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity'])) & 
        (condition_data['Datasets'] == 'all')
    ].copy()
    
    if len(topk_data) == 0:
        print("✗ No Top-K sensitivity data found for 'all' dataset")
        return
    
    # Clean model names and include FairDisCo
    topk_data['Model'] = topk_data['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo'
    })
    
    # Only filter out csv files, keep FairDisCo
    excluded_models = ['csv', 'CSV']
    topk_data = topk_data[~topk_data['Model'].isin(excluded_models)]
    
    if len(topk_data) == 0:
        print("✗ No valid model data after filtering")
        return
    
    # Calculate overall performance (average across all conditions) for each model and metric
    overall_performance = topk_data.groupby(['Model', 'Metric'])['Value'].agg(['mean', 'std']).reset_index()
    overall_performance.columns = ['Model', 'Metric', 'Mean_Sensitivity', 'Std_Sensitivity']
    overall_performance['Std_Sensitivity'] = overall_performance['Std_Sensitivity'].fillna(0)
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    models = sorted(overall_performance['Model'].unique())
    metrics = ['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity']
    metric_labels = ['Top-1', 'Top-3', 'Top-5']
    
    x = np.arange(len(models))
    width = 0.25  # Width of bars
    
    # Color palette for different metrics
    colors = ['#d62728', '#ff7f0e', '#2ca02c']  # Red, Orange, Green
    
    # Create bars for each metric
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        metric_data = overall_performance[overall_performance['Metric'] == metric]
        
        # Ensure we have data for all models
        means = []
        stds = []
        for model in models:
            model_data = metric_data[metric_data['Model'] == model]
            if len(model_data) > 0:
                means.append(model_data['Mean_Sensitivity'].iloc[0])
                stds.append(model_data['Std_Sensitivity'].iloc[0])
            else:
                means.append(0)
                stds.append(0)
        
        bars = plt.bar(x + i * width, means, width, yerr=stds,
                      label=label, alpha=0.8, color=color, capsize=5)
        
        # Add value labels on bars
        for j, (mean, std) in enumerate(zip(means, stds)):
            if mean > 0:  # Only add label if there's data
                plt.text(j + i * width, mean + std + 1, f'{mean:.1f}', 
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Average Sensitivity (%)', fontsize=12)
    plt.title('Overall Performance Comparison: Average Sensitivity Across All Conditions\n(All Datasets Combined)', fontsize=14)
    plt.xticks(x + width, models, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Set y-axis limits to accommodate labels
    max_val = max([max(overall_performance[overall_performance['Metric'] == metric]['Mean_Sensitivity'] + 
                      overall_performance[overall_performance['Metric'] == metric]['Std_Sensitivity']) 
                   for metric in metrics if len(overall_performance[overall_performance['Metric'] == metric]) > 0])
    plt.ylim(0, max_val * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'overall_performance_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: overall_performance_comparison.png")

def create_top1_sensitivity_heatmap(df_dict, output_directory):
    """
    Create heatmap of top1 sensitivity with y:condition, x:model, color:top1 sensitivity (red=0 to green=100)
    """
    print("\n=== Creating Top-1 Sensitivity Heatmap ===")
    
    # Use condition sensitivities data
    if 'ConditionSensitivities' not in df_dict or df_dict['ConditionSensitivities'].empty:
        print("✗ No condition sensitivity data found for heatmap")
        return
    
    condition_data = df_dict['ConditionSensitivities']
    
    # Filter for Top-1 sensitivity and 'all' dataset
    top1_data = condition_data[
        (condition_data['Metric'] == 'Top-1 Sensitivity') & 
        (condition_data['Datasets'] == 'all')
    ].copy()
    
    if len(top1_data) == 0:
        print("✗ No Top-1 sensitivity data found for 'all' dataset")
        return
    
    # Clean model names and include FairDisCo
    top1_data['Model'] = top1_data['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo'
    })
    
    # Only filter out csv files, keep FairDisCo
    excluded_models = ['csv', 'CSV']
    top1_data = top1_data[~top1_data['Model'].isin(excluded_models)]
    
    if len(top1_data) == 0:
        print("✗ No valid model data after filtering")
        return
    
    # Check for duplicates and handle them
    print(f"Data shape before duplicate handling: {top1_data.shape}")
    
    # Group by Model and Condition and take the mean if there are duplicates
    heatmap_data_df = top1_data.groupby(['Condition', 'Model'])['Value'].mean().reset_index()
    
    # Create pivot table for heatmap
    try:
        heatmap_data = heatmap_data_df.pivot(index='Condition', columns='Model', values='Value')
    except ValueError as e:
        print(f"Error creating pivot: {e}")
        print("Duplicate entries found. Let's investigate:")
        duplicates = top1_data.groupby(['Condition', 'Model']).size()
        print("Duplicate counts:")
        print(duplicates[duplicates > 1])
        
        # Use pivot_table with aggregation function
        heatmap_data = top1_data.pivot_table(index='Condition', columns='Model', values='Value', aggfunc='mean')
    
    print(f"Heatmap data shape: {heatmap_data.shape}")
    
    # Create the heatmap
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap from red (0) to green (100)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['red', 'yellow', 'green']
    custom_cmap = LinearSegmentedColormap.from_list('red_to_green', colors_list, N=256)
    
    # Create heatmap with custom colormap
    sns.heatmap(heatmap_data, 
                annot=True, 
                fmt='.1f', 
                cmap=custom_cmap,
                vmin=0,  # Set minimum to 0 (red)
                vmax=100,  # Set maximum to 100 (green)
                cbar_kws={'label': 'Top-1 Sensitivity (%)'},
                linewidths=0.5)
    
    plt.title('Top-1 Sensitivity Heatmap by Condition and Model\n(All Datasets Combined)', 
              fontsize=14, pad=20)
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Condition', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'top1_sensitivity_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: top1_sensitivity_heatmap.png")

def create_balanced_accuracy_plot(df_dict, output_directory):
    """
    Create balanced accuracy comparison plot - NEW FUNCTION
    """
    print("\n=== Creating Balanced Accuracy Plot ===")
    
    # Use balanced accuracies by skin tone data
    if 'BalancedAccuraciesBySkinTone' not in df_dict or df_dict['BalancedAccuraciesBySkinTone'].empty:
        print("✗ No balanced accuracy data found")
        return
    
    balanced_data = df_dict['BalancedAccuraciesBySkinTone']
    
    # Filter for 'all' dataset, Overall Balanced Accuracy, and exclude 'Overall' skin tone
    bal_acc_data = balanced_data[
        (balanced_data['Datasets'] == 'all') & 
        (balanced_data['Metric'] == 'Overall Balanced Accuracy') &
        (balanced_data['Skin Tone'] != 'Overall')  # Exclude overall, only skin tone specific
    ].copy()
    
    if len(bal_acc_data) == 0:
        print("✗ No balanced accuracy data found for 'all' dataset with specific skin tones")
        return
    
    # Clean model names and include FairDisCo
    bal_acc_data['Model'] = bal_acc_data['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo'
    })
    
    # Only filter out csv files, keep FairDisCo
    excluded_models = ['csv', 'CSV']
    bal_acc_data = bal_acc_data[~bal_acc_data['Model'].isin(excluded_models)]
    
    if len(bal_acc_data) == 0:
        print("✗ No valid model data after filtering")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Group by model and skin tone
    pivot_data = bal_acc_data.pivot_table(values='Value', index='Model', columns='Skin Tone', aggfunc='mean')
    
    if not pivot_data.empty:
        # Create grouped bar plot - now all columns should be numeric
        skin_tones = sorted(pivot_data.columns)  # This should work now
        models = pivot_data.index
        x = np.arange(len(models))
        width = 0.8 / len(skin_tones)
        
        for i, skin_tone in enumerate(skin_tones):
            values = [pivot_data.loc[model, skin_tone] if skin_tone in pivot_data.columns else 0 for model in models]
            color = fst_color_map.get(skin_tone, '#1f77b4')
            bars = plt.bar(x + i*width, values, width, label=f'FST {int(skin_tone)}', alpha=0.8, color=color)
            
            # Add value labels
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 0:
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                            f'{value:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Balanced Accuracy (%)', fontsize=12)
        plt.title('Balanced Accuracy by Model and Skin Tone\n(All Datasets Combined)', fontsize=14)
        plt.xticks(x + width * (len(skin_tones) - 1) / 2, models, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'balanced_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: balanced_accuracy_comparison.png")

def create_f1_score_heatmap(df_dict, output_directory):
    """
    Create F1 score heatmap in the same format as top1_sensitivity_heatmap - NEW FUNCTION
    """
    print("\n=== Creating F1 Score Heatmap ===")
    
    # Use F1 scores data
    if 'F1Scores' not in df_dict or df_dict['F1Scores'].empty:
        print("✗ No F1 score data found")
        return
    
    f1_data = df_dict['F1Scores']
    
    # Filter for 'all' dataset and regular F1 Score (condition-specific)
    f1_condition_data = f1_data[
        (f1_data['Datasets'] == 'all') & 
        (f1_data['Metric'] == 'F1 Score') &
        (f1_data['Condition'] != 'Overall')  # Exclude overall F1 scores
    ].copy()
    
    if len(f1_condition_data) == 0:
        print("✗ No condition-specific F1 score data found for 'all' dataset")
        return
    
    # Clean model names and include FairDisCo
    f1_condition_data['Model'] = f1_condition_data['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo'
    })
    
    # Only filter out csv files, keep FairDisCo
    excluded_models = ['csv', 'CSV']
    f1_condition_data = f1_condition_data[~f1_condition_data['Model'].isin(excluded_models)]
    
    if len(f1_condition_data) == 0:
        print("✗ No valid model data after filtering")
        return
    
    # Create pivot table for heatmap
    heatmap_data = f1_condition_data.pivot_table(values='Value', index='Condition', columns='Model', aggfunc='mean')
    
    if not heatmap_data.empty:
        # Create the heatmap
        plt.figure(figsize=(12, 8))
        
        # Use the same RdYlGn colormap as the sensitivity heatmap
        sns.heatmap(
            heatmap_data,
            cmap=sns.color_palette("RdYlGn", as_cmap=True),
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".1f",
            cbar_kws={'label': 'F1 Score (%)'},
            linewidths=0.5
        )
        
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Condition", fontsize=12)
        plt.title("F1 Score Heatmap by Condition and Model\n(All Datasets Combined)", fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'f1_score_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: f1_score_heatmap.png")

def main():
    """
    Main function - generates original plots plus new balanced accuracy and F1 plots
    """
    print("=== Enhanced Overall Analysis ===")
    print("Generating:")
    print("1. Overall Performance Comparison (ORIGINAL)")
    print("2. Top-1 Sensitivity Heatmap (ORIGINAL)")
    print("3. Balanced Accuracy Comparison (NEW)")
    print("4. F1 Score Heatmap (NEW)")
    
    # Parse all log files
    print("\nParsing all log files...")
    df_dict = parse_all_log_files(log_directory)
    
    print("Data parsing complete!")
    print(f"Available dataframes: {list(df_dict.keys())}")
    
    # Check if we have the required data
    if 'ConditionSensitivities' in df_dict and not df_dict['ConditionSensitivities'].empty:
        print(f"ConditionSensitivities shape: {df_dict['ConditionSensitivities'].shape}")
        print(f"Datasets found: {df_dict['ConditionSensitivities']['Datasets'].unique()}")
        print(f"Models found: {df_dict['ConditionSensitivities']['Model'].unique()}")
        print(f"Metrics found: {sorted(df_dict['ConditionSensitivities']['Metric'].unique())}")
        print(f"Conditions found: {sorted(df_dict['ConditionSensitivities']['Condition'].unique())}")
    else:
        print("✗ No ConditionSensitivities data found!")
        return
    
    # Check for new data types
    if 'BalancedAccuraciesBySkinTone' in df_dict and not df_dict['BalancedAccuraciesBySkinTone'].empty:
        print(f"BalancedAccuraciesBySkinTone shape: {df_dict['BalancedAccuraciesBySkinTone'].shape}")
    
    if 'F1Scores' in df_dict and not df_dict['F1Scores'].empty:
        print(f"F1Scores shape: {df_dict['F1Scores'].shape}")
    
    # Generate the original visualizations
    create_overall_performance_comparison(df_dict, output_directory)
    create_top1_sensitivity_heatmap(df_dict, output_directory)
    
    # Generate the new visualizations
    create_balanced_accuracy_plot(df_dict, output_directory)
    # Removed: create_balanced_accuracy_by_skin_tone_plot() - redundant with above
    create_f1_score_heatmap(df_dict, output_directory)
    
    # Summary statistics
    condition_data = df_dict['ConditionSensitivities']
    topk_data = condition_data[
        (condition_data['Metric'].isin(['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity'])) & 
        (condition_data['Datasets'] == 'all')
    ]
    
    print(f"\n=== Summary Statistics ===")
    print(f"Total results processed: {len(condition_data)}")
    
    if len(topk_data) > 0:
        # Clean model names for summary (including FairDisCo)
        topk_clean = topk_data.copy()
        topk_clean['Model'] = topk_clean['Model'].replace({
            'train_Baseline': 'Baseline',
            'train_VAE': 'VAE',
            'train_TABE': 'TABE',
            'train_FairDisCo': 'FairDisCo'
        })
        
        excluded_models = ['csv', 'CSV']
        topk_clean = topk_clean[~topk_clean['Model'].isin(excluded_models)]
        
        print(f"Models analyzed: {list(topk_clean['Model'].unique())}")
        print(f"Metrics analyzed: {list(topk_clean['Metric'].unique())}")
        print(f"Conditions analyzed: {len(topk_clean['Condition'].unique())}")
        
        # Show average for each metric
        for metric in ['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity']:
            metric_data = topk_clean[topk_clean['Metric'] == metric]
            if len(metric_data) > 0:
                print(f"Average {metric}: {metric_data['Value'].mean():.2f}%")
    
    print(f"\nAnalysis complete! Generated files saved to: {output_directory}")
    print("Files created:")
    print("- overall_performance_comparison.png (ORIGINAL)")
    print("- top1_sensitivity_heatmap.png (ORIGINAL)")
    print("- balanced_accuracy_comparison.png (NEW)")
    print("- f1_score_heatmap.png (NEW)")

if __name__ == "__main__":
    main()