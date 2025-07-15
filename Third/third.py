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

# Configuration
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Third"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define all combined files and their descriptive names
dataset_combinations = {
    'combined_all.txt': 'All Datasets',
    'combined_minus_dermie.txt': 'Without Dermie',
    'combined_minus_scin.txt': 'Without SCIN',
    'combined_minus_pad.txt': 'Without PAD',
    'combined_minus_fitz.txt': 'Without Fitzpatrick',
    'combined_minus_india.txt': 'Without India'
}

# Dictionary to store all data
all_data = {}

# Load data from all dataset combinations
print("Loading data from all dataset combinations...")
for filename, description in dataset_combinations.items():
    log_path = os.path.join(log_directory, filename)
    
    if os.path.exists(log_path):
        print(f"Loading {description} from {filename}...")
        try:
            df_dict = parse_combined_log(log_path)
            all_data[description] = df_dict
            print(f"  Successfully loaded {description}")
        except Exception as e:
            print(f"  Error loading {filename}: {e}")
    else:
        print(f"  Warning: {filename} not found")

if not all_data:
    print("No data files found! Please check file paths.")
    sys.exit(1)

# Use the first available dataset for the main analysis
primary_dataset = next(iter(all_data.values()))
avg_sensitivities = primary_dataset['AverageSensitivities']
skin_tone_accuracies = primary_dataset['SkinToneAccuracies'] 
stratified_sensitivities = primary_dataset['StratifiedSensitivities']

print(f"Found {len(avg_sensitivities['Model'].unique())} models in primary dataset")
print(f"Models: {avg_sensitivities['Model'].unique()}")

# 1. Overall Performance Comparison (Top-1, Top-3, Top-5) - Same as before
print("\n=== Overall Performance Analysis ===")

# Filter for sensitivity metrics
overall_performance = avg_sensitivities[avg_sensitivities['Metric'].isin(['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity'])].copy()

# Create pivot table for easier plotting
performance_pivot = overall_performance.pivot_table(values='Value', index='Model', columns='Metric', fill_value=0)

# Plot 1: Overall Performance Comparison
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(performance_pivot.index))
width = 0.25

metrics = ['Top-1 Sensitivity', 'Top-3 Sensitivity', 'Top-5 Sensitivity']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

for i, metric in enumerate(metrics):
    if metric in performance_pivot.columns:
        ax.bar(x + i*width, performance_pivot[metric], width, label=metric, color=colors[i], alpha=0.8)

ax.set_xlabel('Model Architecture')
ax.set_ylabel('Sensitivity (%)')
ax.set_title('Overall Model Performance: Top-1, Top-3, and Top-5 Sensitivity')
ax.set_xticks(x + width)
ax.set_xticklabels(performance_pivot.index, rotation=45, ha='right')
ax.legend()
ax.grid(True, alpha=0.3)

# Add value labels on bars
for i, metric in enumerate(metrics):
    if metric in performance_pivot.columns:
        for j, v in enumerate(performance_pivot[metric]):
            ax.text(j + i*width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'overall_performance_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# Find the best performing model (highest Top-1 sensitivity)
best_model_top1 = performance_pivot['Top-1 Sensitivity'].idxmax()
best_performance = performance_pivot.loc[best_model_top1, 'Top-1 Sensitivity']
print(f"Best performing model: {best_model_top1} with {best_performance:.2f}% Top-1 sensitivity")

# Determine baseline model (assuming it's the first model or has "baseline" in name)
baseline_model = None
for model in avg_sensitivities['Model'].unique():
    if 'baseline' in model.lower() or 'resnet' in model.lower():
        baseline_model = model
        break

if baseline_model is None:
    baseline_model = avg_sensitivities['Model'].unique()[0]  # Use first model as baseline
    print(f"No explicit baseline model found, using first model: {baseline_model}")
else:
    print(f"Using baseline model: {baseline_model}")

# 2. Skin Tone Performance for All Models (same as before)
print(f"\n=== Skin Tone Analysis for All Models ===")

# Get all available models from skin tone data
available_models = skin_tone_accuracies['Model'].unique()

if len(skin_tone_accuracies) > 0:
    # Create a comprehensive plot showing all models
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    accuracy_metrics = ['Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for idx, model in enumerate(available_models[:4]):  # Limit to 4 models for subplot layout
        model_data = skin_tone_accuracies[skin_tone_accuracies['Model'] == model]
        
        if len(model_data) > 0:
            skin_tone_pivot = model_data.pivot_table(values='Value', index='Skin Tone', columns='Metric', fill_value=0)
            
            ax = axes[idx]
            skin_tones = skin_tone_pivot.index
            x = np.arange(len(skin_tones))
            width = 0.25
            
            for i, metric in enumerate(accuracy_metrics):
                if metric in skin_tone_pivot.columns:
                    ax.bar(x + i*width, skin_tone_pivot[metric], width, label=metric, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Skin Tone (Fitzpatrick Scale)')
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(f'Performance Across Skin Tones: {model}')
            ax.set_xticks(x + width)
            ax.set_xticklabels([f'Type {int(tone)}' for tone in skin_tones])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, metric in enumerate(accuracy_metrics):
                if metric in skin_tone_pivot.columns:
                    for j, v in enumerate(skin_tone_pivot[metric]):
                        if v > 0:  # Only show non-zero values
                            ax.text(j + i*width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(available_models), 4):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'skin_tone_performance_all_models.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create individual plots for each model
    for model in available_models:
        model_data = skin_tone_accuracies[skin_tone_accuracies['Model'] == model]
        
        if len(model_data) > 0:
            skin_tone_pivot = model_data.pivot_table(values='Value', index='Skin Tone', columns='Metric', fill_value=0)
            
            plt.figure(figsize=(12, 8))
            skin_tones = skin_tone_pivot.index
            x = np.arange(len(skin_tones))
            width = 0.25
            
            for i, metric in enumerate(accuracy_metrics):
                if metric in skin_tone_pivot.columns:
                    plt.bar(x + i*width, skin_tone_pivot[metric], width, label=metric, color=colors[i], alpha=0.8)
            
            plt.xlabel('Skin Tone (Fitzpatrick Scale)')
            plt.ylabel('Accuracy (%)')
            plt.title(f'Performance Across Skin Tones: {model}')
            plt.xticks(x + width, [f'Type {int(tone)}' for tone in skin_tones])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, metric in enumerate(accuracy_metrics):
                if metric in skin_tone_pivot.columns:
                    for j, v in enumerate(skin_tone_pivot[metric]):
                        if v > 0:  # Only show non-zero values
                            plt.text(j + i*width, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            safe_model_name = model.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.savefig(os.path.join(output_directory, f'skin_tone_performance_{safe_model_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Skin tone performance saved for model: {model}")

    # Create a comparison plot showing Top-1 accuracy across all models and skin tones
    plt.figure(figsize=(14, 8))
    
    # Filter for Top-1 Accuracy only
    top1_skin_data = skin_tone_accuracies[skin_tone_accuracies['Metric'] == 'Top-1 Accuracy']
    
    if len(top1_skin_data) > 0:
        # Create pivot table: models as groups, skin tones as x-axis
        comparison_pivot = top1_skin_data.pivot_table(values='Value', index='Skin Tone', columns='Model', fill_value=0)
        
        skin_tones = comparison_pivot.index
        models = comparison_pivot.columns
        x = np.arange(len(skin_tones))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            plt.bar(x + i*width, comparison_pivot[model], width, label=model, alpha=0.8)
        
        plt.xlabel('Skin Tone (Fitzpatrick Scale)')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.title('Top-1 Accuracy Comparison Across All Models and Skin Tones')
        plt.xticks(x + width * (len(models) - 1) / 2, [f'Type {int(tone)}' for tone in skin_tones])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'skin_tone_comparison_all_models.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Skin tone comparison across all models saved")

# 3. Original Heatmaps: Skin Condition x Skin Tone for Each Model
print(f"\n=== Creating Original Condition x Skin Tone Heatmaps ===")

# Filter for Top-1 Sensitivity only
top1_stratified = stratified_sensitivities[stratified_sensitivities['Metric'] == 'Top-1 Sensitivity'].copy()

if len(top1_stratified) > 0:
    models = top1_stratified['Model'].unique()
    
    # Create a heatmap for each model
    for model in models:
        model_data = top1_stratified[top1_stratified['Model'] == model]
        
        # Create pivot table: conditions as rows, skin tones as columns
        heatmap_data = model_data.pivot_table(values='Value', index='Condition', columns='Skin Tone', fill_value=np.nan)
        
        # Only create heatmap if we have data
        if not heatmap_data.empty:
            plt.figure(figsize=(10, 8))
            
            # Create heatmap
            mask = heatmap_data.isnull()
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt='.1f', 
                       cmap='RdYlBu_r',
                       mask=mask,
                       cbar_kws={'label': 'Top-1 Sensitivity (%)'})
            
            plt.title(f'Top-1 Sensitivity: Skin Conditions vs Skin Tones\nModel: {model}')
            plt.xlabel('Skin Tone (Fitzpatrick Scale)')
            plt.ylabel('Skin Conditions')
            
            # Improve skin tone labels
            current_labels = plt.gca().get_xticklabels()
            new_labels = [f'Type {label.get_text()}' for label in current_labels]
            plt.gca().set_xticklabels(new_labels)
            
            plt.tight_layout()
            
            # Save with model name in filename
            safe_model_name = model.replace('/', '_').replace('\\', '_').replace(':', '_')
            plt.savefig(os.path.join(output_directory, f'heatmap_condition_tone_{safe_model_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Heatmap saved for model: {model}")

# 4. NEW: Baseline Model Heatmaps Across All Dataset Combinations
print(f"\n=== Creating Baseline Model Heatmaps Across Dataset Combinations ===")

# Create output subdirectory for baseline comparison
baseline_output_dir = os.path.join(output_directory, 'baseline_comparison')
os.makedirs(baseline_output_dir, exist_ok=True)

# Store baseline heatmap data for all combinations
baseline_heatmaps = {}

for dataset_name, data_dict in all_data.items():
    print(f"Processing baseline model for {dataset_name}...")
    
    # Get stratified data for this dataset
    dataset_stratified = data_dict['StratifiedSensitivities']
    
    # Filter for baseline model and Top-1 Sensitivity
    baseline_data = dataset_stratified[
        (dataset_stratified['Model'] == baseline_model) & 
        (dataset_stratified['Metric'] == 'Top-1 Sensitivity')
    ].copy()
    
    if len(baseline_data) > 0:
        # Create pivot table: conditions as rows, skin tones as columns
        heatmap_data = baseline_data.pivot_table(values='Value', index='Condition', columns='Skin Tone', fill_value=np.nan)
        
        if not heatmap_data.empty:
            # Store for later comparison
            baseline_heatmaps[dataset_name] = heatmap_data
            
            # Create individual heatmap
            plt.figure(figsize=(10, 8))
            
            mask = heatmap_data.isnull()
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt='.1f', 
                       cmap='RdYlBu_r',
                       mask=mask,
                       vmin=0, vmax=100,  # Consistent scale across all heatmaps
                       cbar_kws={'label': 'Top-1 Sensitivity (%)'})
            
            plt.title(f'Baseline Model ({baseline_model}) - Top-1 Sensitivity\nDataset: {dataset_name}')
            plt.xlabel('Skin Tone (Fitzpatrick Scale)')
            plt.ylabel('Skin Conditions')
            
            # Improve skin tone labels
            current_labels = plt.gca().get_xticklabels()
            new_labels = [f'Type {label.get_text()}' for label in current_labels]
            plt.gca().set_xticklabels(new_labels)
            
            plt.tight_layout()
            
            # Save individual heatmap
            safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(baseline_output_dir, f'baseline_heatmap_{safe_dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Baseline heatmap saved for {dataset_name}")
        else:
            print(f"  No stratified data available for {dataset_name}")
    else:
        print(f"  No baseline model data found for {dataset_name}")

# Create a comprehensive comparison plot showing all dataset combinations
if len(baseline_heatmaps) > 1:
    print(f"\nCreating comprehensive baseline comparison across {len(baseline_heatmaps)} datasets...")
    
    # Calculate the grid size for subplots
    n_datasets = len(baseline_heatmaps)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 8, n_rows * 6))
    
    # Handle case where we have only one row
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # Find global min/max for consistent color scaling
    all_values = []
    for heatmap_data in baseline_heatmaps.values():
        all_values.extend(heatmap_data.values.flatten())
    all_values = [v for v in all_values if not np.isnan(v)]
    
    if all_values:
        vmin, vmax = min(all_values), max(all_values)
    else:
        vmin, vmax = 0, 100
    
    for idx, (dataset_name, heatmap_data) in enumerate(baseline_heatmaps.items()):
        ax = axes[idx]
        
        mask = heatmap_data.isnull()
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='RdYlBu_r',
                   mask=mask,
                   vmin=vmin, vmax=vmax,
                   ax=ax,
                   cbar_kws={'label': 'Top-1 Sensitivity (%)'})
        
        ax.set_title(f'{dataset_name}')
        ax.set_xlabel('Skin Tone (Fitzpatrick Scale)')
        ax.set_ylabel('Skin Conditions')
        
        # Improve skin tone labels
        current_labels = ax.get_xticklabels()
        new_labels = [f'Type {label.get_text()}' for label in current_labels]
        ax.set_xticklabels(new_labels)
    
    # Hide unused subplots
    for idx in range(len(baseline_heatmaps), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Baseline Model ({baseline_model}) - Top-1 Sensitivity Comparison\nAcross Different Dataset Combinations', 
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(baseline_output_dir, 'baseline_comparison_all_datasets.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive baseline comparison saved")

# 5. Dataset Impact Analysis
print(f"\n=== Dataset Impact Analysis ===")

if len(baseline_heatmaps) > 1:
    # Calculate performance differences when datasets are removed
    all_datasets_key = 'All Datasets'
    
    if all_datasets_key in baseline_heatmaps:
        reference_data = baseline_heatmaps[all_datasets_key]
        
        print(f"Impact of removing individual datasets (compared to {all_datasets_key}):")
        
        for dataset_name, heatmap_data in baseline_heatmaps.items():
            if dataset_name != all_datasets_key:
                # Calculate differences
                diff_data = heatmap_data - reference_data
                
                # Create difference heatmap
                plt.figure(figsize=(10, 8))
                
                mask = diff_data.isnull()
                sns.heatmap(diff_data, 
                           annot=True, 
                           fmt='.1f', 
                           cmap='RdBu_r',
                           center=0,
                           mask=mask,
                           cbar_kws={'label': 'Sensitivity Difference (%)'})
                
                plt.title(f'Performance Difference: {dataset_name} vs {all_datasets_key}\nBaseline Model ({baseline_model})')
                plt.xlabel('Skin Tone (Fitzpatrick Scale)')
                plt.ylabel('Skin Conditions')
                
                # Improve skin tone labels
                current_labels = plt.gca().get_xticklabels()
                new_labels = [f'Type {label.get_text()}' for label in current_labels]
                plt.gca().set_xticklabels(new_labels)
                
                plt.tight_layout()
                
                safe_dataset_name = dataset_name.replace(' ', '_').replace('/', '_')
                plt.savefig(os.path.join(baseline_output_dir, f'difference_{safe_dataset_name}_vs_all.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Calculate summary statistics
                valid_diffs = diff_data.values.flatten()
                valid_diffs = valid_diffs[~np.isnan(valid_diffs)]
                
                if len(valid_diffs) > 0:
                    mean_diff = np.mean(valid_diffs)
                    std_diff = np.std(valid_diffs)
                    max_improvement = np.max(valid_diffs)
                    max_degradation = np.min(valid_diffs)
                    
                    print(f"  {dataset_name}:")
                    print(f"    Mean difference: {mean_diff:.2f}%")
                    print(f"    Std deviation: {std_diff:.2f}%")
                    print(f"    Max improvement: {max_improvement:.2f}%")
                    print(f"    Max degradation: {max_degradation:.2f}%")
                    print(f"    Net impact: {'Positive' if mean_diff > 0 else 'Negative'}")

# 6. Summary Statistics (Enhanced)
print(f"\n=== Enhanced Summary Statistics ===")

# Overall performance summary
print("Overall Performance Rankings (Top-1 Sensitivity):")
top1_ranking = performance_pivot['Top-1 Sensitivity'].sort_values(ascending=False)
for i, (model, score) in enumerate(top1_ranking.items()):
    print(f"  {i+1}. {model}: {score:.2f}%")

# Performance improvement from Top-1 to Top-5
print(f"\nPerformance Improvement (Top-1 to Top-5):")
for model in performance_pivot.index:
    top1 = performance_pivot.loc[model, 'Top-1 Sensitivity']
    top5 = performance_pivot.loc[model, 'Top-5 Sensitivity']
    improvement = top5 - top1
    print(f"  {model}: +{improvement:.2f}% (from {top1:.2f}% to {top5:.2f}%)")

# Baseline model performance across datasets
if len(baseline_heatmaps) > 1:
    print(f"\nBaseline Model ({baseline_model}) Performance Across Datasets:")
    
    for dataset_name, heatmap_data in baseline_heatmaps.items():
        valid_values = heatmap_data.values.flatten()
        valid_values = valid_values[~np.isnan(valid_values)]
        
        if len(valid_values) > 0:
            mean_perf = np.mean(valid_values)
            std_perf = np.std(valid_values)
            min_perf = np.min(valid_values)
            max_perf = np.max(valid_values)
            
            print(f"  {dataset_name}:")
            print(f"    Mean performance: {mean_perf:.2f}%")
            print(f"    Std deviation: {std_perf:.2f}%")
            print(f"    Range: {min_perf:.2f}% - {max_perf:.2f}%")

# Skin tone fairness analysis for all models
if len(skin_tone_accuracies) > 0:
    print(f"\nSkin Tone Fairness Analysis for All Models:")
    
    top1_skin_data = skin_tone_accuracies[skin_tone_accuracies['Metric'] == 'Top-1 Accuracy']
    
    for model in top1_skin_data['Model'].unique():
        model_data = top1_skin_data[top1_skin_data['Model'] == model]
        if len(model_data) > 0:
            model_pivot = model_data.pivot_table(values='Value', index='Skin Tone', columns='Model', fill_value=0)
            if model in model_pivot.columns:
                accuracy_by_tone = model_pivot[model]
                print(f"\n  {model}:")
                print(f"    Best performing skin tone: Type {accuracy_by_tone.idxmax()} ({accuracy_by_tone.max():.2f}%)")
                print(f"    Worst performing skin tone: Type {accuracy_by_tone.idxmin()} ({accuracy_by_tone.min():.2f}%)")
                print(f"    Performance gap: {accuracy_by_tone.max() - accuracy_by_tone.min():.2f}%")
                print(f"    Standard deviation across skin tones: {accuracy_by_tone.std():.2f}%")

print(f"\nAll figures saved to: {output_directory}")
print(f"Baseline comparison figures saved to: {baseline_output_dir}")

# Display available data for verification
print(f"\n=== Data Verification ===")
print(f"Number of dataset combinations loaded: {len(all_data)}")
print(f"Dataset combinations: {list(all_data.keys())}")
print(f"Baseline model used: {baseline_model}")

for dataset_name, data_dict in all_data.items():
    avg_sens = data_dict['AverageSensitivities']
    skin_acc = data_dict['SkinToneAccuracies']
    strat_sens = data_dict['StratifiedSensitivities']
    
    print(f"\n{dataset_name}:")
    print(f"  Average sensitivities shape: {avg_sens.shape}")
    print(f"  Skin tone accuracies shape: {skin_acc.shape}")
    print(f"  Stratified sensitivities shape: {strat_sens.shape}")
    print(f"  Models: {avg_sens['Model'].unique()}")