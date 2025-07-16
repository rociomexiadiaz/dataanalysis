import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add parent directory for import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

# Config
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinConditionAnalysis"
log_files = [
    'combined_all.txt',
    'combined_minus_dermie.txt', 
    'combined_minus_fitz.txt',
    'combined_minus_india.txt',
    'combined_minus_pad.txt',
    'combined_minus_scin.txt'
]

os.makedirs(output_directory, exist_ok=True)

# Load data
dataset_mapping = {
    'combined_all.txt': 'All Datasets',
    'combined_minus_dermie.txt': 'Minus Dermie',
    'combined_minus_fitz.txt': 'Minus Fitz',
    'combined_minus_india.txt': 'Minus India',
    'combined_minus_pad.txt': 'Minus PAD',
    'combined_minus_scin.txt': 'Minus SCIN'
}

all_condition_sensitivities = []
all_avg_sensitivities = []

for log_file in log_files:
    log_path = os.path.join(log_directory, log_file)
    if os.path.exists(log_path):
        df_dict = parse_combined_log(log_path)
        dataset_name = dataset_mapping[log_file]
        df_dict['ConditionSensitivities']['Dataset_Config'] = dataset_name
        df_dict['AverageSensitivities']['Dataset_Config'] = dataset_name
        all_condition_sensitivities.append(df_dict['ConditionSensitivities'])
        all_avg_sensitivities.append(df_dict['AverageSensitivities'])

# Combine data
condition_sensitivities = pd.concat(all_condition_sensitivities, ignore_index=True)
avg_sensitivities = pd.concat(all_avg_sensitivities, ignore_index=True)

# Filter & rename
top1_conditions = condition_sensitivities[condition_sensitivities['Metric'] == 'Top-1 Sensitivity'].copy()
top1_conditions = top1_conditions[~top1_conditions['Model'].str.contains('air')]
top1_conditions['Model'] = top1_conditions['Model'].replace({
    'train_Baseline': 'Baseline',
    'train_VAE': 'VAE',
    'train_TABE': 'TABE'
})

baseline_config = 'All Datasets'
models = top1_conditions['Model'].unique()
conditions = top1_conditions['Condition'].unique()
dataset_configs = top1_conditions['Dataset_Config'].unique()

# Color palette for bars
custom_palette = ['#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']

# === Modified Architecture Analysis ===
arch_data = top1_conditions[top1_conditions['Dataset_Config'] == baseline_config].copy()
arch_stats = arch_data.groupby(['Condition', 'Model'])['Value'].agg(['mean', 'std', 'count']).reset_index()
arch_stats['std'] = arch_stats['std'].fillna(0)

# Get baseline values for each condition
baseline_values = {}
baseline_data = arch_stats[arch_stats['Model'] == 'Baseline']
for _, row in baseline_data.iterrows():
    baseline_values[row['Condition']] = row['mean']

# Calculate differences from baseline for other models
other_models = [model for model in models if model != 'Baseline']
differences_data = []

for model in other_models:
    model_data = arch_stats[arch_stats['Model'] == model]
    for _, row in model_data.iterrows():
        condition = row['Condition']
        if condition in baseline_values:
            diff = row['mean'] - baseline_values[condition]  # model - baseline
            differences_data.append({
                'Model': model,
                'Condition': condition,
                'Difference': diff
            })

# Convert to DataFrame
diff_df = pd.DataFrame(differences_data)

# Plot architecture differences
fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(conditions))
width = 0.8 / len(other_models)

for i, model in enumerate(other_models):
    model_data = diff_df[diff_df['Model'] == model]
    means = [model_data[model_data['Condition'] == cond]['Difference'].values[0] if cond in model_data['Condition'].values else 0 for cond in conditions]
    ax.bar(x + i*width, means, width, label=model, alpha=0.9, color=custom_palette[i % len(custom_palette)])

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax.set_xlabel('Skin Conditions')
ax.set_ylabel('Difference from Baseline (%)')
ax.set_title('Architecture Sensitivity: Difference from Baseline')
ax.set_xticks(x + width * (len(other_models) - 1) / 2)
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.legend()
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_sensitivity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Max-Min (Architecture) ===
disparity_max_min = arch_stats.groupby('Model')['mean'].agg(['max', 'min']).reset_index()
disparity_max_min['disparity'] = disparity_max_min['max'] - disparity_max_min['min']

plt.figure(figsize=(10, 6))
plt.bar(disparity_max_min['Model'], disparity_max_min['disparity'], color='#76b5c5')
plt.xlabel('Model Architecture')
plt.ylabel('Disparity (Max - Min) %')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_disparity_max_min.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Std Dev (Architecture) ===
disparity_std = arch_stats.groupby('Model')['mean'].std().reset_index()
disparity_std.columns = ['Model', 'std_disparity']

plt.figure(figsize=(10, 6))
plt.bar(disparity_std['Model'], disparity_std['std_disparity'], color='#76b5c5')
plt.xlabel('Model Architecture')
plt.ylabel('Disparity (Std Dev) %')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_disparity_std.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Modified Dataset Analysis ===
baseline_model = 'Baseline'
dataset_data = top1_conditions[top1_conditions['Model'] == baseline_model].copy()
dataset_stats = dataset_data.groupby(['Condition', 'Dataset_Config'])['Value'].agg(['mean', 'std', 'count']).reset_index()
dataset_stats['std'] = dataset_stats['std'].fillna(0)

# Get "All Datasets" baseline values for each condition
all_datasets_values = {}
all_datasets_data = dataset_stats[dataset_stats['Dataset_Config'] == 'All Datasets']
for _, row in all_datasets_data.iterrows():
    all_datasets_values[row['Condition']] = row['mean']

# Calculate differences from "All Datasets" baseline for other configurations
other_configs = [config for config in dataset_configs if config != 'All Datasets']
dataset_differences_data = []

for config in other_configs:
    config_data = dataset_stats[dataset_stats['Dataset_Config'] == config]
    for _, row in config_data.iterrows():
        condition = row['Condition']
        if condition in all_datasets_values:
            diff = all_datasets_values[condition] - row['mean']  # all - minus_x
            dataset_differences_data.append({
                'Dataset_Config': config,
                'Condition': condition,
                'Difference': diff
            })

# Convert to DataFrame
dataset_diff_df = pd.DataFrame(dataset_differences_data)

# Map dataset configuration names for display
config_name_mapping = {
    'Minus Dermie': 'Dermie',
    'Minus Fitz': 'Fitzpatrick17k', 
    'Minus India': 'India',
    'Minus PAD': 'PADUFES',
    'Minus SCIN': 'SCIN'
}

# Apply mapping to dataset names
dataset_diff_df['Display_Config'] = dataset_diff_df['Dataset_Config'].map(config_name_mapping).fillna(dataset_diff_df['Dataset_Config'])
display_configs = [config_name_mapping.get(config, config) for config in other_configs]

# Plot dataset differences  
fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(conditions))
width = 0.8 / len(other_configs)

for i, (original_config, display_config) in enumerate(zip(other_configs, display_configs)):
    config_data = dataset_diff_df[dataset_diff_df['Dataset_Config'] == original_config]
    means = [config_data[config_data['Condition'] == cond]['Difference'].values[0] if cond in config_data['Condition'].values else 0 for cond in conditions]
    ax.bar(x + i*width, means, width, label=display_config, alpha=0.9, color=custom_palette[i % len(custom_palette)])

# Add horizontal line at y=0 for reference
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

ax.set_xlabel('Skin Conditions')
ax.set_ylabel('Performance Drop from "All Datasets" (%)')
ax.set_title('Dataset Sensitivity: Performance Drop from "All Datasets"')
ax.set_xticks(x + width * (len(other_configs) - 1) / 2)
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_sensitivity_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Max-Min (Dataset) ===
dataset_disparity_max_min = dataset_stats.groupby('Dataset_Config')['mean'].agg(['max', 'min']).reset_index()
dataset_disparity_max_min['disparity'] = dataset_disparity_max_min['max'] - dataset_disparity_max_min['min']

plt.figure(figsize=(10, 6))
plt.bar(dataset_disparity_max_min['Dataset_Config'], dataset_disparity_max_min['disparity'], color='#76b5c5')
plt.xlabel('Dataset Configuration')
plt.ylabel('Disparity (Max - Min) %')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_disparity_max_min.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Std Dev (Dataset) ===
dataset_disparity_std = dataset_stats.groupby('Dataset_Config')['mean'].std().reset_index()
dataset_disparity_std.columns = ['Dataset_Config', 'std_disparity']

plt.figure(figsize=(10, 6))
plt.bar(dataset_disparity_std['Dataset_Config'], dataset_disparity_std['std_disparity'], color='#76b5c5')
plt.xlabel('Dataset Configuration')
plt.ylabel('Disparity (Std Dev) %')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_disparity_std.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All figures saved to: {output_directory}")

# === Heatmap of Architecture Performance ===
# Using stratified_df for heatmap
heatmap_data = top1_conditions[top1_conditions['Dataset_Config'] == baseline_config]
heatmap_data = heatmap_data[heatmap_data['Metric'] == 'Top-1 Sensitivity']

# Pivot table for heatmap
heatmap_pivot = heatmap_data.pivot_table(index='Model', columns='Condition', values='Value', aggfunc='mean')
heatmap_pivot = heatmap_pivot.reindex(index=models)  # Optional: keep consistent model order

plt.figure(figsize=(12, 6))
sns.heatmap(
    heatmap_pivot,
    cmap=sns.color_palette("RdYlGn", as_cmap=True),
    vmin=0,
    vmax=100,
    annot=True,
    fmt=".1f",
    cbar_kws={'label': 'Top-1 Sensitivity (%)'}
)
plt.xlabel("Skin Condition")
plt.ylabel("Model Architecture")
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("All skin condition analysis plots completed!")