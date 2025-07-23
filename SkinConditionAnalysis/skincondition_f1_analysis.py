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

all_f1_scores = []  # Changed from condition_sensitivities

for log_file in log_files:
    log_path = os.path.join(log_directory, log_file)
    if os.path.exists(log_path):
        df_dict = parse_combined_log(log_path)
        dataset_name = dataset_mapping[log_file]
        
        # Use F1Scores instead of ConditionSensitivities
        if 'F1Scores' in df_dict and not df_dict['F1Scores'].empty:
            df_dict['F1Scores']['Dataset_Config'] = dataset_name
            all_f1_scores.append(df_dict['F1Scores'])

# Combine F1 data
if all_f1_scores:
    f1_scores = pd.concat(all_f1_scores, ignore_index=True)
else:
    print("No F1 score data found!")
    exit()

# Filter for condition-specific F1 scores (not overall or stratified)
condition_f1_scores = f1_scores[
    (f1_scores['Metric'] == 'F1 Score') & 
    (f1_scores['Condition'] != 'Overall')
].copy()

# Filter & rename models
condition_f1_scores = condition_f1_scores[~condition_f1_scores['Model'].str.contains('air')]
condition_f1_scores['Model'] = condition_f1_scores['Model'].replace({
    'train_Baseline': 'Baseline',
    'train_VAE': 'VAE',
    'train_TABE': 'TABE'
})

baseline_config = 'All Datasets'
models = condition_f1_scores['Model'].unique()
conditions = condition_f1_scores['Condition'].unique()
dataset_configs = condition_f1_scores['Dataset_Config'].unique()

# Color palette for bars
custom_palette = ['#444e86', '#955196', '#dd5182', '#ff6e54', '#ffa600']

print("=== Skin Condition Analysis - F1 Score Version ===")
print(f"Models found: {models}")
print(f"Conditions found: {conditions}")
print(f"Dataset configs found: {dataset_configs}")
print(f"Total F1 records: {len(condition_f1_scores)}")

# === Architecture Analysis ===
print("\nGenerating architecture F1 analysis...")
arch_data = condition_f1_scores[condition_f1_scores['Dataset_Config'] == baseline_config].copy()
arch_stats = arch_data.groupby(['Condition', 'Model'])['Value'].agg(['mean', 'std', 'count']).reset_index()
arch_stats['std'] = arch_stats['std'].fillna(0)
arch_stats['sem'] = arch_stats['std'] / np.sqrt(arch_stats['count'])

fig, ax = plt.subplots(figsize=(15, 8))
x = np.arange(len(conditions))
width = 0.8 / len(models)

for i, model in enumerate(models):
    model_data = arch_stats[arch_stats['Model'] == model]
    means = [model_data[model_data['Condition'] == cond]['mean'].values[0] if cond in model_data['Condition'].values else 0 for cond in conditions]
    stds = [model_data[model_data['Condition'] == cond]['std'].values[0] if cond in model_data['Condition'].values else 0 for cond in conditions]
    ax.bar(x + i*width, means, width, label=model, yerr=stds, capsize=5, alpha=0.9, color=custom_palette[i % len(custom_palette)])

ax.set_xlabel('Skin Conditions')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Architecture Comparison: F1 Score by Model and Condition\n(All Datasets Combined)')
ax.set_xticks(x + width * (len(models) - 1) / 2)
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.legend()
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_f1_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Max-Min (Architecture) ===
disparity_max_min = arch_stats.groupby('Model')['mean'].agg(['max', 'min']).reset_index()
disparity_max_min['disparity'] = disparity_max_min['max'] - disparity_max_min['min']

plt.figure(figsize=(10, 6))
plt.bar(disparity_max_min['Model'], disparity_max_min['disparity'], color='#76b5c5')
plt.xlabel('Model Architecture')
plt.ylabel('F1 Score Disparity (Max - Min) %')
plt.title('F1 Score Disparity Across Conditions by Architecture')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_f1_disparity_max_min.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Std Dev (Architecture) ===
disparity_std = arch_stats.groupby('Model')['mean'].std().reset_index()
disparity_std.columns = ['Model', 'std_disparity']

plt.figure(figsize=(10, 6))
plt.bar(disparity_std['Model'], disparity_std['std_disparity'], color='#76b5c5')
plt.xlabel('Model Architecture')
plt.ylabel('F1 Score Disparity (Std Dev) %')
plt.title('F1 Score Standard Deviation Across Conditions by Architecture')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_f1_disparity_std.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Dataset Analysis ===
print("Generating dataset F1 analysis...")
baseline_model = 'Baseline'
dataset_data = condition_f1_scores[condition_f1_scores['Model'] == baseline_model].copy()
dataset_stats = dataset_data.groupby(['Condition', 'Dataset_Config'])['Value'].agg(['mean', 'std', 'count']).reset_index()
dataset_stats['std'] = dataset_stats['std'].fillna(0)
dataset_stats['sem'] = dataset_stats['std'] / np.sqrt(dataset_stats['count'])

fig, ax = plt.subplots(figsize=(16, 8))
x = np.arange(len(conditions))
width = 0.8 / len(dataset_configs)

for i, dataset_config in enumerate(dataset_configs):
    config_data = dataset_stats[dataset_stats['Dataset_Config'] == dataset_config]
    means = [config_data[config_data['Condition'] == cond]['mean'].values[0] if cond in config_data['Condition'].values else 0 for cond in conditions]
    stds = [config_data[config_data['Condition'] == cond]['std'].values[0] if cond in config_data['Condition'].values else 0 for cond in conditions]
    ax.bar(x + i*width, means, width, label=dataset_config, yerr=stds, capsize=5, alpha=0.9, color=custom_palette[i % len(custom_palette)])

ax.set_xlabel('Skin Conditions')
ax.set_ylabel('F1 Score (%)')
ax.set_title('Dataset Comparison: F1 Score by Dataset Configuration and Condition\n(Baseline Model)')
ax.set_xticks(x + width * (len(dataset_configs) - 1) / 2)
ax.set_xticklabels(conditions, rotation=45, ha='right')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_f1_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Max-Min (Dataset) ===
dataset_disparity_max_min = dataset_stats.groupby('Dataset_Config')['mean'].agg(['max', 'min']).reset_index()
dataset_disparity_max_min['disparity'] = dataset_disparity_max_min['max'] - dataset_disparity_max_min['min']

plt.figure(figsize=(10, 6))
plt.bar(dataset_disparity_max_min['Dataset_Config'], dataset_disparity_max_min['disparity'], color='#76b5c5')
plt.xlabel('Dataset Configuration')
plt.ylabel('F1 Score Disparity (Max - Min) %')
plt.title('F1 Score Disparity Across Conditions by Dataset Configuration')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_f1_disparity_max_min.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Disparity Std Dev (Dataset) ===
dataset_disparity_std = dataset_stats.groupby('Dataset_Config')['mean'].std().reset_index()
dataset_disparity_std.columns = ['Dataset_Config', 'std_disparity']

plt.figure(figsize=(10, 6))
plt.bar(dataset_disparity_std['Dataset_Config'], dataset_disparity_std['std_disparity'], color='#76b5c5')
plt.xlabel('Dataset Configuration')
plt.ylabel('F1 Score Disparity (Std Dev) %')
plt.title('F1 Score Standard Deviation Across Conditions by Dataset Configuration')
plt.xticks(rotation=45, ha='right')
plt.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'dataset_f1_disparity_std.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ All figures saved to: {output_directory}")

# === Heatmap of Architecture Performance ===
print("Generating architecture F1 heatmap...")
heatmap_data = condition_f1_scores[condition_f1_scores['Dataset_Config'] == baseline_config]

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
    cbar_kws={'label': 'F1 Score (%)'},
    linewidths=0.5
)
plt.xlabel("Skin Condition")
plt.ylabel("Model Architecture")
plt.title("F1 Score Heatmap: Performance by Model and Condition\n(All Datasets Combined)")
plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'architecture_f1_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

# === Summary Statistics ===
print(f"\n=== F1 Score Analysis Summary ===")
print(f"Total F1 records processed: {len(condition_f1_scores)}")
print(f"Models analyzed: {list(models)}")
print(f"Conditions analyzed: {list(conditions)}")
print(f"Dataset configurations: {list(dataset_configs)}")

# Show overall F1 performance per condition
overall_f1_by_condition = condition_f1_scores[
    condition_f1_scores['Dataset_Config'] == baseline_config
].groupby('Condition')['Value'].mean().sort_values(ascending=False)

print(f"\nAverage F1 Score by Condition (All Datasets):")
for condition, f1_score in overall_f1_by_condition.items():
    print(f"  {condition}: {f1_score:.2f}%")

# Show overall F1 performance per model
overall_f1_by_model = condition_f1_scores[
    condition_f1_scores['Dataset_Config'] == baseline_config
].groupby('Model')['Value'].mean().sort_values(ascending=False)

print(f"\nAverage F1 Score by Model (All Datasets):")
for model, f1_score in overall_f1_by_model.items():
    print(f"  {model}: {f1_score:.2f}%")

print(f"\n✅ All F1 condition analysis files saved to: {output_directory}")
print("Generated F1 score files:")
print("- architecture_f1_comparison.png")
print("- architecture_f1_disparity_max_min.png")  
print("- architecture_f1_disparity_std.png")
print("- dataset_f1_comparison.png")
print("- dataset_f1_disparity_max_min.png")
print("- dataset_f1_disparity_std.png") 
print("- architecture_f1_heatmap.png")
print("\n📊 F1 Analysis Complete!")