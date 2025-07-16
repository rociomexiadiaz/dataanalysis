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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\OverallPlots"

os.makedirs(output_directory, exist_ok=True)

# Load data from "All Datasets" configuration
log_path = os.path.join(log_directory, 'combined_all.txt')

print("Loading data from combined_all.txt...")
df_dict = parse_combined_log(log_path)

# Extract the relevant dataframes
overall_accuracies = df_dict['OverallAccuracies']
condition_sensitivities = df_dict['ConditionSensitivities']

# Clean and rename models
overall_accuracies['Model'] = overall_accuracies['Model'].replace({
    'train_Baseline': 'Baseline',
    'train_VAE': 'VAE',
    'train_TABE': 'TABE',
    'train_FairDisCo': 'FairDisCo'
})

condition_sensitivities['Model'] = condition_sensitivities['Model'].replace({
    'train_Baseline': 'Baseline',
    'train_VAE': 'VAE',
    'train_TABE': 'TABE',
    'train_FairDisCo': 'FairDisCo'
})

# === PLOT 1: Overall Accuracy Bar Chart ===
print("Creating overall accuracy bar chart...")

# Filter for the models we want
models_to_include = ['Baseline', 'TABE', 'VAE', 'FairDisCo']
overall_filtered = overall_accuracies[overall_accuracies['Model'].isin(models_to_include)]

# Prepare data for plotting
models = overall_filtered['Model'].tolist()
top1_acc = overall_filtered['Top-1'].tolist()
top3_acc = overall_filtered['Top-3'].tolist()
top5_acc = overall_filtered['Top-5'].tolist()

# Create bar chart
x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 8))

bars1 = ax.bar(x - width, top1_acc, width, label='Top-1', alpha=0.8, color='#1f77b4')
bars2 = ax.bar(x, top3_acc, width, label='Top-3', alpha=0.8, color='#ff7f0e')
bars3 = ax.bar(x + width, top5_acc, width, label='Top-5', alpha=0.8, color='#2ca02c')

# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),  # 3 points vertical offset
                   textcoords="offset points",
                   ha='center', va='bottom',
                   fontsize=10)

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

ax.set_xlabel('Model Architecture', fontsize=12)
ax.set_ylabel('Overall Accuracy (%)', fontsize=12)
ax.set_title('Overall Accuracy Comparison (All Datasets)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)  # Give some space for labels

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'overall_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: overall_accuracy_comparison.png")

# === PLOT 2: Condition Sensitivity Heatmap ===
print("Creating condition sensitivity heatmap...")

# Filter for Top-1 Sensitivity and the models we want
top1_sensitivities = condition_sensitivities[
    (condition_sensitivities['Metric'] == 'Top-1 Sensitivity') &
    (condition_sensitivities['Model'].isin(models_to_include))
]

if not top1_sensitivities.empty:
    # Create pivot table for heatmap
    heatmap_data = top1_sensitivities.pivot_table(
        index='Condition', 
        columns='Model', 
        values='Value', 
        aggfunc='mean'
    )
    
    # Reorder columns to match the order we want
    heatmap_data = heatmap_data.reindex(columns=models_to_include)
    
    # Create heatmap
    plt.figure(figsize=(10, 12))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',  # Red for low sensitivity, Green for high sensitivity
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Top-1 Sensitivity (%)'}
    )
    
    plt.title('Top-1 Sensitivity by Condition and Model', fontsize=14, fontweight='bold')
    plt.xlabel('Model Architecture', fontsize=12)
    plt.ylabel('Skin Condition', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'condition_sensitivity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: condition_sensitivity_heatmap.png")
else:
    print("⚠️ No Top-1 sensitivity data found for heatmap")

print(f"\n✅ All plots saved to: {output_directory}")
print("Generated plots:")
print("1. overall_accuracy_comparison.png - Bar chart of Top-1,3,5 overall accuracy")
print("2. condition_sensitivity_heatmap.png - Heatmap of condition vs model sensitivity")

# Print summary statistics
print(f"\n=== Summary Statistics ===")
print(f"Models analyzed: {models}")
if not overall_filtered.empty:
    print(f"\nOverall Accuracy Results:")
    for _, row in overall_filtered.iterrows():
        print(f"  {row['Model']}: Top-1: {row['Top-1']:.1f}%, Top-3: {row['Top-3']:.1f}%, Top-5: {row['Top-5']:.1f}%")

if not top1_sensitivities.empty:
    print(f"\nConditions analyzed: {sorted(top1_sensitivities['Condition'].unique())}")
    print(f"Average sensitivity across all models and conditions: {top1_sensitivities['Value'].mean():.1f}%")