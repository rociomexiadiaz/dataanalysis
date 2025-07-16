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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Heatmaps"
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

all_stratified_sensitivities = []

for log_file in log_files:
    log_path = os.path.join(log_directory, log_file)
    if os.path.exists(log_path):
        df_dict = parse_combined_log(log_path)
        dataset_name = dataset_mapping[log_file]
        df_dict['StratifiedSensitivities']['Dataset_Config'] = dataset_name
        all_stratified_sensitivities.append(df_dict['StratifiedSensitivities'])

# Combine data
stratified_data = pd.concat(all_stratified_sensitivities, ignore_index=True)

# Filter for Top-1 Sensitivity and rename models
top1_data = stratified_data[stratified_data['Metric'] == 'Top-1 Sensitivity'].copy()
top1_data = top1_data[~top1_data['Model'].str.contains('FairDisco', case=False, na=False)]
top1_data['Model'] = top1_data['Model'].replace({
    'train_Baseline': 'Baseline',
    'train_VAE': 'VAE',
    'train_TABE': 'TABE'
})

print("Creating difference heatmaps...")

# === ARCHITECTURE HEATMAPS (TABE and VAE vs Baseline) ===
all_datasets_data = top1_data[top1_data['Dataset_Config'] == 'All Datasets']

# Get baseline data for comparison
baseline_data = all_datasets_data[all_datasets_data['Model'] == 'Baseline']
baseline_pivot = baseline_data.pivot_table(index='Skin Tone', columns='Condition', values='Value', aggfunc='mean')

# Create heatmaps for TABE and VAE
for model in ['TABE', 'VAE']:
    model_data = all_datasets_data[all_datasets_data['Model'] == model]
    model_pivot = model_data.pivot_table(index='Skin Tone', columns='Condition', values='Value', aggfunc='mean')
    
    # Calculate difference: model - baseline
    diff_pivot = model_pivot - baseline_pivot
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        diff_pivot, 
        annot=True, 
        fmt='.1f',
        cmap='RdYlGn',  # Red for negative (worse), Green for positive (better)
        center=0,  # Center colormap at 0
        cbar_kws={'label': 'Difference from Baseline (%)'}
    )
    plt.title(f'{model} vs Baseline: Performance Difference\n(Positive = Better than Baseline)')
    plt.xlabel('Skin Condition')
    plt.ylabel('Fitzpatrick Skin Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'{model}_vs_baseline_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {model}_vs_baseline_heatmap.png")

# === DATASET HEATMAPS (All vs Minus datasets) ===
baseline_model_data = top1_data[top1_data['Model'] == 'Baseline']

# Get "All Datasets" baseline
all_data = baseline_model_data[baseline_model_data['Dataset_Config'] == 'All Datasets']
all_pivot = all_data.pivot_table(index='Skin Tone', columns='Condition', values='Value', aggfunc='mean')

# Dataset name mapping for cleaner titles
dataset_name_mapping = {
    'Minus Dermie': 'Dermie',
    'Minus Fitz': 'Fitzpatrick17k',
    'Minus India': 'India',
    'Minus PAD': 'PADUFES',
    'Minus SCIN': 'SCIN'
}

# Create heatmaps for each minus dataset
for dataset_config in ['Minus Dermie', 'Minus Fitz', 'Minus India', 'Minus PAD', 'Minus SCIN']:
    dataset_data = baseline_model_data[baseline_model_data['Dataset_Config'] == dataset_config]
    
    if not dataset_data.empty:
        dataset_pivot = dataset_data.pivot_table(index='Skin Tone', columns='Condition', values='Value', aggfunc='mean')
        
        # Calculate difference: all - minus_dataset (positive = removing dataset hurts performance)
        diff_pivot = all_pivot - dataset_pivot
        
        # Get clean dataset name for title
        clean_name = dataset_name_mapping.get(dataset_config, dataset_config)
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            diff_pivot, 
            annot=True, 
            fmt='.1f',
            cmap='RdYlGn',  # Red for negative (dataset not important), Green for positive (dataset important)
            center=0,  # Center colormap at 0
            cbar_kws={'label': 'Performance Drop (%)'}
        )
        plt.title(f'Impact of Removing {clean_name} Dataset\n(Positive = Dataset is Important)')
        plt.xlabel('Skin Condition')
        plt.ylabel('Fitzpatrick Skin Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'impact_removing_{clean_name.lower()}_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: impact_removing_{clean_name.lower()}_heatmap.png")

print(f"\n✅ All 7 heatmaps saved to: {output_directory}")
print("Generated heatmaps:")
print("1. TABE_vs_baseline_heatmap.png")
print("2. VAE_vs_baseline_heatmap.png") 
print("3. impact_removing_dermie_heatmap.png")
print("4. impact_removing_fitzpatrick17k_heatmap.png")
print("5. impact_removing_india_heatmap.png")
print("6. impact_removing_padufes_heatmap.png")
print("7. impact_removing_scin_heatmap.png")