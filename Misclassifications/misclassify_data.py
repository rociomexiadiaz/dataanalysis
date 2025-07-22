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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Misclassifications"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Define high-risk misclassification pairs
high_risk_pairs = [
    ('melanoma â', 'nevus'),
    ('melanoma â', 'seborrheic_keratosis'),
    ('melanoma â', 'melanocytic_nevus'),
    ('melanoma â', 'seborrheic'),
    ('melanoma â', 'acne'),
    ('melanoma â', 'eczema'),
    ('melanoma â', 'psoriasis'),
    ('melanoma â', 'urticaria'),
    ('basal_cell_carcinoma â', 'nevus'),
    ('basal_cell_carcinoma â', 'seborrheic_keratosis'),
    ('basal_cell_carcinoma â', 'melanocytic_nevus'),
    ('bcc â', 'nevus'),
    ('bcc â', 'seborrheic'),
    ('bcc â', 'melanocytic'),
    ('bcc â', 'acne'),
    ('bcc â', 'eczema'),
    ('bcc â', 'psoriasis'),
    ('bcc â', 'urticaria'),
    ('squamous_cell_carcinoma â', 'nevus'),
    ('squamous_cell_carcinoma â', 'seborrheic_keratosis'),
    ('squamous_cell_carcinoma â', 'melanocytic_nevus'),
    ('scc â', 'nevus'),
    ('scc â', 'seborrheic'),
    ('scc â', 'melanocytic'),
    ('scc â', 'acne'),
    ('scc â', 'eczema'),
    ('scc â', 'psoriasis'),
    ('scc â', 'urticaria'),
]

def normalize_condition_name(name):
    """Normalize condition names for matching"""
    return name.lower().replace('_', '').replace(' ', '').replace('-', '')

def is_high_risk_pair(true_class, pred_class, high_risk_pairs):
    """Check if a misclassification pair is high-risk"""
    true_norm = normalize_condition_name(true_class)
    pred_norm = normalize_condition_name(pred_class)
    
    for risk_true, risk_pred in high_risk_pairs:
        risk_true_norm = normalize_condition_name(risk_true)
        risk_pred_norm = normalize_condition_name(risk_pred)
        
        if (true_norm == risk_true_norm and pred_norm == risk_pred_norm):
            return True
    return False

def get_dataset_removed_name(datasets_str):
    """Extract which dataset was removed from the datasets string"""
    # Clean up the datasets string
    datasets_str = datasets_str.strip().lower()
    
    # Map actual dataset combinations to what was removed
    if 'dermie' in datasets_str and 'padufes' in datasets_str and 'scin' in datasets_str and 'fitzpatrick17k' in datasets_str and 'india' in datasets_str:
        return 'All'  # All datasets included
    elif 'padufes' in datasets_str and 'scin' in datasets_str and 'fitzpatrick17k' in datasets_str and 'india' in datasets_str and 'dermie' not in datasets_str:
        return 'Minus Dermie'  # Dermie removed
    elif 'dermie' in datasets_str and 'padufes' in datasets_str and 'scin' in datasets_str and 'india' in datasets_str and 'fitzpatrick17k' not in datasets_str:
        return 'Minus Fitzpatrick17k'  # Fitzpatrick17k removed
    elif 'dermie' in datasets_str and 'padufes' in datasets_str and 'scin' in datasets_str and 'fitzpatrick17k' in datasets_str and 'india' not in datasets_str:
        return 'Minus India'  # India removed
    elif 'dermie' in datasets_str and 'scin' in datasets_str and 'fitzpatrick17k' in datasets_str and 'india' in datasets_str and 'padufes' not in datasets_str:
        return 'Minus PADUFES'  # PADUFES removed
    elif 'dermie' in datasets_str and 'padufes' in datasets_str and 'fitzpatrick17k' in datasets_str and 'india' in datasets_str and 'scin' not in datasets_str:
        return 'Minus SCIN'  # SCIN removed
    else:
        # Return the original string if no match found
        return datasets_str

# Define test condition counts for each dataset configuration (from your document)
test_condition_counts = {
    'All': 1173,  # Total test condition samples when all datasets included
    'Minus Dermie': 1086,  # Total test condition samples when dermie removed
    'Minus PADUFES': 1033,  # Total test condition samples when padufes removed
    'Minus SCIN': 1055,  # Total test condition samples when scin removed
    'Minus Fitzpatrick17k': 835,  # Total test condition samples when fitzpatrick17k removed
    'Minus India': 683  # Total test condition samples when india removed
}

# Define test condition counts stratified by skin tone (based on the provided test set image)
# These are the actual test set counts for each skin tone per dataset
test_condition_counts_by_skin_tone = {
    'All': {
        # Combined all datasets - actual test set counts
        1.0: 48 + 18 + 74 + 2 + 11,    # FST I: 153 samples
        2.0: 22 + 85 + 119 + 0 + 34,   # FST II: 260 samples  
        3.0: 8 + 31 + 57 + 0 + 30,     # FST III: 126 samples
        4.0: 2 + 5 + 52 + 407 + 23,    # FST IV: 489 samples
        5.0: 6 + 1 + 27 + 0 + 13,      # FST V: 47 samples
        6.0: 1 + 0 + 9 + 81 + 7        # FST VI: 98 samples
    },
    'Minus Dermie': {
        # Without Dermie dataset
        1.0: 18 + 74 + 2 + 11,         # 105 samples
        2.0: 85 + 119 + 0 + 34,        # 238 samples
        3.0: 31 + 57 + 0 + 30,         # 118 samples
        4.0: 5 + 52 + 407 + 23,        # 487 samples
        5.0: 1 + 27 + 0 + 13,          # 41 samples
        6.0: 0 + 9 + 81 + 7            # 97 samples
    },
    'Minus PADUFES': {
        # Without PADUFES dataset
        1.0: 48 + 74 + 2 + 11,         # 135 samples
        2.0: 22 + 119 + 0 + 34,        # 175 samples
        3.0: 8 + 57 + 0 + 30,          # 95 samples
        4.0: 2 + 52 + 407 + 23,        # 484 samples
        5.0: 6 + 27 + 0 + 13,          # 46 samples
        6.0: 1 + 9 + 81 + 7            # 98 samples
    },
    'Minus SCIN': {
        # Without SCIN dataset
        1.0: 48 + 18 + 74 + 2,         # 142 samples
        2.0: 22 + 85 + 119 + 0,        # 226 samples
        3.0: 8 + 31 + 57 + 0,          # 96 samples
        4.0: 2 + 5 + 52 + 407,         # 466 samples
        5.0: 6 + 1 + 27 + 0,           # 34 samples
        6.0: 1 + 0 + 9 + 81            # 91 samples
    },
    'Minus Fitzpatrick17k': {
        # Without Fitzpatrick17k dataset
        1.0: 48 + 18 + 2 + 11,         # 79 samples
        2.0: 22 + 85 + 0 + 34,         # 141 samples
        3.0: 8 + 31 + 0 + 30,          # 69 samples
        4.0: 2 + 5 + 407 + 23,         # 437 samples
        5.0: 6 + 1 + 0 + 13,           # 20 samples
        6.0: 1 + 0 + 81 + 7            # 89 samples
    },
    'Minus India': {
        # Without India dataset
        1.0: 48 + 18 + 74 + 11,        # 151 samples
        2.0: 22 + 85 + 119 + 34,       # 260 samples
        3.0: 8 + 31 + 57 + 30,         # 126 samples
        4.0: 2 + 5 + 52 + 23,          # 82 samples
        5.0: 6 + 1 + 27 + 13,          # 47 samples
        6.0: 1 + 0 + 9 + 7             # 17 samples
    }
}

# Load data from combined_baseline_datsets.txt
log_path = os.path.join(log_directory, 'combined_baseline_datasets.txt')

if not os.path.exists(log_path):
    print(f"ERROR: File {log_path} not found!")
    print("Please ensure the combined_baseline_datasets.txt file exists in the specified directory.")
    exit()

print("Loading misclassification data from combined_baseline_datasets.txt...")
df_dict = parse_combined_log(log_path)

# Extract misclassification dataframes
misclassifications_by_tone = df_dict['MisclassificationDetails']
misclassified_counts = df_dict['MisclassifiedCounts']

print(f"Found detailed misclassification data: {len(misclassifications_by_tone)} records")
print(f"Found total misclassification counts: {len(misclassified_counts)} models")

if len(misclassifications_by_tone) == 0:
    print("ERROR: No detailed misclassification data found!")
    print("Please ensure your dataframe.py has been updated with the fixed parsing code.")
    exit()

# Identify baseline models (assuming they contain 'Baseline' in the name)
baseline_models = misclassifications_by_tone['Model'].unique()
print(f"Models found: {list(baseline_models)}")

# Filter for baseline model only (adjust model name if needed)
BASELINE_MODEL = "train_Baseline"
baseline_data = misclassifications_by_tone[
    misclassifications_by_tone['Model'] == BASELINE_MODEL
].copy()

if len(baseline_data) == 0:
    print(f"ERROR: No data found for baseline model '{BASELINE_MODEL}'")
    print(f"Available models: {list(baseline_models)}")
    print("Please adjust the BASELINE_MODEL variable to match your actual baseline model name.")
    exit()

print(f"Found {len(baseline_data)} baseline misclassifications across different dataset combinations")

# Add high-risk indicator to the baseline data
baseline_data['High_Risk'] = baseline_data.apply(
    lambda row: is_high_risk_pair(row['True Class'], row['Predicted Class'], high_risk_pairs), 
    axis=1
)

# Create misclassification pair labels
baseline_data['Misclassification_Pair'] = (
    baseline_data['True Class'] + ' → ' + baseline_data['Predicted Class']
)

# Add dataset removed information
baseline_data['Dataset_Removed'] = baseline_data['Datasets'].apply(get_dataset_removed_name)

print(f"Dataset combinations found: {list(baseline_data['Dataset_Removed'].unique())}")
print(f"Identified {baseline_data['High_Risk'].sum()} high-risk misclassifications out of {len(baseline_data)} total")

# ========== VISUALIZATIONS BY DATASET REMOVED ==========

print("\n=== Creating Visualizations by Dataset Removed ===")

# 1. Total Misclassifications by Dataset Configuration (as percentage of test condition data)
# Filter for 'All' skin tone only to avoid double counting
dataset_totals = baseline_data[baseline_data['Skin Tone'] == 'All'].groupby('Dataset_Removed')['Count'].sum().reset_index()

# Debug: Print what we're actually getting
print(f"\nDEBUG - Total misclassifications by dataset:")
for _, row in dataset_totals.iterrows():
    print(f"  {row['Dataset_Removed']}: {row['Count']} total misclassifications")

# Calculate percentages based on test condition counts
dataset_totals['Test_Condition_Count'] = dataset_totals['Dataset_Removed'].map(test_condition_counts)
dataset_totals['Percentage'] = (dataset_totals['Count'] / dataset_totals['Test_Condition_Count'] * 100).round(2)

# Debug: Print the percentage calculation
print(f"\nDEBUG - Percentage calculations:")
for _, row in dataset_totals.iterrows():
    print(f"  {row['Dataset_Removed']}: {row['Count']} / {row['Test_Condition_Count']} = {row['Percentage']:.2f}%")

dataset_totals = dataset_totals.sort_values('Percentage', ascending=False)

plt.figure(figsize=(12, 8))
bars = plt.bar(dataset_totals['Dataset_Removed'], dataset_totals['Percentage'], 
               color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.8)
plt.xlabel('Dataset Configuration', fontsize=12)
plt.ylabel('Misclassification Rate (% of Test Conditions)', fontsize=12)
plt.title('Baseline Misclassification Rate by Dataset Configuration', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for i, (v, count, test_size) in enumerate(zip(dataset_totals['Percentage'], dataset_totals['Count'], dataset_totals['Test_Condition_Count'])):
    plt.text(i, v + max(dataset_totals['Percentage']) * 0.01, f'{v:.1f}%\n({count}/{test_size})', 
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'baseline_misclassifications_by_dataset_removed.png'), 
           dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: baseline_misclassifications_by_dataset_removed.png")

# 2. High-Risk vs Regular Misclassifications by Dataset Configuration (as percentage of test condition data)
# Filter for 'All' skin tone only to avoid double counting
all_skin_tone_data = baseline_data[baseline_data['Skin Tone'] == 'All']
high_risk_summary = all_skin_tone_data.groupby(['Dataset_Removed', 'High_Risk'])['Count'].sum().reset_index()
high_risk_pivot = high_risk_summary.pivot_table(values='Count', index='Dataset_Removed', 
                                               columns='High_Risk', fill_value=0)

# Convert to percentages based on test condition counts
high_risk_pivot_pct = high_risk_pivot.copy()
for dataset in high_risk_pivot_pct.index:
    test_size = test_condition_counts[dataset]
    high_risk_pivot_pct.loc[dataset] = (high_risk_pivot.loc[dataset] / test_size * 100).round(2)

plt.figure(figsize=(12, 8))
x = np.arange(len(high_risk_pivot_pct.index))
width = 0.35

bars_regular = None
bars_high_risk = None

if False in high_risk_pivot_pct.columns:
    bars_regular = plt.bar(x - width/2, high_risk_pivot_pct[False], width, 
                          label='Regular Misclassifications', alpha=0.8, color='skyblue')
if True in high_risk_pivot_pct.columns:
    bars_high_risk = plt.bar(x + width/2, high_risk_pivot_pct[True], width, 
                           label='High-Risk Misclassifications', color='red', alpha=0.8)

plt.xlabel('Dataset Configuration', fontsize=12)
plt.ylabel('Misclassification Rate (% of Test Conditions)', fontsize=12)
plt.title('High-Risk vs Regular Misclassification Rates by Dataset Configuration', fontsize=14, fontweight='bold')
plt.xticks(x, high_risk_pivot_pct.index, rotation=45, ha='right')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
if bars_regular:
    for i, (v, count) in enumerate(zip(high_risk_pivot_pct[False], high_risk_pivot[False])):
        if v > 0:
            plt.text(i - width/2, v + max(high_risk_pivot_pct.values.flatten()) * 0.01, 
                     f'{v:.1f}%\n({count})', ha='center', va='bottom', fontsize=9)

if bars_high_risk:
    for i, (v, count) in enumerate(zip(high_risk_pivot_pct[True], high_risk_pivot[True])):
        if v > 0:
            plt.text(i + width/2, v + max(high_risk_pivot_pct.values.flatten()) * 0.01, 
                     f'{v:.1f}%\n({count})', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'high_risk_vs_regular_by_dataset_removed.png'), 
           dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: high_risk_vs_regular_by_dataset_removed.png")

# 3. All Misclassifications Heatmap by Dataset Configuration and Skin Tone (STRATIFIED BY SKIN TONE)
# Use all skin tone data (not just 'All')
skin_tone_data = baseline_data[baseline_data['Skin Tone'] != 'All']

if len(skin_tone_data) > 0:
    # Create pivot table for all misclassifications by skin tone
    all_misclass_pivot = skin_tone_data.groupby(['Dataset_Removed', 'Skin Tone'])['Count'].sum().reset_index()
    all_misclass_pivot_table = all_misclass_pivot.pivot_table(values='Count', index='Skin Tone', 
                                                             columns='Dataset_Removed', fill_value=0)
    
    # Convert to percentages based on test condition counts BY SKIN TONE
    all_misclass_pivot_pct = all_misclass_pivot_table.copy()
    for dataset in all_misclass_pivot_pct.columns:
        if dataset in test_condition_counts_by_skin_tone:
            for skin_tone in all_misclass_pivot_pct.index:
                if skin_tone in test_condition_counts_by_skin_tone[dataset]:
                    test_size = test_condition_counts_by_skin_tone[dataset][skin_tone]
                    if test_size > 0:
                        all_misclass_pivot_pct.loc[skin_tone, dataset] = (all_misclass_pivot_table.loc[skin_tone, dataset] / test_size * 100)
                    else:
                        all_misclass_pivot_pct.loc[skin_tone, dataset] = 0.0
                else:
                    all_misclass_pivot_pct.loc[skin_tone, dataset] = 0.0
    
    # Round to 2 decimal places
    all_misclass_pivot_pct = all_misclass_pivot_pct.round(2)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(all_misclass_pivot_pct, 
               annot=True, 
               fmt='.1f', 
               cmap='YlOrRd',
               cbar_kws={'label': 'Misclassification Rate (% of Test Conditions by Skin Tone)'},
               linewidths=0.5)
    
    plt.title('All Misclassifications by Skin Tone and Dataset Configuration\n(Percentages Stratified by Skin Tone)', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset Configuration', fontsize=12)
    plt.ylabel('Skin Tone (Fitzpatrick Scale)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'all_misclassifications_heatmap_by_dataset.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: all_misclassifications_heatmap_by_dataset.png")
else:
    print("No skin tone data found for all misclassifications heatmap")

# 4. High-Risk Misclassifications Heatmap by Dataset Configuration and Skin Tone (STRATIFIED BY SKIN TONE)
# Filter for high-risk misclassifications only (excluding 'All' skin tone)
high_risk_skin_tone_data = baseline_data[
    (baseline_data['High_Risk'] == True) & 
    (baseline_data['Skin Tone'] != 'All')
]

if len(high_risk_skin_tone_data) > 0:
    # Create pivot table for high-risk misclassifications by skin tone
    high_risk_pivot = high_risk_skin_tone_data.groupby(['Dataset_Removed', 'Skin Tone'])['Count'].sum().reset_index()
    high_risk_pivot_table = high_risk_pivot.pivot_table(values='Count', index='Skin Tone', 
                                                       columns='Dataset_Removed', fill_value=0)
    
    # Convert to percentages based on test condition counts BY SKIN TONE
    high_risk_pivot_pct = high_risk_pivot_table.copy()
    for dataset in high_risk_pivot_pct.columns:
        if dataset in test_condition_counts_by_skin_tone:
            for skin_tone in high_risk_pivot_pct.index:
                if skin_tone in test_condition_counts_by_skin_tone[dataset]:
                    test_size = test_condition_counts_by_skin_tone[dataset][skin_tone]
                    if test_size > 0:
                        high_risk_pivot_pct.loc[skin_tone, dataset] = (high_risk_pivot_table.loc[skin_tone, dataset] / test_size * 100)
                    else:
                        high_risk_pivot_pct.loc[skin_tone, dataset] = 0.0
                else:
                    high_risk_pivot_pct.loc[skin_tone, dataset] = 0.0
    
    # Round to 2 decimal places
    high_risk_pivot_pct = high_risk_pivot_pct.round(2)
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(high_risk_pivot_pct, 
               annot=True, 
               fmt='.1f', 
               cmap='Reds',
               cbar_kws={'label': 'High-Risk Misclassification Rate (% of Test Conditions by Skin Tone)'},
               linewidths=0.5)
    
    plt.title('High-Risk Misclassifications by Skin Tone and Dataset Configuration\n(Percentages Stratified by Skin Tone)', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset Configuration', fontsize=12)
    plt.ylabel('Skin Tone (Fitzpatrick Scale)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'high_risk_misclassifications_heatmap_by_dataset.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: high_risk_misclassifications_heatmap_by_dataset.png")
else:
    print("No high-risk skin tone data found for heatmap")

# ========== SUMMARY STATISTICS ==========

# Calculate metrics for summary (using 'All' skin tone data only)
all_skin_tone_data = baseline_data[baseline_data['Skin Tone'] == 'All']
metrics_df = all_skin_tone_data.groupby('Dataset_Removed').agg({
    'Count': 'sum',
    'High_Risk': ['sum', 'count']
}).round(2)

# Flatten column names
metrics_df.columns = ['Total_Count', 'High_Risk_Count', 'Total_Records']
metrics_df = metrics_df.reset_index()

# Add test condition counts and calculate percentages
metrics_df['Test_Condition_Count'] = metrics_df['Dataset_Removed'].map(test_condition_counts)
metrics_df['High_Risk_Percentage'] = (metrics_df['High_Risk_Count'] / metrics_df['Test_Condition_Count'] * 100).round(2)

print(f"\n=== Dataset Configuration Impact Analysis Summary ===")
print(f"Baseline Model: {BASELINE_MODEL}")
print(f"Total dataset configurations analyzed: {len(baseline_data['Dataset_Removed'].unique())}")

print(f"\nMisclassification Rates by Dataset Configuration (% of test conditions):")
for _, row in dataset_totals.iterrows():
    print(f"  {row['Dataset_Removed']}: {row['Count']} misclassifications ({row['Percentage']:.2f}% of {row['Test_Condition_Count']} test conditions)")

print(f"\nHigh-Risk Misclassification Rates by Dataset Configuration:")
for _, row in metrics_df.iterrows():
    print(f"  {row['Dataset_Removed']}: {row['High_Risk_Count']} high-risk ({row['High_Risk_Percentage']:.2f}% of {row['Test_Condition_Count']} test conditions)")

# Find best and worst performing dataset configurations
best_dataset = metrics_df.loc[metrics_df['High_Risk_Percentage'].idxmin()]
worst_dataset = metrics_df.loc[metrics_df['High_Risk_Percentage'].idxmax()]

print(f"\nDataset Safety Ranking (by high-risk misclassification rate):")
print(f"  Safest: {best_dataset['Dataset_Removed']} ({best_dataset['High_Risk_Percentage']:.2f}% high-risk rate)")
print(f"  Riskiest: {worst_dataset['Dataset_Removed']} ({worst_dataset['High_Risk_Percentage']:.2f}% high-risk rate)")

# Most common misclassifications across all datasets (using 'All' skin tone only)
print(f"\nMost Common Misclassification Pairs (across all dataset configurations):")
all_pairs = all_skin_tone_data.groupby('Misclassification_Pair')['Count'].sum().sort_values(ascending=False).head(10)
for pair, count in all_pairs.items():
    print(f"  {pair}: {count} cases")

# Most common high-risk misclassifications
high_risk_pairs_data = all_skin_tone_data[all_skin_tone_data['High_Risk'] == True]

if len(high_risk_pairs_data) > 0:
    print(f"\nMost Common High-Risk Misclassification Pairs:")
    top_high_risk = high_risk_pairs_data.groupby('Misclassification_Pair')['Count'].sum().sort_values(ascending=False).head(10)
    for pair, count in top_high_risk.items():
        print(f"  {pair}: {count} cases")

# Skin tone bias analysis - now using stratified percentages
if len(skin_tone_data) > 0:
    print(f"\nSkin Tone Misclassification Rates (Stratified by Skin Tone):")
    for dataset in test_condition_counts_by_skin_tone.keys():
        if dataset in skin_tone_data['Dataset_Removed'].values:
            print(f"\n  {dataset}:")
            dataset_skin_data = skin_tone_data[skin_tone_data['Dataset_Removed'] == dataset]
            skin_tone_totals = dataset_skin_data.groupby('Skin_Tone')['Count'].sum()
            
            for skin_tone in sorted(skin_tone_totals.index):
                if skin_tone in test_condition_counts_by_skin_tone[dataset]:
                    misclass_count = skin_tone_totals[skin_tone]
                    test_size = test_condition_counts_by_skin_tone[dataset][skin_tone]
                    if test_size > 0:
                        percentage = (misclass_count / test_size) * 100
                        print(f"    Skin Tone {int(skin_tone)}: {misclass_count} misclassifications / {test_size} test samples = {percentage:.1f}%")
                    else:
                        print(f"    Skin Tone {int(skin_tone)}: {misclass_count} misclassifications / 0 test samples = N/A%")

print(f"\nAll misclassification visualizations saved to: {output_directory}")
print(f"Total visualization files created: {len([f for f in os.listdir(output_directory) if f.endswith('.png') and 'dataset_removed' in f])}")
print(f"\nNote: Heatmaps now display misclassification rates as percentages stratified by skin tone for each dataset configuration.")
print(f"Formula: (misclassifications for skin tone X in dataset Y / test samples for skin tone X in dataset Y) × 100")