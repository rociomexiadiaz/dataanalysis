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

# Load data from the "All Datasets" configuration
log_path = os.path.join(log_directory, 'combined_all.txt')

print("Loading misclassification data from combined_all.txt...")
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

print(f"Models with misclassification data: {misclassifications_by_tone['Model'].unique()}")
print(f"Sample misclassification data:")
print(misclassifications_by_tone.head())

# Define high-risk misclassification pairs (customize based on your conditions)
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

# Add high-risk indicator to the data
misclassifications_by_tone['High_Risk'] = misclassifications_by_tone.apply(
    lambda row: is_high_risk_pair(row['True Class'], row['Predicted Class'], high_risk_pairs), 
    axis=1
)

# Create misclassification pair labels
misclassifications_by_tone['Misclassification_Pair'] = (
    misclassifications_by_tone['True Class'] + ' → ' + misclassifications_by_tone['Predicted Class']
)

print(f"Identified {misclassifications_by_tone['High_Risk'].sum()} high-risk misclassifications out of {len(misclassifications_by_tone)} total")

# 1. Overall Misclassification Summary by Model
print("\n=== Creating Visualizations ===")

# Total misclassifications by model (from detailed data)
total_misclass = misclassifications_by_tone.groupby('Model')['Count'].sum().reset_index()
total_misclass.columns = ['Model', 'Total_Misclassifications']

plt.figure(figsize=(12, 6))
bars = plt.bar(total_misclass['Model'], total_misclass['Total_Misclassifications'])
plt.xlabel('Model')
plt.ylabel('Total Detailed Misclassifications')
plt.title('Total Detailed Misclassifications by Model')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, v in enumerate(total_misclass['Total_Misclassifications']):
    plt.text(i, v + max(total_misclass['Total_Misclassifications']) * 0.01, str(v), 
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'total_detailed_misclassifications_by_model.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: total_detailed_misclassifications_by_model.png")

# 2. High-Risk vs Regular Misclassifications
high_risk_summary = misclassifications_by_tone.groupby(['Model', 'High_Risk'])['Count'].sum().reset_index()
high_risk_pivot = high_risk_summary.pivot_table(values='Count', index='Model', columns='High_Risk', fill_value=0)

plt.figure(figsize=(12, 6))
x = np.arange(len(high_risk_pivot.index))
width = 0.35

bars_regular = None
bars_high_risk = None

if False in high_risk_pivot.columns:
    bars_regular = plt.bar(x - width/2, high_risk_pivot[False], width, label='Regular Misclassifications', alpha=0.8)
if True in high_risk_pivot.columns:
    bars_high_risk = plt.bar(x + width/2, high_risk_pivot[True], width, label='High-Risk Misclassifications', 
             color='red', alpha=0.8)

plt.xlabel('Model')
plt.ylabel('Number of Misclassifications')
plt.title('High-Risk vs Regular Misclassifications by Model')
plt.xticks(x, high_risk_pivot.index, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)

# Add value labels on bars
if bars_regular:
    for i, v in enumerate(high_risk_pivot[False]):
        if v > 0:
            plt.text(i - width/2, v + max(high_risk_pivot.values.flatten()) * 0.01, str(v), 
                     ha='center', va='bottom')

if bars_high_risk:
    for i, v in enumerate(high_risk_pivot[True]):
        if v > 0:
            plt.text(i + width/2, v + max(high_risk_pivot.values.flatten()) * 0.01, str(v), 
                     ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_directory, 'high_risk_vs_regular_misclassifications.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: high_risk_vs_regular_misclassifications.png")

# 3. Top Misclassification Pairs by Model
models = misclassifications_by_tone['Model'].unique()
n_top_pairs = 15  # Show top 15 misclassification pairs

for model in models:
    model_data = misclassifications_by_tone[
        (misclassifications_by_tone['Model'] == model) & 
        (misclassifications_by_tone['Skin Tone'] == 'All')
    ]
    
    if len(model_data) > 0:
        # Get top misclassification pairs
        top_pairs = model_data.nlargest(n_top_pairs, 'Count')
        
        plt.figure(figsize=(14, 10))
        colors = ['red' if risk else 'steelblue' for risk in top_pairs['High_Risk']]
        
        y_pos = range(len(top_pairs))
        bars = plt.barh(y_pos, top_pairs['Count'], color=colors, alpha=0.8)
        plt.yticks(y_pos, top_pairs['Misclassification_Pair'])
        plt.xlabel('Number of Misclassifications')
        plt.title(f'Top {min(n_top_pairs, len(top_pairs))} Misclassification Pairs - {model}')
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, v in enumerate(top_pairs['Count']):
            plt.text(v + max(top_pairs['Count']) * 0.01, i, str(v), 
                     va='center', ha='left')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.8, label='High-Risk'),
                          Patch(facecolor='steelblue', alpha=0.8, label='Regular')]
        plt.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        safe_model_name = model.replace('/', '_').replace('\\', '_').replace(':', '_')
        plt.savefig(os.path.join(output_directory, f'top_misclassifications_{safe_model_name}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: top_misclassifications_{safe_model_name}.png")

# 4. Misclassification Heatmap by Skin Tone for Each Model
skin_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] != 'All']

if len(skin_tone_data) > 0:
    for model in models:
        model_data = skin_tone_data[skin_tone_data['Model'] == model]
        
        if len(model_data) > 0:
            # Create heatmap data: misclassification pairs vs skin tones
            heatmap_data = model_data.pivot_table(
                values='Count', 
                index='Misclassification_Pair', 
                columns='Skin Tone', 
                fill_value=0
            )
            
            # Only show top misclassification pairs to avoid overcrowding
            top_pairs_overall = misclassifications_by_tone[
                (misclassifications_by_tone['Model'] == model) & 
                (misclassifications_by_tone['Skin Tone'] == 'All')
            ].nlargest(15, 'Count')['Misclassification_Pair'].tolist()
            
            # Filter heatmap data to show only top pairs
            heatmap_filtered = heatmap_data.loc[
                heatmap_data.index.intersection(top_pairs_overall)
            ]
            
            if not heatmap_filtered.empty:
                plt.figure(figsize=(12, 10))
                
                # Create the heatmap
                sns.heatmap(heatmap_filtered, 
                           annot=True, 
                           fmt='.0f', 
                           cmap='YlOrRd',
                           cbar_kws={'label': 'Number of Misclassifications'})
                
                plt.title(f'Misclassifications by Skin Tone - {model}')
                plt.xlabel('Skin Tone (Fitzpatrick Scale)')
                plt.ylabel('Misclassification Pairs')
                
                # Color the y-axis labels based on risk
                ax = plt.gca()
                for i, label in enumerate(ax.get_yticklabels()):
                    pair_text = label.get_text()
                    if pair_text in heatmap_filtered.index:
                        true_class, pred_class = pair_text.split(' → ')
                        if is_high_risk_pair(true_class, pred_class, high_risk_pairs):
                            label.set_color('red')
                            label.set_weight('bold')
                
                plt.tight_layout()
                safe_model_name = model.replace('/', '_').replace('\\', '_').replace(':', '_')
                plt.savefig(os.path.join(output_directory, f'misclassification_heatmap_{safe_model_name}.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Saved: misclassification_heatmap_{safe_model_name}.png")

# 5. High-Risk Misclassification Analysis by Skin Tone
high_risk_data = skin_tone_data[skin_tone_data['High_Risk'] == True]

if len(high_risk_data) > 0:
    # Sum high-risk misclassifications by model and skin tone
    high_risk_summary = high_risk_data.groupby(['Model', 'Skin Tone'])['Count'].sum().reset_index()
    
    # Create pivot table
    high_risk_pivot = high_risk_summary.pivot_table(
        values='Count', 
        index='Model', 
        columns='Skin Tone', 
        fill_value=0
    )
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(high_risk_pivot, 
               annot=True, 
               fmt='.0f', 
               cmap='Reds',
               cbar_kws={'label': 'High-Risk Misclassifications'})
    
    plt.title('High-Risk Misclassifications by Model and Skin Tone')
    plt.xlabel('Skin Tone (Fitzpatrick Scale)')
    plt.ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'high_risk_misclassifications_by_skin_tone.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: high_risk_misclassifications_by_skin_tone.png")

# 6. Total High-Risk Misclassifications by Model
if misclassifications_by_tone['High_Risk'].sum() > 0:
    overall_high_risk = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] == 'All')
    ]
    
    model_high_risk = overall_high_risk.groupby('Model')['Count'].sum().reset_index()
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_high_risk['Model'], model_high_risk['Count'], color='red', alpha=0.7)
    plt.xlabel('Model')
    plt.ylabel('Total High-Risk Misclassifications')
    plt.title('Total High-Risk Misclassifications by Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(model_high_risk['Count']):
        plt.text(i, v + max(model_high_risk['Count']) * 0.01, str(v), 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'total_high_risk_by_model.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: total_high_risk_by_model.png")

# 7. Comparison with Total Misclassified Samples
if len(misclassified_counts) > 0:
    plt.figure(figsize=(12, 6))
    
    # Merge data for comparison
    comparison_data = misclassified_counts.merge(total_misclass, on='Model', how='left')
    comparison_data['Total_Misclassifications'] = comparison_data['Total_Misclassifications'].fillna(0)
    
    x = np.arange(len(comparison_data))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, comparison_data['Misclassified Samples'], width, 
                    label='Total Misclassified (Log Summary)', alpha=0.8)
    bars2 = plt.bar(x + width/2, comparison_data['Total_Misclassifications'], width, 
                    label='Detailed Misclassifications (Parsed)', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Number of Misclassifications')
    plt.title('Comparison: Total vs Detailed Misclassifications')
    plt.xticks(x, comparison_data['Model'], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(comparison_data['Misclassified Samples']):
        plt.text(i - width/2, v + max(comparison_data['Misclassified Samples']) * 0.01, str(v), 
                 ha='center', va='bottom')
    
    for i, v in enumerate(comparison_data['Total_Misclassifications']):
        plt.text(i + width/2, v + max(comparison_data['Total_Misclassifications']) * 0.01, str(int(v)), 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'total_vs_detailed_misclassifications.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Saved: total_vs_detailed_misclassifications.png")

# Summary Statistics and Reports
print(f"\n=== Summary Statistics ===")

print("Detailed Misclassification Counts by Model:")
for _, row in total_misclass.iterrows():
    print(f"  {row['Model']}: {row['Total_Misclassifications']} detailed misclassifications")


if misclassifications_by_tone['High_Risk'].sum() > 0:
    print(f"\nHigh-Risk Misclassification Analysis:")
    high_risk_by_model = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] == 'All')
    ].groupby('Model')['Count'].sum().reset_index()
    
    for _, row in high_risk_by_model.iterrows():
        total_for_model = total_misclass[total_misclass['Model'] == row['Model']]['Total_Misclassifications'].iloc[0]
        percentage = (row['Count'] / total_for_model) * 100 if total_for_model > 0 else 0
        print(f"  {row['Model']}: {row['Count']} high-risk ({percentage:.1f}% of detailed)")
    
    # Most common high-risk misclassifications
    print(f"\nMost Common High-Risk Misclassification Pairs:")
    top_high_risk = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] == 'All')
    ].groupby('Misclassification_Pair')['Count'].sum().sort_values(ascending=False).head(10)
    
    for pair, count in top_high_risk.items():
        print(f"  {pair}: {count} cases")

    # Model safety ranking
    print(f"\nModel Safety Ranking (by high-risk misclassifications, lowest = safest):")
    if len(high_risk_by_model) > 0:
        safety_ranking = high_risk_by_model.sort_values('Count')
        for i, (_, row) in enumerate(safety_ranking.iterrows()):
            print(f"  {i+1}. {row['Model']}: {row['Count']} high-risk misclassifications")

# Skin tone bias in high-risk misclassifications
if len(high_risk_data) > 0:
    print(f"\nSkin Tone Bias in High-Risk Misclassifications:")
    skin_tone_bias = high_risk_data.groupby('Skin Tone')['Count'].sum().sort_values(ascending=False)
    total_high_risk = skin_tone_bias.sum()
    
    for tone, count in skin_tone_bias.items():
        percentage = (count / total_high_risk) * 100
        print(f"  Skin Tone {tone}: {count} cases ({percentage:.1f}%)")


# Define dataset combinations to analyze
dataset_combinations = [
    'combined_all.txt',
    'combined_minus_dermie.txt',
    'combined_minus_fitz.txt',
    'combined_minus_india.txt',
    'combined_minus_pad.txt',
    'combined_minus_scin.txt']

# Define baseline model name (adjust based on your actual baseline model name)
BASELINE_MODEL = "train_Baseline"  # Change this to your actual baseline model name

# Load data from all dataset combinations
all_data = {}
baseline_data = {}

print("Loading misclassification data from all dataset combinations...")

for combination in dataset_combinations:
    log_path = os.path.join(log_directory, combination)
    
    if os.path.exists(log_path):
        print(f"Processing {combination}...")
        df_dict = parse_combined_log(log_path)
        
        dataset_name = combination.replace('combined_', '').replace('.txt', '')
        all_data[dataset_name] = df_dict
        
        # Extract baseline model data specifically
        misclassifications_by_tone = df_dict['MisclassificationDetails']
        baseline_misclass = misclassifications_by_tone[
            misclassifications_by_tone['Model'] == BASELINE_MODEL
        ]
        
        if len(baseline_misclass) > 0:
            baseline_data[dataset_name] = baseline_misclass
            print(f"  Found {len(baseline_misclass)} baseline misclassifications for {dataset_name}")
        else:
            print(f"  No baseline model data found for {dataset_name}")
    else:
        print(f"Warning: {combination} not found, skipping...")

# Define high-risk misclassification pairs (same as original)
high_risk_pairs = [
    ('melanoma', 'nevus'),
    ('melanoma', 'seborrheic_keratosis'),
    ('melanoma', 'melanocytic_nevus'),
    ('melanoma', 'seborrheic'),
    ('melanoma', 'acne'),
    ('melanoma', 'eczema'),
    ('melanoma', 'psoriasis'),
    ('melanoma', 'urticaria'),
    ('basal_cell_carcinoma', 'nevus'),
    ('basal_cell_carcinoma', 'seborrheic_keratosis'),
    ('basal_cell_carcinoma', 'melanocytic_nevus'),
    ('bcc', 'nevus'),
    ('bcc', 'seborrheic'),
    ('bcc', 'melanocytic'),
    ('bcc', 'acne'),
    ('bcc', 'eczema'),
    ('bcc', 'psoriasis'),
    ('bcc', 'urticaria'),
    ('squamous_cell_carcinoma', 'nevus'),
    ('squamous_cell_carcinoma', 'seborrheic_keratosis'),
    ('squamous_cell_carcinoma', 'melanocytic_nevus'),
    ('scc', 'nevus'),
    ('scc', 'seborrheic'),
    ('scc', 'melanocytic'),
    ('scc', 'acne'),
    ('scc', 'eczema'),
    ('scc', 'psoriasis'),
    ('scc', 'urticaria'),
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

# Process baseline data for all dataset combinations
processed_baseline_data = {}

for dataset_name, data in baseline_data.items():
    # Add high-risk indicator
    data['High_Risk'] = data.apply(
        lambda row: is_high_risk_pair(row['True Class'], row['Predicted Class'], high_risk_pairs), 
        axis=1
    )
    
    # Create misclassification pair labels
    data['Misclassification_Pair'] = (
        data['True Class'] + ' → ' + data['Predicted Class']
    )
    
    # Add dataset name to the data
    data['Dataset'] = dataset_name
    
    processed_baseline_data[dataset_name] = data

# Combine all baseline data for analysis
if processed_baseline_data:
    combined_baseline_df = pd.concat(processed_baseline_data.values(), ignore_index=True)
    
    print(f"\n=== Baseline Model Analysis Across Dataset Combinations ===")
    print(f"Total baseline misclassifications across all datasets: {len(combined_baseline_df)}")
    print(f"Dataset combinations analyzed: {list(processed_baseline_data.keys())}")
    
    # ========== NEW VISUALIZATIONS FOR DATASET COMBINATIONS ==========
    
    # 1. Total Misclassifications by Dataset Combination (Baseline Model)
    dataset_totals = combined_baseline_df.groupby('Dataset')['Count'].sum().reset_index()
    dataset_totals = dataset_totals.sort_values('Count', ascending=False)
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(dataset_totals['Dataset'], dataset_totals['Count'], color='steelblue', alpha=0.7)
    plt.xlabel('Dataset Combination')
    plt.ylabel('Total Misclassifications')
    plt.title(f'Total Misclassifications by Dataset Combination - {BASELINE_MODEL}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(dataset_totals['Count']):
        plt.text(i, v + max(dataset_totals['Count']) * 0.01, str(v), 
                 ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'baseline_misclassifications_by_dataset.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: baseline_misclassifications_by_dataset.png")
    
    # 2. High-Risk vs Regular Misclassifications by Dataset (Baseline Model)
    high_risk_by_dataset = combined_baseline_df.groupby(['Dataset', 'High_Risk'])['Count'].sum().reset_index()
    high_risk_pivot = high_risk_by_dataset.pivot_table(values='Count', index='Dataset', columns='High_Risk', fill_value=0)
    
    plt.figure(figsize=(14, 8))
    x = np.arange(len(high_risk_pivot.index))
    width = 0.35
    
    bars_regular = None
    bars_high_risk = None
    
    if False in high_risk_pivot.columns:
        bars_regular = plt.bar(x - width/2, high_risk_pivot[False], width, 
                              label='Regular Misclassifications', alpha=0.8)
    if True in high_risk_pivot.columns:
        bars_high_risk = plt.bar(x + width/2, high_risk_pivot[True], width, 
                                label='High-Risk Misclassifications', color='red', alpha=0.8)
    
    plt.xlabel('Dataset Combination')
    plt.ylabel('Number of Misclassifications')
    plt.title(f'High-Risk vs Regular Misclassifications by Dataset - {BASELINE_MODEL}')
    plt.xticks(x, high_risk_pivot.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    if bars_regular:
        for i, v in enumerate(high_risk_pivot[False]):
            if v > 0:
                plt.text(i - width/2, v + max(high_risk_pivot.values.flatten()) * 0.01, str(v), 
                         ha='center', va='bottom')
    
    if bars_high_risk:
        for i, v in enumerate(high_risk_pivot[True]):
            if v > 0:
                plt.text(i + width/2, v + max(high_risk_pivot.values.flatten()) * 0.01, str(v), 
                         ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'baseline_high_risk_by_dataset.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: baseline_high_risk_by_dataset.png")
    
    # 3. Top Misclassification Pairs by Dataset (Baseline Model)
    n_top_pairs = 10
    
    for dataset_name in processed_baseline_data.keys():
        dataset_data = combined_baseline_df[
            (combined_baseline_df['Dataset'] == dataset_name) & 
            (combined_baseline_df['Skin Tone'] == 'All')
        ]
        
        if len(dataset_data) > 0:
            top_pairs = dataset_data.nlargest(n_top_pairs, 'Count')
            
            plt.figure(figsize=(14, 10))
            colors = ['red' if risk else 'steelblue' for risk in top_pairs['High_Risk']]
            
            y_pos = range(len(top_pairs))
            bars = plt.barh(y_pos, top_pairs['Count'], color=colors, alpha=0.8)
            plt.yticks(y_pos, top_pairs['Misclassification_Pair'])
            plt.xlabel('Number of Misclassifications')
            plt.title(f'Top {min(n_top_pairs, len(top_pairs))} Misclassification Pairs - {BASELINE_MODEL} ({dataset_name})')
            plt.gca().invert_yaxis()
            
            # Add value labels
            for i, v in enumerate(top_pairs['Count']):
                plt.text(v + max(top_pairs['Count']) * 0.01, i, str(v), 
                         va='center', ha='left')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='red', alpha=0.8, label='High-Risk'),
                              Patch(facecolor='steelblue', alpha=0.8, label='Regular')]
            plt.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_directory, f'baseline_top_pairs_{dataset_name}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved: baseline_top_pairs_{dataset_name}.png")
    
    # 4. Comparative Analysis: Dataset Performance Heatmap (Baseline Model)
    # Create a heatmap showing different metrics across datasets
    
    dataset_metrics = []
    for dataset_name in processed_baseline_data.keys():
        dataset_data = combined_baseline_df[combined_baseline_df['Dataset'] == dataset_name]
        all_data_subset = dataset_data[dataset_data['Skin Tone'] == 'All']
        
        total_misclass = all_data_subset['Count'].sum()
        high_risk_count = all_data_subset[all_data_subset['High_Risk'] == True]['Count'].sum()
        high_risk_percentage = (high_risk_count / total_misclass * 100) if total_misclass > 0 else 0
        unique_pairs = all_data_subset['Misclassification_Pair'].nunique()
        
        dataset_metrics.append({
            'Dataset': dataset_name,
            'Total_Misclassifications': total_misclass,
            'High_Risk_Count': high_risk_count,
            'High_Risk_Percentage': high_risk_percentage,
            'Unique_Pairs': unique_pairs
        })
    
    metrics_df = pd.DataFrame(dataset_metrics)
    
    # Create heatmap for different metrics
    metrics_for_heatmap = metrics_df.set_index('Dataset')[['Total_Misclassifications', 'High_Risk_Count', 'High_Risk_Percentage', 'Unique_Pairs']]
    
    # Normalize each column for better visualization
    metrics_normalized = metrics_for_heatmap.copy()
    for col in metrics_normalized.columns:
        if col != 'High_Risk_Percentage':  # Don't normalize percentage
            metrics_normalized[col] = (metrics_normalized[col] - metrics_normalized[col].min()) / (metrics_normalized[col].max() - metrics_normalized[col].min())
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(metrics_normalized.T, 
               annot=metrics_for_heatmap.T, 
               fmt='.0f',
               cmap='YlOrRd',
               cbar_kws={'label': 'Normalized Score'})
    
    plt.title(f'Dataset Performance Metrics - {BASELINE_MODEL}')
    plt.xlabel('Dataset Combination')
    plt.ylabel('Metrics')
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, f'baseline_dataset_metrics_heatmap.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: baseline_dataset_metrics_heatmap.png")
    
    # 5. Skin Tone Analysis Across Datasets (Baseline Model)
    skin_tone_data = combined_baseline_df[combined_baseline_df['Skin Tone'] != 'All']
    
    if len(skin_tone_data) > 0:
        skin_tone_summary = skin_tone_data.groupby(['Dataset', 'Skin Tone'])['Count'].sum().reset_index()
        skin_tone_pivot = skin_tone_summary.pivot_table(values='Count', index='Dataset', columns='Skin Tone', fill_value=0)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(skin_tone_pivot, 
                   annot=True, 
                   fmt='.0f', 
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Misclassifications'})
        
        plt.title(f'Misclassifications by Skin Tone Across Datasets - {BASELINE_MODEL}')
        plt.xlabel('Skin Tone (Fitzpatrick Scale)')
        plt.ylabel('Dataset Combination')
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, f'baseline_skin_tone_across_datasets.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: baseline_skin_tone_across_datasets.png")
    
    # ========== SUMMARY STATISTICS FOR DATASET COMBINATIONS ==========
    
    print(f"\n=== Dataset Combination Analysis Summary ===")
    print(f"Baseline Model: {BASELINE_MODEL}")
    
    print(f"\nMisclassifications by Dataset Combination:")
    for _, row in dataset_totals.iterrows():
        print(f"  {row['Dataset']}: {row['Count']} misclassifications")
    
    print(f"\nHigh-Risk Analysis by Dataset:")
    for _, row in metrics_df.iterrows():
        print(f"  {row['Dataset']}: {row['High_Risk_Count']} high-risk ({row['High_Risk_Percentage']:.1f}%)")
    
    # Find best and worst performing datasets
    best_dataset = metrics_df.loc[metrics_df['High_Risk_Count'].idxmin()]
    worst_dataset = metrics_df.loc[metrics_df['High_Risk_Count'].idxmax()]
    
    print(f"\nDataset Safety Ranking:")
    print(f"  Safest: {best_dataset['Dataset']} ({best_dataset['High_Risk_Count']} high-risk)")
    print(f"  Riskiest: {worst_dataset['Dataset']} ({worst_dataset['High_Risk_Count']} high-risk)")
    
    # Most common misclassifications across all datasets
    print(f"\nMost Common Misclassification Pairs (across all datasets):")
    all_pairs = combined_baseline_df[combined_baseline_df['Skin Tone'] == 'All'].groupby('Misclassification_Pair')['Count'].sum().sort_values(ascending=False).head(10)
    for pair, count in all_pairs.items():
        print(f"  {pair}: {count} cases")

# ========== ORIGINAL CODE FOR ALL MODELS (MODIFIED) ==========

# Continue with original analysis for "all" dataset
if 'all' in all_data:
    print(f"\n=== Original Analysis - All Models on Combined Dataset ===")
    
    # Original code continues here...
    df_dict = all_data['all']
    misclassifications_by_tone = df_dict['MisclassificationDetails']
    misclassified_counts = df_dict['MisclassifiedCounts']
    
    print(f"Found detailed misclassification data: {len(misclassifications_by_tone)} records")
    print(f"Found total misclassification counts: {len(misclassified_counts)} models")
    
    if len(misclassifications_by_tone) == 0:
        print("ERROR: No detailed misclassification data found!")
        print("Please ensure your dataframe.py has been updated with the fixed parsing code.")
        exit()
    
    # Add high-risk indicator to the original data
    misclassifications_by_tone['High_Risk'] = misclassifications_by_tone.apply(
        lambda row: is_high_risk_pair(row['True Class'], row['Predicted Class'], high_risk_pairs), 
        axis=1
    )
    
    # Create misclassification pair labels
    misclassifications_by_tone['Misclassification_Pair'] = (
        misclassifications_by_tone['True Class'] + ' → ' + misclassifications_by_tone['Predicted Class']
    )
###############################


print(f"\nAll misclassification visualizations saved to: {output_directory}")
print(f"Total files created: {len([f for f in os.listdir(output_directory) if f.endswith('.png')])}")