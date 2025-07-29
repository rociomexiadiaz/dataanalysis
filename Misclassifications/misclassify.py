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
    ('cancer â', 'non-cancer')
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

def create_overall_misclassifications_chart(misclassifications_by_tone, output_directory):
    """Create overall misclassifications chart by model"""
    print("\n=== Creating Overall Misclassifications Chart ===")
    
    # Filter data for 'All' skin tone to avoid double counting
    all_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] == 'All']
    
    # Total misclassifications by model (from detailed data)
    total_misclass = all_tone_data.groupby('Model')['Count'].sum().reset_index()
    total_misclass.columns = ['Model', 'Total_Misclassifications']
    
    # Clean model names and exclude FairDisCo
    total_misclass['Model'] = total_misclass['Model'].replace({
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo'
    })
    
    # Filter out FairDisCo
    total_misclass = total_misclass[total_misclass['Model'] != 'FairDisCo']
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(total_misclass['Model'], total_misclass['Total_Misclassifications'],
                   alpha=0.8, color='steelblue')
    plt.xlabel('Model')
    plt.ylabel('Total Misclassifications')
    plt.title('Overall Number of Misclassifications per Model')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(total_misclass['Total_Misclassifications']):
        plt.text(i, v + max(total_misclass['Total_Misclassifications']) * 0.01, str(v),
                 ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'overall_misclassifications_by_model.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: overall_misclassifications_by_model.png")

def create_high_low_risk_barchart(misclassifications_by_tone, output_directory):
    """Create bar chart showing high-risk vs low-risk misclassifications per model"""
    print("\n=== Creating High-Risk vs Low-Risk Bar Chart ===")
    
    # Filter data for 'All' skin tone to avoid double counting
    all_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] == 'All']
    
    # Calculate high-risk and low-risk counts
    risk_summary = all_tone_data.groupby(['Model', 'High_Risk'])['Count'].sum().reset_index()
    risk_pivot = risk_summary.pivot_table(values='Count', index='Model', columns='High_Risk', fill_value=0)
    
    # Clean model names and exclude FairDisCo
    risk_pivot.index = risk_pivot.index.str.replace('train_', '')
    risk_pivot = risk_pivot[risk_pivot.index != 'FairDisCo']
    
    # Ensure we have both True and False columns
    if True not in risk_pivot.columns:
        risk_pivot[True] = 0
    if False not in risk_pivot.columns:
        risk_pivot[False] = 0
    
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(risk_pivot.index))
    width = 0.35
    
    bars1 = plt.bar(x - width/2, risk_pivot[False], width, 
                   label='Low-Risk Misclassifications', color='lightblue', alpha=0.8)
    bars2 = plt.bar(x + width/2, risk_pivot[True], width, 
                   label='High-Risk Misclassifications', color='red', alpha=0.8)
    
    plt.xlabel('Model')
    plt.ylabel('Number of Misclassifications')
    plt.title('High-Risk vs Low-Risk Misclassifications by Model')
    plt.xticks(x, risk_pivot.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, v in enumerate(risk_pivot[False]):
        if v > 0:
            plt.text(i - width/2, v + max(risk_pivot.values.flatten()) * 0.01, str(v), 
                     ha='center', va='bottom', fontweight='bold')
    
    for i, v in enumerate(risk_pivot[True]):
        if v > 0:
            plt.text(i + width/2, v + max(risk_pivot.values.flatten()) * 0.01, str(v), 
                     ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'high_low_risk_by_model.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: high_low_risk_by_model.png")

def create_total_misclassifications_heatmap(misclassifications_by_tone, output_directory):
    """Create heatmap with skin tones on y-axis, models on x-axis, colored by TOTAL misclassification count"""
    print("\n=== Creating Total Misclassifications Heatmap ===")
    
    # Filter out 'All' skin tone data as we want specific skin tones
    skin_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] != 'All']
    
    if len(skin_tone_data) == 0:
        print("No skin tone specific data found, skipping this visualization")
        return
    
    # Sum ALL misclassifications by skin tone and model
    heatmap_data = skin_tone_data.groupby(['Skin Tone', 'Model'])['Count'].sum().reset_index()
    
    # Clean model names and exclude FairDisCo
    heatmap_data['Model'] = heatmap_data['Model'].str.replace('train_', '')
    heatmap_data = heatmap_data[heatmap_data['Model'] != 'FairDisCo']
    
    # Create pivot table
    heatmap_pivot = heatmap_data.pivot_table(
        values='Count', 
        index='Skin Tone', 
        columns='Model', 
        fill_value=0
    )
    
    # Sort skin tones numerically if they're numeric
    try:
        heatmap_pivot.index = heatmap_pivot.index.astype(int)
        heatmap_pivot = heatmap_pivot.sort_index()
        heatmap_pivot.index = heatmap_pivot.index.astype(str)
    except:
        pass  # Keep original order if not numeric
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_pivot, 
                annot=True, 
                fmt='.0f', 
                cmap='Blues',
                cbar_kws={'label': 'Total Misclassification Count'},
                linewidths=0.5)
    
    plt.title('Total Misclassifications by Skin Tone and Model', fontweight='bold', pad=20)
    plt.xlabel('Model')
    plt.ylabel('Skin Tone (Fitzpatrick Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'total_misclassifications_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: total_misclassifications_heatmap.png")

def create_high_risk_only_heatmap(misclassifications_by_tone, output_directory):
    """Create heatmap with skin tones on y-axis, models on x-axis, colored by HIGH-RISK misclassification count only"""
    print("\n=== Creating High-Risk Only Heatmap ===")
    
    # Filter for high-risk misclassifications only, excluding 'All' skin tone
    high_risk_data = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] != 'All')
    ]
    
    if len(high_risk_data) == 0:
        print("No high-risk skin tone data found, skipping this visualization")
        return
    
    # Sum high-risk misclassifications by skin tone and model
    heatmap_data = high_risk_data.groupby(['Skin Tone', 'Model'])['Count'].sum().reset_index()
    
    # Clean model names and exclude FairDisCo
    heatmap_data['Model'] = heatmap_data['Model'].str.replace('train_', '')
    heatmap_data = heatmap_data[heatmap_data['Model'] != 'FairDisCo']
    
    # Create pivot table
    heatmap_pivot = heatmap_data.pivot_table(
        values='Count', 
        index='Skin Tone', 
        columns='Model', 
        fill_value=0
    )
    
    # Sort skin tones numerically if they're numeric
    try:
        heatmap_pivot.index = heatmap_pivot.index.astype(int)
        heatmap_pivot = heatmap_pivot.sort_index()
        heatmap_pivot.index = heatmap_pivot.index.astype(str)
    except:
        pass  # Keep original order if not numeric
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(heatmap_pivot, 
                annot=True, 
                fmt='.0f', 
                cmap='Reds',
                cbar_kws={'label': 'High-Risk Misclassification Count'},
                linewidths=0.5)
    
    plt.title('High-Risk Misclassifications by Skin Tone and Model', fontweight='bold', pad=20)
    plt.xlabel('Model')
    plt.ylabel('Skin Tone (Fitzpatrick Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'high_risk_only_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: high_risk_only_heatmap.png")

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

# Create the 4 visualizations
print("\n=== Creating Visualizations ===")

# 1. Overall misclassifications bar chart (excludes FairDisCo)
create_overall_misclassifications_chart(misclassifications_by_tone, output_directory)

# 2. High-risk vs low-risk bar chart (2 bars per model, excludes FairDisCo)
create_high_low_risk_barchart(misclassifications_by_tone, output_directory)

# 3. Total misclassifications heatmap (skin tone vs model, excludes FairDisCo)
create_total_misclassifications_heatmap(misclassifications_by_tone, output_directory)

# 4. High-risk only heatmap (skin tone vs model, excludes FairDisCo)
create_high_risk_only_heatmap(misclassifications_by_tone, output_directory)

# Summary statistics
print("\n=== Summary Statistics ===")

# Total misclassifications by model
all_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] == 'All']
total_misclass = all_tone_data.groupby('Model')['Count'].sum().reset_index()
total_misclass.columns = ['Model', 'Total_Misclassifications']

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

print(f"\nMisclassification visualizations saved to: {output_directory}")
print(f"Total files created: {len([f for f in os.listdir(output_directory) if f.endswith('.png')])}")