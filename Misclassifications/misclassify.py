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

# Define dataset combinations to analyze
dataset_combinations = [
    'combined_all.txt',
    'combined_minus_dermie.txt',
    'combined_minus_fitz.txt',
    'combined_minus_india.txt',
    'combined_minus_pad.txt',
    'combined_minus_scin.txt'
]

# Define baseline model name
BASELINE_MODEL = "train_Baseline"

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
        print(f"Warning: {combination} not found, skipping...")

# Process baseline data for all dataset combinations
processed_baseline_data = {}

for dataset_name, data in baseline_data.items():
    data = data.copy()  # Make a copy to avoid modifying original
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

# ========== REQUIRED VISUALIZATIONS ==========

# 1. Total detailed misclassifications by model (using 'all' dataset)
if 'all' in all_data:
    print("\n=== Creating total_detailed_misclassifications_by_model.png ===")
    df_dict = all_data['all']
    misclassifications_by_tone = df_dict['MisclassificationDetails']
    
    # Add high-risk indicator
    misclassifications_by_tone['High_Risk'] = misclassifications_by_tone.apply(
        lambda row: is_high_risk_pair(row['True Class'], row['Predicted Class'], high_risk_pairs), 
        axis=1
    )
    
    # Create misclassification pair labels
    misclassifications_by_tone['Misclassification_Pair'] = (
        misclassifications_by_tone['True Class'] + ' → ' + misclassifications_by_tone['Predicted Class']
    )
    
    # Total misclassifications by model
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

    # 2. Top misclassification pairs by specific models
    target_models = ['train_Baseline', 'train_VAE', 'train_TABE', 'train_FairDisCo']
    n_top_pairs = 15
    
    for model in target_models:
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

# 3. Baseline top pairs for different datasets
if processed_baseline_data:
    print("\n=== Creating baseline top pairs for different datasets ===")
    n_top_pairs = 10
    
    for dataset_name in processed_baseline_data.keys():
        dataset_data = processed_baseline_data[dataset_name]
        all_data_subset = dataset_data[dataset_data['Skin Tone'] == 'All']
        
        if len(all_data_subset) > 0:
            top_pairs = all_data_subset.nlargest(n_top_pairs, 'Count')
            
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

# 4. High-risk misclassifications by skin tone and model (using 'all' dataset)
if 'all' in all_data:
    print("\n=== Creating high-risk misclassifications by skin tone and model ===")
    
    # Filter for high-risk misclassifications only, excluding 'All' skin tone
    high_risk_skin_tone = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] != 'All')
    ]
    
    if len(high_risk_skin_tone) > 0:
        # Group by model and skin tone
        skin_tone_summary = high_risk_skin_tone.groupby(['Model', 'Skin Tone'])['Count'].sum().reset_index()
        
        # Create a pivot table
        skin_tone_pivot = skin_tone_summary.pivot_table(values='Count', index='Model', columns='Skin Tone', fill_value=0)
        
        # Define Fitzpatrick skin tone color mapping
        fst_color_map = {
            1: '#F5D5A0',
            2: '#E4B589',
            3: '#D1A479',
            4: '#C0874F',
            5: '#A56635',
            6: '#4C2C27'
        }
        
        skin_tones = skin_tone_pivot.columns
        colors = [fst_color_map.get(int(tone), '#888888') for tone in skin_tones]
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        x = np.arange(len(skin_tone_pivot.index))
        width = 0.8 / len(skin_tones)
        
        for i, (skin_tone, color) in enumerate(zip(skin_tones, colors)):
            offset = (i - len(skin_tones)/2 + 0.5) * width
            bars = plt.bar(x + offset, skin_tone_pivot[skin_tone], width, 
                          label=f'Skin Tone {skin_tone}', color=color, alpha=0.8)
            
            # Add value labels on bars
            for j, v in enumerate(skin_tone_pivot[skin_tone]):
                if v > 0:
                    plt.text(j + offset, v + max(skin_tone_pivot.values.flatten()) * 0.01, str(v), 
                             ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Model')
        plt.ylabel('Count')
        plt.title('High-Risk Misclassifications by Model and Skin Tone')
        plt.xticks(x, skin_tone_pivot.index, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'high_risk_by_model_and_skin_tone.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: high_risk_by_model_and_skin_tone.png")

# 5. High-risk misclassifications by skin tone and dataset (baseline only)
if processed_baseline_data:
    print("\n=== Creating high-risk misclassifications by skin tone and dataset (baseline) ===")
    
    # Combine all baseline data
    combined_baseline_df = pd.concat(processed_baseline_data.values(), ignore_index=True)
    
    # Filter for high-risk misclassifications only, excluding 'All' skin tone
    high_risk_baseline_skin_tone = combined_baseline_df[
        (combined_baseline_df['High_Risk'] == True) &
        (combined_baseline_df['Skin Tone'] != 'All')
    ]
    
    if len(high_risk_baseline_skin_tone) > 0:
        # Group by dataset and skin tone
        dataset_skin_tone_summary = high_risk_baseline_skin_tone.groupby(['Dataset', 'Skin Tone'])['Count'].sum().reset_index()
        
        # Create a pivot table
        dataset_skin_tone_pivot = dataset_skin_tone_summary.pivot_table(values='Count', index='Dataset', columns='Skin Tone', fill_value=0)
        
        # Define Fitzpatrick skin tone color mapping
        fst_color_map = {
            1: '#F5D5A0',
            2: '#E4B589',
            3: '#D1A479',
            4: '#C0874F',
            5: '#A56635',
            6: '#4C2C27'
        }
        
        skin_tones = dataset_skin_tone_pivot.columns
        colors = [fst_color_map.get(int(tone), '#888888') for tone in skin_tones]
        
        plt.figure(figsize=(14, 8))
        
        # Create grouped bar chart
        x = np.arange(len(dataset_skin_tone_pivot.index))
        width = 0.8 / len(skin_tones)
        
        for i, (skin_tone, color) in enumerate(zip(skin_tones, colors)):
            offset = (i - len(skin_tones)/2 + 0.5) * width
            bars = plt.bar(x + offset, dataset_skin_tone_pivot[skin_tone], width, 
                          label=f'Skin Tone {skin_tone}', color=color, alpha=0.8)
            
            # Add value labels on bars
            for j, v in enumerate(dataset_skin_tone_pivot[skin_tone]):
                if v > 0:
                    plt.text(j + offset, v + max(dataset_skin_tone_pivot.values.flatten()) * 0.01, str(v), 
                             ha='center', va='bottom', fontsize=8)
        
        plt.xlabel('Dataset Combination')
        plt.ylabel('Count')
        plt.title('High-Risk Misclassifications by Dataset and Skin Tone (Baseline Model)')
        plt.xticks(x, dataset_skin_tone_pivot.index, rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'high_risk_by_dataset_and_skin_tone_baseline.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: high_risk_by_dataset_and_skin_tone_baseline.png")

print(f"\nAll required visualizations saved to: {output_directory}")
print("Files created:")
print("- total_detailed_misclassifications_by_model.png")
print("- top_misclassifications_train_Baseline.png")
print("- top_misclassifications_train_VAE.png") 
print("- top_misclassifications_train_TABE.png")
print("- top_misclassifications_train_FairDisCo.png")
print("- baseline_top_pairs_all.png")
print("- baseline_top_pairs_minus_dermie.png")
print("- baseline_top_pairs_minus_fitz.png")
print("- baseline_top_pairs_minus_india.png")
print("- baseline_top_pairs_minus_pad.png")
print("- baseline_top_pairs_minus_scin.png")
print("- high_risk_by_model_and_skin_tone.png")
print("- high_risk_by_dataset_and_skin_tone_baseline.png")