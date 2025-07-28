import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re

# Add parent directory for import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

# Configuration
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneAnalysis"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Data source files
MODEL_COMPARISON_FILE = 'combined_all.txt'  # For comparing models (Baseline, VAE, TABE)
DATASET_COMPARISON_FILE = 'combined_baseline_datasets.txt'  # For comparing dataset configurations

# FST color mapping
fst_color_map = {
    1.0: '#F5D5A0',
    2.0: '#E4B589',
    3.0: '#D1A479',
    4.0: '#C0874F',
    5.0: '#A56635',
    6.0: '#4C2C27'
}

def parse_stratified_f1_scores_from_log(filepath):
    """Parse stratified F1 scores per skin tone from log files"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Check if this is the baseline datasets file format
    if 'Dataset Combination:' in content:
        # This is the baseline datasets file format
        model_blocks = re.split(r'(?=Python Filename:)', content)
        
        f1_scores = []
        
        for block in model_blocks:
            if not block.strip():
                continue
                
            # Extract dataset configuration
            dataset_match = re.search(r'Dataset Combination:\s*(\S+)', block)
            if not dataset_match:
                continue
            dataset_config = dataset_match.group(1)
            
            # Parse Stratified F1 Scores
            stratified_f1_pattern = r'Skin Tone:\s*([\d.]+)\n.*?Per Condition:\n(.*?)(?=\nSkin Tone:|\n\nNumber|\n===|$)'
            stratified_f1_matches = re.findall(stratified_f1_pattern, block, re.DOTALL)
            
            for skin_tone, conditions_text in stratified_f1_matches:
                condition_f1_matches = re.findall(r'(\w+):\s*([\d.]+)%', conditions_text)
                for condition, f1_score in condition_f1_matches:
                    f1_scores.append({
                        'Model': 'Baseline',  # Always baseline in this file
                        'Dataset_Config': dataset_config,
                        'Skin_Tone': float(skin_tone),
                        'Condition': condition,
                        'F1_Score': float(f1_score)
                    })
        
        return pd.DataFrame(f1_scores)
    else:
        # This is the regular combined_all.txt format
        model_blocks = re.split(r'(?=Python Filename:)', content)
        
        f1_scores = []
        
        for block in model_blocks:
            if not block.strip():
                continue
                
            model_match = re.search(r'Python Filename:\s*(\S+)', block)
            if not model_match:
                continue
            model = model_match.group(1).replace('.py', '').replace('train_', '')
            
            # Skip FairDisCo
            if 'fairdisco' in model.lower():
                continue
            
            # Parse Stratified F1 Scores
            stratified_f1_pattern = r'Skin Tone:\s*([\d.]+)\n.*?Per Condition:\n(.*?)(?=\nSkin Tone:|\n\nNumber|\n===|$)'
            stratified_f1_matches = re.findall(stratified_f1_pattern, block, re.DOTALL)
            
            for skin_tone, conditions_text in stratified_f1_matches:
                condition_f1_matches = re.findall(r'(\w+):\s*([\d.]+)%', conditions_text)
                for condition, f1_score in condition_f1_matches:
                    f1_scores.append({
                        'Model': model,
                        'Dataset_Config': 'All Datasets',  # Default for model comparison
                        'Skin_Tone': float(skin_tone),
                        'Condition': condition,
                        'F1_Score': float(f1_score)
                    })
        
        return pd.DataFrame(f1_scores)

def parse_stratified_sensitivity_with_dataset_config(filepath):
    """Parse stratified sensitivity data from baseline datasets file"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split by 'Python Filename:'
    model_blocks = re.split(r'(?=Python Filename:)', content)
    
    sensitivity_data = []
    
    for block in model_blocks:
        if not block.strip():
            continue
            
        # Extract dataset configuration
        dataset_match = re.search(r'Dataset Combination:\s*(\S+)', block)
        if not dataset_match:
            continue
        dataset_config = dataset_match.group(1)
        
        # Parse stratified sensitivities
        skin_tone_sections = re.findall(r'Skin Tone:\s*([\d.]+)\n(.*?)(?=\nSkin Tone:|\n\n===|\nNumber|$)', block, re.DOTALL)

        for skin_tone, section_content in skin_tone_sections:
            # Look for condition sensitivities in this skin tone section
            condition_matches = re.findall(r'Condition:\s*(.+?), Top-1 Sensitivity:\s*([\d.]+)%', section_content)
            
            for condition, sensitivity in condition_matches:
                sensitivity_data.append({
                    'Model': 'Baseline',  # Always baseline in this file
                    'Dataset_Config': dataset_config,
                    'Skin_Tone': float(skin_tone),
                    'Condition': condition,
                    'Metric': 'Top-1 Sensitivity',
                    'Value': float(sensitivity)
                })
    
    return pd.DataFrame(sensitivity_data)

def clean_model_names(df, column='Model'):
    """Clean and standardize model names"""
    model_mapping = {
        'Baseline': 'Baseline',
        'VAE': 'VAE',
        'TABE': 'TABE',
        # Handle train_ prefixed versions
        'train_Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'train_TABE': 'TABE'
    }
    
    df[column] = df[column].replace(model_mapping)
    return df

def load_skin_tone_data():
    """Load stratified sensitivity and F1 score data from the correct log files"""

    print("Loading model comparison data from combined_all.txt...")

    # Load model comparison data (Baseline vs VAE vs TABE)
    model_comparison_path = os.path.join(log_directory, MODEL_COMPARISON_FILE)

    if not os.path.exists(model_comparison_path):
        print(f"ERROR: {model_comparison_path} not found!")
        return pd.DataFrame(), pd.DataFrame()

    # Parse model comparison data
    model_df_dict = parse_combined_log(model_comparison_path)
    model_sensitivity_data = model_df_dict['StratifiedSensitivities'].copy()
    model_sensitivity_data['Dataset_Config'] = 'All Datasets'

    # Normalize skin tone column in model data
    if 'Skin Tone' in model_sensitivity_data.columns:
        model_sensitivity_data = model_sensitivity_data.rename(columns={'Skin Tone': 'Skin_Tone'})

    model_f1_data = parse_stratified_f1_scores_from_log(model_comparison_path)
    if 'Skin Tone' in model_f1_data.columns:
        model_f1_data = model_f1_data.rename(columns={'Skin Tone': 'Skin_Tone'})

    print(f"Raw stratified sensitivity data from combined_all.txt:")
    print(f"  Records: {len(model_sensitivity_data)}")
    print(f"  Models: {sorted(model_sensitivity_data['Model'].unique()) if len(model_sensitivity_data) > 0 else 'No data'}")
    print(f"  Skin tones: {sorted(model_sensitivity_data['Skin_Tone'].dropna().unique()) if 'Skin_Tone' in model_sensitivity_data.columns else 'No data'}")
    print(f"  Conditions: {sorted(model_sensitivity_data['Condition'].unique()) if len(model_sensitivity_data) > 0 else 'No data'}")

    print("Loading dataset comparison data from combined_baseline_datasets.txt...")

    dataset_comparison_path = os.path.join(log_directory, DATASET_COMPARISON_FILE)

    if not os.path.exists(dataset_comparison_path):
        print(f"ERROR: {dataset_comparison_path} not found!")
        return model_sensitivity_data, model_f1_data

    dataset_sensitivity_data = parse_stratified_sensitivity_with_dataset_config(dataset_comparison_path)
    dataset_f1_data = parse_stratified_f1_scores_from_log(dataset_comparison_path)

    # Normalize skin tone column in dataset data
    if 'Skin Tone' in dataset_sensitivity_data.columns:
        dataset_sensitivity_data = dataset_sensitivity_data.rename(columns={'Skin Tone': 'Skin_Tone'})
    if 'Skin Tone' in dataset_f1_data.columns:
        dataset_f1_data = dataset_f1_data.rename(columns={'Skin Tone': 'Skin_Tone'})

    print(f"Dataset stratified sensitivity data:")
    print(f"  Records: {len(dataset_sensitivity_data)}")
    print(f"  Skin tones found: {sorted(dataset_sensitivity_data['Skin_Tone'].dropna().unique()) if 'Skin_Tone' in dataset_sensitivity_data.columns else 'None'}")

    # Combine model and dataset data
    combined_sensitivity = pd.concat([model_sensitivity_data, dataset_sensitivity_data], ignore_index=True)
    combined_f1 = pd.concat([model_f1_data, dataset_f1_data], ignore_index=True)

    # Remove FairDisco
    combined_sensitivity = combined_sensitivity[combined_sensitivity['Model'] != 'train_FairDisCo'].copy()
    combined_f1 = combined_f1[combined_f1['Model'] != 'train_FairDisCo'].copy()


    print(f"After combining data:")
    print(f"  Combined sensitivity shape: {combined_sensitivity.shape}")
    print(f"  Combined sensitivity columns: {list(combined_sensitivity.columns)}")
    print(f"  Skin tones after concat: {sorted(combined_sensitivity['Skin_Tone'].dropna().unique()) if 'Skin_Tone' in combined_sensitivity.columns else 'Missing'}")

    # Filter for Top-1 sensitivity
    if 'Metric' in combined_sensitivity.columns:
        print(f"Before filtering - Metrics available: {combined_sensitivity['Metric'].unique()}")
        combined_sensitivity = combined_sensitivity[
            combined_sensitivity['Metric'] == 'Top-1 Sensitivity'
        ].copy()
        print(f"After filtering for Top-1 Sensitivity: {len(combined_sensitivity)} records")
    else:
        print("WARNING: No 'Metric' column found in sensitivity data!")

    # Clean model names
    if len(combined_sensitivity) > 0:
        combined_sensitivity = clean_model_names(combined_sensitivity)
    if len(combined_f1) > 0:
        combined_f1 = clean_model_names(combined_f1)

    # Final output summary
    print(f"Final results:")
    print(f"  Total stratified sensitivity records: {len(combined_sensitivity)}")
    print(f"  Total stratified F1 records: {len(combined_f1)}")
    print(f"  Models found: {sorted(combined_sensitivity['Model'].unique())}")
    print(f"  Dataset configurations: {sorted(combined_sensitivity['Dataset_Config'].unique())}")
    if 'Skin_Tone' in combined_sensitivity.columns:
        print(f"  Skin tones found: {sorted(combined_sensitivity['Skin_Tone'].dropna().unique())}")
    else:
        print("  WARNING: No 'Skin_Tone' column found in final data!")

    return combined_sensitivity, combined_f1


def create_model_skin_tone_charts(sensitivity_data, f1_data, save_path):
    """Create charts showing performance by model across skin tones with disparity subplots"""
    
    # Use 'All Datasets' configuration for model comparison
    sens_model_data = sensitivity_data[
        sensitivity_data['Dataset_Config'] == 'All Datasets'
    ].copy()
    
    f1_model_data = f1_data[
        f1_data['Dataset_Config'] == 'All Datasets'
    ].copy()
    
    print(f"Model comparison data - Sensitivity records: {len(sens_model_data)}")
    print(f"Model comparison data - F1 records: {len(f1_model_data)}")
    
    # 1. F1 Score by Model across Skin Tones with Disparity
    print("Creating F1 score by model across skin tones chart...")
    
    if len(f1_model_data) > 0:
        # Average F1 scores across conditions for each model and skin tone
        f1_avg_data = f1_model_data.groupby(['Model', 'Skin_Tone'])['F1_Score'].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main F1 chart
        models = sorted(f1_avg_data['Model'].unique())
        skin_tones = sorted(f1_avg_data['Skin_Tone'].unique())
        
        x = np.arange(len(models))
        width = 0.12  # Width of bars
        
        # Create bars for each FST
        for i, fst in enumerate(skin_tones):
            fst_data = f1_avg_data[f1_avg_data['Skin_Tone'] == fst]
            
            # Ensure we have data for all models
            scores = []
            for model in models:
                model_data = fst_data[fst_data['Model'] == model]
                if len(model_data) > 0:
                    scores.append(model_data['F1_Score'].iloc[0])
                else:
                    scores.append(0)
            
            ax1.bar(x + i*width, scores, width, 
                   label=f'FST {int(fst)}', 
                   color=fst_color_map[fst], 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=0.5)
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('F1 Score (%)', fontsize=12)
        ax1.set_title('F1 Score by Model and Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(models)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # F1 Disparity subplot
        f1_pivot = f1_avg_data.pivot_table(values='F1_Score', index='Model', columns='Skin_Tone', fill_value=0)
        f1_std = f1_pivot.std(axis=1)
        
        model_colors = {'Baseline': '#2E86C1', 'VAE': '#28B463', 'TABE': '#F39C12'}
        bars = ax2.bar(f1_std.index, f1_std.values, 
                       color=[model_colors.get(model, '#888888') for model in f1_std.index],
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
        
        # Add value labels
        for bar, std_val in zip(bars, f1_std.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('F1 Score Disparity (Std Dev)', fontsize=12)
        ax2.set_title('F1 Score Skin Tone Disparity by Model', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'f1_score_by_model_skin_tone_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: f1_score_by_model_skin_tone_with_disparity.png")
    
    # 2. Sensitivity by Model across Skin Tones with Disparity
    print("Creating sensitivity by model across skin tones chart...")
    
    print(f"DEBUG - sens_model_data shape: {sens_model_data.shape}")
    print(f"DEBUG - sens_model_data columns: {list(sens_model_data.columns)}")
    print(f"DEBUG - sens_model_data models: {sorted(sens_model_data['Model'].unique()) if len(sens_model_data) > 0 else 'No data'}")
    print(f"DEBUG - sens_model_data skin tones: {sorted(sens_model_data['Skin_Tone'].unique()) if len(sens_model_data) > 0 else 'No data'}")
    
    if len(sens_model_data) > 0:
        print(f"DEBUG - Sample sensitivity data:")
        print(sens_model_data.head())
        
        # Average sensitivity across conditions for each model and skin tone
        sens_avg_data = sens_model_data.groupby(['Model', 'Skin_Tone'])['Value'].mean().reset_index()
        
        print(f"DEBUG - sens_avg_data shape: {sens_avg_data.shape}")
        print(f"DEBUG - sens_avg_data sample:")
        print(sens_avg_data.head())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main sensitivity chart
        models = sorted(sens_avg_data['Model'].unique())
        skin_tones = sorted(sens_avg_data['Skin_Tone'].unique())
        
        print(f"DEBUG - Chart models: {models}")
        print(f"DEBUG - Chart skin tones: {skin_tones}")
        
        x = np.arange(len(models))
        width = 0.12  # Width of bars
        
        # Create bars for each FST
        for i, fst in enumerate(skin_tones):
            fst_data = sens_avg_data[sens_avg_data['Skin_Tone'] == fst]
            
            # Ensure we have data for all models
            scores = []
            for model in models:
                model_data = fst_data[fst_data['Model'] == model]
                if len(model_data) > 0:
                    scores.append(model_data['Value'].iloc[0])
                else:
                    scores.append(0)
            
            print(f"DEBUG - FST {fst} scores: {scores}")
            
            ax1.bar(x + i*width, scores, width, 
                   label=f'FST {int(fst)}', 
                   color=fst_color_map[fst], 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=0.5)
        
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('Sensitivity (%)', fontsize=12)
        ax1.set_title('Sensitivity by Model and Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(models)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Sensitivity Disparity subplot
        sens_pivot = sens_avg_data.pivot_table(values='Value', index='Model', columns='Skin_Tone', fill_value=0)
        sens_std = sens_pivot.std(axis=1)
        
        model_colors = {'Baseline': '#2E86C1', 'VAE': '#28B463', 'TABE': '#F39C12'}
        bars = ax2.bar(sens_std.index, sens_std.values, 
                       color=[model_colors.get(model, '#888888') for model in sens_std.index],
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
        
        # Add value labels
        for bar, std_val in zip(bars, sens_std.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('Sensitivity Disparity (Std Dev)', fontsize=12)
        ax2.set_title('Sensitivity Skin Tone Disparity by Model', fontsize=14, fontweight='bold')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sensitivity_by_model_skin_tone_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: sensitivity_by_model_skin_tone_with_disparity.png")
    else:
        print("WARNING: No sensitivity data found for model comparison chart!")

def create_dataset_skin_tone_charts(sensitivity_data, f1_data, save_path):
    """Create charts showing performance by dataset across skin tones with disparity subplots"""
    
    # Use Baseline model for dataset comparison, exclude 'All Datasets'
    sens_dataset_data = sensitivity_data[
        (sensitivity_data['Model'] == 'Baseline') & 
        (sensitivity_data['Dataset_Config'] != 'All Datasets')
    ].copy()
    
    f1_dataset_data = f1_data[
        (f1_data['Model'] == 'Baseline') & 
        (f1_data['Dataset_Config'] != 'All Datasets')
    ].copy()
    
    print(f"Dataset comparison data - Sensitivity records: {len(sens_dataset_data)}")
    print(f"Dataset comparison data - F1 records: {len(f1_dataset_data)}")
    
    # Dataset colors
    dataset_colors = {
        'All': '#1f77b4',
        'Minus_Dermie': '#ff7f0e',
        'Minus_Fitz': '#2ca02c',
        'Minus_India': '#d62728',
        'Minus_PAD': '#9467bd',
        'Minus_SCIN': '#8c564b'
    }
    
    # 3. F1 Score by Dataset across Skin Tones with Disparity
    print("Creating F1 score by dataset across skin tones chart...")
    
    if len(f1_dataset_data) > 0:
        # Average F1 scores across conditions for each dataset and skin tone
        f1_avg_data = f1_dataset_data.groupby(['Dataset_Config', 'Skin_Tone'])['F1_Score'].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main F1 chart
        datasets = sorted(f1_avg_data['Dataset_Config'].unique())
        skin_tones = sorted(f1_avg_data['Skin_Tone'].unique())
        
        x = np.arange(len(datasets))
        width = 0.12  # Width of bars
        
        # Create bars for each FST
        for i, fst in enumerate(skin_tones):
            fst_data = f1_avg_data[f1_avg_data['Skin_Tone'] == fst]
            
            # Ensure we have data for all datasets
            scores = []
            for dataset in datasets:
                dataset_data = fst_data[fst_data['Dataset_Config'] == dataset]
                if len(dataset_data) > 0:
                    scores.append(dataset_data['F1_Score'].iloc[0])
                else:
                    scores.append(0)
            
            ax1.bar(x + i*width, scores, width, 
                   label=f'FST {int(fst)}', 
                   color=fst_color_map[fst], 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=0.5)
        
        ax1.set_xlabel('Dataset Configuration', fontsize=12)
        ax1.set_ylabel('F1 Score (%)', fontsize=12)
        ax1.set_title('F1 Score by Dataset and Fitzpatrick Skin Type (Baseline Model)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # F1 Disparity subplot
        f1_pivot = f1_avg_data.pivot_table(values='F1_Score', index='Dataset_Config', columns='Skin_Tone', fill_value=0)
        f1_std = f1_pivot.std(axis=1)
        
        bars = ax2.bar(range(len(f1_std)), f1_std.values, 
                       color=[dataset_colors.get(dataset, '#888888') for dataset in f1_std.index],
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
        
        # Add value labels
        for bar, std_val in zip(bars, f1_std.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Dataset Configuration', fontsize=12)
        ax2.set_ylabel('F1 Score Disparity (Std Dev)', fontsize=12)
        ax2.set_title('F1 Score Skin Tone Disparity by Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(f1_std)))
        ax2.set_xticklabels(f1_std.index, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'f1_score_by_dataset_skin_tone_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: f1_score_by_dataset_skin_tone_with_disparity.png")
    
    # 4. Sensitivity by Dataset across Skin Tones with Disparity
    print("Creating sensitivity by dataset across skin tones chart...")
    
    if len(sens_dataset_data) > 0:
        # Average sensitivity across conditions for each dataset and skin tone
        sens_avg_data = sens_dataset_data.groupby(['Dataset_Config', 'Skin_Tone'])['Value'].mean().reset_index()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Main sensitivity chart
        datasets = sorted(sens_avg_data['Dataset_Config'].unique())
        skin_tones = sorted(sens_avg_data['Skin_Tone'].unique())
        
        x = np.arange(len(datasets))
        width = 0.12  # Width of bars
        
        # Create bars for each FST
        for i, fst in enumerate(skin_tones):

            fst_data = sens_avg_data[sens_avg_data['Skin_Tone'] == fst]
            
            # Ensure we have data for all datasets
            scores = []
            for dataset in datasets:
                dataset_data = fst_data[fst_data['Dataset_Config'] == dataset]
                if len(dataset_data) > 0:
                    scores.append(dataset_data['Value'].iloc[0])
                else:
                    scores.append(0)
            
            ax1.bar(x + i*width, scores, width, 
                   label=f'FST {int(fst)}', 
                   color=fst_color_map[fst], 
                   alpha=0.8, 
                   edgecolor='black', 
                   linewidth=0.5)
        
        ax1.set_xlabel('Dataset Configuration', fontsize=12)
        ax1.set_ylabel('Sensitivity (%)', fontsize=12)
        ax1.set_title('Sensitivity by Dataset and Fitzpatrick Skin Type (Baseline Model)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Sensitivity Disparity subplot
        sens_pivot = sens_avg_data.pivot_table(values='Value', index='Dataset_Config', columns='Skin_Tone', fill_value=0)
        sens_std = sens_pivot.std(axis=1)
        
        bars = ax2.bar(range(len(sens_std)), sens_std.values, 
                       color=[dataset_colors.get(dataset, '#888888') for dataset in sens_std.index],
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
        
        # Add value labels
        for bar, std_val in zip(bars, sens_std.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{std_val:.1f}', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_xlabel('Dataset Configuration', fontsize=12)
        ax2.set_ylabel('Sensitivity Disparity (Std Dev)', fontsize=12)
        ax2.set_title('Sensitivity Skin Tone Disparity by Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(sens_std)))
        ax2.set_xticklabels(sens_std.index, rotation=45, ha='right')
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'sensitivity_by_dataset_skin_tone_with_disparity.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: sensitivity_by_dataset_skin_tone_with_disparity.png")

def main():
    """Main function to run skin tone analysis"""
    
    print("=== Skin Tone Analysis ===")
    print("Loading stratified sensitivity and F1 score data...")
    
    # Load all skin tone data
    sensitivity_data, f1_data = load_skin_tone_data()
    
    
    if len(sensitivity_data) == 0:
        print("ERROR: No sensitivity data loaded!")
        return
    
    print(f"Models found: {sorted(sensitivity_data['Model'].unique())}")
    print(f"Dataset configurations: {sorted(sensitivity_data['Dataset_Config'].unique())}")
    print(f"Skin tones: {sorted(sensitivity_data['Skin_Tone'].unique())}")
    
    # Create visualizations
    print("\n=== Creating Model Performance Charts ===")
    create_model_skin_tone_charts(sensitivity_data, f1_data, output_directory)
    
    print("\n=== Creating Dataset Performance Charts ===")
    create_dataset_skin_tone_charts(sensitivity_data, f1_data, output_directory)
    
    print(f"\n✅ All visualizations saved to: {output_directory}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    # Model disparity summary
    all_data = sensitivity_data[sensitivity_data['Dataset_Config'] == 'All Datasets']
    if len(all_data) > 0:
        model_avg_data = all_data.groupby(['Model', 'Skin_Tone'])['Value'].mean().reset_index()
        model_pivot = model_avg_data.pivot_table(values='Value', index='Model', columns='Skin_Tone', fill_value=0)
        model_std = model_pivot.std(axis=1)
        
        print("\nSensitivity Skin Tone Disparity by Model:")
        for model, std_val in model_std.items():
            print(f"  {model}: {std_val:.2f}% std deviation")
    
    # Dataset disparity summary
    baseline_data = sensitivity_data[
        (sensitivity_data['Model'] == 'Baseline') & 
        (sensitivity_data['Dataset_Config'] != 'All Datasets')
    ]
    if len(baseline_data) > 0:
        dataset_avg_data = baseline_data.groupby(['Dataset_Config', 'Skin_Tone'])['Value'].mean().reset_index()
        dataset_pivot = dataset_avg_data.pivot_table(values='Value', index='Dataset_Config', columns='Skin_Tone', fill_value=0)
        dataset_std = dataset_pivot.std(axis=1)
        
        print(f"\nSensitivity Skin Tone Disparity by Dataset (Baseline Model):")
        for dataset, std_val in dataset_std.items():
            print(f"  {dataset}: {std_val:.2f}% std deviation")

if __name__ == "__main__":
    main()