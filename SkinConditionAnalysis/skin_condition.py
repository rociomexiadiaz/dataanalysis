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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinConditionAnalysis"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Data source files
MODEL_COMPARISON_FILE = 'combined_all.txt'  # For comparing models (Baseline, VAE, TABE, FairDisCo)
DATASET_COMPARISON_FILE = 'combined_baseline_datasets.txt'  # For comparing dataset configurations

def parse_f1_scores_from_log(filepath):
    """Parse F1 scores per condition from log files"""
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
            
            # Parse Per Condition F1 Scores
            f1_section = re.search(r'Per Condition F1 Score:\n(.*?)(?=\n===|\n\nNumber|$)', block, re.DOTALL)
            if f1_section:
                f1_content = f1_section.group(1)
                f1_condition_matches = re.findall(r'(\w+):\s*([\d.]+)%', f1_content)
                
                for condition, f1_score in f1_condition_matches:
                    f1_scores.append({
                        'Model': 'Baseline',  # Always baseline in this file
                        'Dataset_Config': dataset_config,
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
            
            # Parse Per Condition F1 Scores
            f1_section = re.search(r'Per Condition F1 Score:\n(.*?)(?=\n===|\n\nNumber|$)', block, re.DOTALL)
            if f1_section:
                f1_content = f1_section.group(1)
                f1_condition_matches = re.findall(r'(\w+):\s*([\d.]+)%', f1_content)
                
                for condition, f1_score in f1_condition_matches:
                    f1_scores.append({
                        'Model': model,
                        'Dataset_Config': 'All Datasets',  # Default for model comparison
                        'Condition': condition,
                        'F1_Score': float(f1_score)
                    })
        
        return pd.DataFrame(f1_scores)

def parse_sensitivity_with_dataset_config(filepath):
    """Parse sensitivity data from baseline datasets file with proper dataset configuration mapping"""
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
        
        # Extract datasets info
        datasets_match = re.search(r'Datasets:\s*(.+)', block)
        datasets = datasets_match.group(1) if datasets_match else 'Unknown'
        
        # Parse condition sensitivities
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                sensitivity_data.append({
                    'Model': 'Baseline',  # Always baseline in this file
                    'Datasets': datasets,
                    'Dataset_Config': dataset_config,
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })
    
    return pd.DataFrame(sensitivity_data)

def parse_sensitivity_with_dataset_config(filepath):
    """Parse sensitivity data from baseline datasets file with proper dataset configuration mapping"""
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
        
        # Extract datasets info
        datasets_match = re.search(r'Datasets:\s*(.+)', block)
        datasets = datasets_match.group(1) if datasets_match else 'Unknown'
        
        # Parse condition sensitivities
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                sensitivity_data.append({
                    'Model': 'Baseline',  # Always baseline in this file
                    'Datasets': datasets,
                    'Dataset_Config': dataset_config,
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })
    
    return pd.DataFrame(sensitivity_data)
    """Parse F1 scores per condition from log files"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split models by 'Python Filename:' or 'Dataset Combination:'
    if 'Dataset Combination:' in content:
        # This is the baseline datasets file format
        model_blocks = re.split(r'(?=Dataset Combination:)', content)
        identifier_pattern = r'Dataset Combination:\s*(\S+)'
        datasets_pattern = r'Datasets:\s*(.+)'
    else:
        # This is the regular combined_all.txt format
        model_blocks = re.split(r'(?=Python Filename:)', content)
        identifier_pattern = r'Python Filename:\s*(\S+)'
        datasets_pattern = r'Datasets:\s*(.+)'
    
    f1_scores = []
    
    for block in model_blocks:
        if not block.strip():
            continue
            
        # Try to match the identifier
        identifier_match = re.search(identifier_pattern, block)
        if not identifier_match:
            continue
            
        if 'Dataset Combination:' in content:
            # For baseline datasets file, use dataset combination as identifier
            model = 'Baseline'  # Always baseline in this file
            dataset_config = identifier_match.group(1)
        else:
            # For regular file, extract model name
            model = identifier_match.group(1).replace('.py', '').replace('train_', '')
            datasets_match = re.search(datasets_pattern, block)
            dataset_config = datasets_match.group(1) if datasets_match else 'Unknown'
        
        # Parse Per Condition F1 Scores
        f1_section = re.search(r'Per Condition F1 Score:\n(.*?)(?=\n===|\n\nNumber|$)', block, re.DOTALL)
        if f1_section:
            f1_content = f1_section.group(1)
            f1_condition_matches = re.findall(r'(\w+):\s*([\d.]+)%', f1_content)
            
            for condition, f1_score in f1_condition_matches:
                f1_scores.append({
                    'Model': model,
                    'Dataset_Config': dataset_config,
                    'Condition': condition,
                    'F1_Score': float(f1_score)
                })
    
            return pd.DataFrame(f1_scores)

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
        # FairDisCo is excluded at parsing level
    }
    
    df[column] = df[column].replace(model_mapping)
    return df

def load_all_data():
    """Load sensitivity and F1 score data from the correct log files"""
    
    print("Loading model comparison data from combined_all.txt...")
    
    # Load model comparison data (Baseline vs VAE vs TABE, excluding FairDisCo)
    model_comparison_path = os.path.join(log_directory, MODEL_COMPARISON_FILE)
    
    if not os.path.exists(model_comparison_path):
        print(f"ERROR: {model_comparison_path} not found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Parse model comparison data
    model_df_dict = parse_combined_log(model_comparison_path)
    model_sensitivity_data = model_df_dict['ConditionSensitivities'].copy()
    model_sensitivity_data['Dataset_Config'] = 'All Datasets'  # All models use all datasets
    
    print(f"Raw sensitivity data from combined_all.txt:")
    print(f"  Records: {len(model_sensitivity_data)}")
    print(f"  Models before filtering: {sorted(model_sensitivity_data['Model'].unique())}")
    
    model_f1_data = parse_f1_scores_from_log(model_comparison_path)
    
    print(f"Raw F1 data from combined_all.txt:")
    print(f"  Records: {len(model_f1_data)}")
    print(f"  Models before filtering: {sorted(model_f1_data['Model'].unique()) if len(model_f1_data) > 0 else 'No F1 data'}")
    
    print("Loading dataset comparison data from combined_baseline_datasets.txt...")
    
    # Load dataset comparison data (All vs Minus_Dermie vs Minus_Fitz, etc.)
    dataset_comparison_path = os.path.join(log_directory, DATASET_COMPARISON_FILE)
    
    if not os.path.exists(dataset_comparison_path):
        print(f"ERROR: {dataset_comparison_path} not found!")
        return model_sensitivity_data, model_f1_data
    
    # Parse dataset comparison data using the specialized function
    dataset_sensitivity_data = parse_sensitivity_with_dataset_config(dataset_comparison_path)
    dataset_f1_data = parse_f1_scores_from_log(dataset_comparison_path)
    
    print(f"Raw dataset sensitivity data:")
    print(f"  Records: {len(dataset_sensitivity_data)}")
    print(f"  Models: {sorted(dataset_sensitivity_data['Model'].unique()) if len(dataset_sensitivity_data) > 0 else 'No data'}")
    print(f"  Dataset configs: {sorted(dataset_sensitivity_data['Dataset_Config'].unique()) if len(dataset_sensitivity_data) > 0 else 'No data'}")
    
    # Combine model and dataset data
    combined_sensitivity = pd.concat([model_sensitivity_data, dataset_sensitivity_data], ignore_index=True)
    combined_f1 = pd.concat([model_f1_data, dataset_f1_data], ignore_index=True)
    
    print(f"After combining, before FairDisCo filtering:")
    print(f"  Sensitivity models: {sorted(combined_sensitivity['Model'].unique())}")
    print(f"  F1 models: {sorted(combined_f1['Model'].unique()) if len(combined_f1) > 0 else 'No F1 data'}")
    
    # Filter out FairDisCo from all data - be more specific about the filtering
    print("Filtering out FairDisCo...")
    
    fairdisc_sens_before = len(combined_sensitivity)
    combined_sensitivity = combined_sensitivity[
        ~combined_sensitivity['Model'].str.contains('FairDisCo', case=False, na=False)
    ]
    fairdisc_sens_after = len(combined_sensitivity)
    print(f"  Sensitivity: {fairdisc_sens_before} -> {fairdisc_sens_after} records (removed {fairdisc_sens_before - fairdisc_sens_after})")
    
    fairdisc_f1_before = len(combined_f1) if len(combined_f1) > 0 else 0
    combined_f1 = combined_f1[
        ~combined_f1['Model'].str.contains('FairDisCo', case=False, na=False)
    ]
    fairdisc_f1_after = len(combined_f1) if len(combined_f1) > 0 else 0
    print(f"  F1: {fairdisc_f1_before} -> {fairdisc_f1_after} records (removed {fairdisc_f1_before - fairdisc_f1_after})")
    
    # Handle duplicate "All" vs "All Datasets" more carefully
    # We want to keep both:
    # 1. "All Datasets" entries for model comparison (multiple models)
    # 2. "All" entries for dataset comparison (baseline model only)
    # But remove any duplicate "All Datasets" entries that are just copies of "All"
    
    print("Checking for duplicate handling...")
    print(f"  'All Datasets' sensitivity records: {len(combined_sensitivity[combined_sensitivity['Dataset_Config'] == 'All Datasets'])}")
    print(f"  'All' sensitivity records: {len(combined_sensitivity[combined_sensitivity['Dataset_Config'] == 'All'])}")
    print(f"  'All Datasets' F1 records: {len(combined_f1[combined_f1['Dataset_Config'] == 'All Datasets'])}")
    print(f"  'All' F1 records: {len(combined_f1[combined_f1['Dataset_Config'] == 'All'])}")
    
    # Don't remove any duplicates - we need both configurations for different purposes
    # "All Datasets" = model comparison data
    # "All" = dataset comparison data
    print("Keeping both 'All Datasets' and 'All' configurations for different chart types")
    
    # Filter for Top-1 sensitivity
    combined_sensitivity = combined_sensitivity[
        combined_sensitivity['Metric'] == 'Top-1 Sensitivity'
    ].copy()
    
    # Clean model names
    combined_sensitivity = clean_model_names(combined_sensitivity)
    combined_f1 = clean_model_names(combined_f1)
    
    print(f"Final results:")
    print(f"  Total sensitivity records: {len(combined_sensitivity)}")
    print(f"  Total F1 records: {len(combined_f1)}")
    print(f"  Models found: {sorted(combined_sensitivity['Model'].unique())}")
    print(f"  Dataset configurations: {sorted(combined_sensitivity['Dataset_Config'].unique())}")
    
    return combined_sensitivity, combined_f1

def create_model_performance_charts(sensitivity_data, f1_data, save_path):
    """Create charts showing performance by model with disparity subplots"""
    
    # Use 'All Datasets' configuration for model comparison (from combined_all.txt)
    sens_model_data = sensitivity_data[
        sensitivity_data['Dataset_Config'] == 'All Datasets'
    ].copy()
    
    f1_model_data = f1_data[
        f1_data['Dataset_Config'] == 'All Datasets'
    ].copy()
    
    print(f"Model comparison data - Sensitivity records: {len(sens_model_data)}")
    print(f"Model comparison data - F1 records: {len(f1_model_data)}")
    print(f"Models in sensitivity data: {sorted(sens_model_data['Model'].unique()) if len(sens_model_data) > 0 else 'None'}")
    print(f"Models in F1 data: {sorted(f1_model_data['Model'].unique()) if len(f1_model_data) > 0 else 'None'}")
    
    # Model colors (excluding FairDisCo)
    model_colors = {
        'Baseline': '#2E86C1',
        'VAE': '#28B463', 
        'TABE': '#F39C12'
    }
    
    # 1. F1 Score by Model with Disparity
    print("Creating F1 score by model chart...")
    
    if len(f1_model_data) == 0:
        print("Warning: No F1 data found for model comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main F1 chart
    conditions = sorted(f1_model_data['Condition'].unique())
    models = sorted(f1_model_data['Model'].unique())
    
    x = np.arange(len(conditions))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = f1_model_data[f1_model_data['Model'] == model]
        scores = []
        for condition in conditions:
            cond_data = model_data[model_data['Condition'] == condition]
            if len(cond_data) > 0:
                scores.append(cond_data['F1_Score'].iloc[0])
            else:
                scores.append(0)
        
        ax1.bar(x + i*width, scores, width, 
               label=model, 
               color=model_colors.get(model, '#888888'),
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)
    
    ax1.set_xlabel('Skin Condition', fontsize=12)
    ax1.set_ylabel('F1 Score (%)', fontsize=12)
    ax1.set_title('F1 Score by Condition and Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # F1 Disparity subplot
    f1_pivot = f1_model_data.pivot_table(values='F1_Score', index='Model', columns='Condition', fill_value=0)
    f1_std = f1_pivot.std(axis=1)
    
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
    ax2.set_title('F1 Score Disparity by Model', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'f1_score_by_model_with_disparity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: f1_score_by_model_with_disparity.png")
    
    # 2. Sensitivity by Model with Disparity
    print("Creating sensitivity by model chart...")
    
    if len(sens_model_data) == 0:
        print("Warning: No sensitivity data found for model comparison")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main sensitivity chart
    conditions = sorted(sens_model_data['Condition'].unique())
    models = sorted(sens_model_data['Model'].unique())
    
    print(f"Sensitivity chart - Models found: {models}")
    print(f"Sensitivity chart - Model colors mapping: {[(model, model_colors.get(model, 'NOT_FOUND')) for model in models]}")
    
    x = np.arange(len(conditions))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = sens_model_data[sens_model_data['Model'] == model]
        scores = []
        for condition in conditions:
            cond_data = model_data[model_data['Condition'] == condition]
            if len(cond_data) > 0:
                scores.append(cond_data['Value'].iloc[0])
            else:
                scores.append(0)
        
        ax1.bar(x + i*width, scores, width, 
               label=model, 
               color=model_colors.get(model, '#888888'),
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)
    
    ax1.set_xlabel('Skin Condition', fontsize=12)
    ax1.set_ylabel('Sensitivity (%)', fontsize=12)
    ax1.set_title('Sensitivity by Condition and Model', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Sensitivity Disparity subplot
    sens_pivot = sens_model_data.pivot_table(values='Value', index='Model', columns='Condition', fill_value=0)
    sens_std = sens_pivot.std(axis=1)
    
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
    ax2.set_title('Sensitivity Disparity by Model', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sensitivity_by_model_with_disparity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sensitivity_by_model_with_disparity.png")

def create_dataset_performance_charts(sensitivity_data, f1_data, baseline_model, save_path):
    """Create charts showing performance by dataset with disparity subplots"""
    
    # Debug: Print available models
    print(f"Available models in sensitivity data: {sensitivity_data['Model'].unique()}")
    print(f"Available models in F1 data: {f1_data['Model'].unique()}")
    print(f"Using model '{baseline_model}' for dataset comparison")
    
    sens_dataset_data = sensitivity_data[
        sensitivity_data['Model'] == baseline_model
    ].copy()
    
    f1_dataset_data = f1_data[
        f1_data['Model'] == baseline_model
    ].copy()
    
    # For dataset comparison, exclude 'All Datasets' entries (keep only dataset ablation entries)
    sens_dataset_data = sens_dataset_data[
        sens_dataset_data['Dataset_Config'] != 'All Datasets'
    ]
    
    f1_dataset_data = f1_dataset_data[
        f1_dataset_data['Dataset_Config'] != 'All Datasets'
    ]
    
    print(f"After filtering out 'All Datasets' for dataset comparison:")
    print(f"Available datasets in sensitivity data: {sorted(sens_dataset_data['Dataset_Config'].unique())}")
    print(f"Available datasets in F1 data: {sorted(f1_dataset_data['Dataset_Config'].unique())}")
    
    # Check if we have data
    if len(sens_dataset_data) == 0:
        print(f"ERROR: No sensitivity data found for model '{baseline_model}' after filtering")
        return
    
    if len(f1_dataset_data) == 0:
        print(f"ERROR: No F1 data found for model '{baseline_model}' after filtering")
        return
    
    # Dataset colors - updated for the actual dataset configurations
    dataset_colors = {
        'All': '#1f77b4',
        'Minus_Dermie': '#ff7f0e',
        'Minus_Fitz': '#2ca02c',
        'Minus_India': '#d62728',
        'Minus_PAD': '#9467bd',
        'Minus_SCIN': '#8c564b'
    }
    
    # 3. F1 Score by Dataset with Disparity
    print("Creating F1 score by dataset chart...")
    
    # Check if we have F1 data and datasets
    if len(f1_dataset_data) == 0:
        print("Warning: No F1 data available for dataset comparison. Skipping F1 charts.")
    else:
        conditions = sorted(f1_dataset_data['Condition'].unique())
        datasets = sorted(f1_dataset_data['Dataset_Config'].unique())
        
        print(f"F1 datasets for chart: {datasets}")
        
        if len(datasets) == 0:
            print("Warning: No dataset configurations found in F1 data. Skipping F1 dataset chart.")
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Main F1 chart
            x = np.arange(len(conditions))
            width = 0.8 / len(datasets)
            
            for i, dataset in enumerate(datasets):
                dataset_data = f1_dataset_data[f1_dataset_data['Dataset_Config'] == dataset]
                scores = []
                for condition in conditions:
                    cond_data = dataset_data[dataset_data['Condition'] == condition]
                    if len(cond_data) > 0:
                        scores.append(cond_data['F1_Score'].iloc[0])
                    else:
                        scores.append(0)
                
                ax1.bar(x + i*width, scores, width, 
                       label=dataset, 
                       color=dataset_colors.get(dataset, '#888888'),
                       alpha=0.8,
                       edgecolor='black',
                       linewidth=0.5)
            
            ax1.set_xlabel('Skin Condition', fontsize=12)
            ax1.set_ylabel('F1 Score (%)', fontsize=12)
            ax1.set_title(f'F1 Score by Condition and Dataset ({baseline_model} Model)', fontsize=14, fontweight='bold')
            ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
            ax1.set_xticklabels(conditions, rotation=45, ha='right')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax1.grid(True, axis='y', alpha=0.3)
            
            # F1 Disparity subplot
            f1_pivot = f1_dataset_data.pivot_table(values='F1_Score', index='Dataset_Config', columns='Condition', fill_value=0)
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
            ax2.set_title('F1 Score Disparity by Dataset', fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(f1_std)))
            ax2.set_xticklabels(f1_std.index, rotation=45, ha='right')
            ax2.grid(True, axis='y', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'f1_score_by_dataset_with_disparity.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: f1_score_by_dataset_with_disparity.png")
    
    # 4. Sensitivity by Dataset with Disparity
    print("Creating sensitivity by dataset chart...")
    
    # Check if we have sensitivity data and datasets
    conditions = sorted(sens_dataset_data['Condition'].unique())
    datasets = sorted(sens_dataset_data['Dataset_Config'].unique())
    
    print(f"Sensitivity datasets for chart: {datasets}")
    
    if len(datasets) == 0:
        print("Warning: No dataset configurations found in sensitivity data. Skipping sensitivity dataset chart.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main sensitivity chart
    x = np.arange(len(conditions))
    width = 0.8 / len(datasets)
    
    for i, dataset in enumerate(datasets):
        dataset_data = sens_dataset_data[sens_dataset_data['Dataset_Config'] == dataset]
        scores = []
        for condition in conditions:
            cond_data = dataset_data[dataset_data['Condition'] == condition]
            if len(cond_data) > 0:
                scores.append(cond_data['Value'].iloc[0])
            else:
                scores.append(0)
        
        ax1.bar(x + i*width, scores, width, 
               label=dataset, 
               color=dataset_colors.get(dataset, '#888888'),
               alpha=0.8,
               edgecolor='black',
               linewidth=0.5)
    
    ax1.set_xlabel('Skin Condition', fontsize=12)
    ax1.set_ylabel('Sensitivity (%)', fontsize=12)
    ax1.set_title(f'Sensitivity by Condition and Dataset ({baseline_model} Model)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax1.set_xticklabels(conditions, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Sensitivity Disparity subplot
    sens_pivot = sens_dataset_data.pivot_table(values='Value', index='Dataset_Config', columns='Condition', fill_value=0)
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
    ax2.set_title('Sensitivity Disparity by Dataset', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(sens_std)))
    ax2.set_xticklabels(sens_std.index, rotation=45, ha='right')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sensitivity_by_dataset_with_disparity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sensitivity_by_dataset_with_disparity.png")

def main():
    """Main function to run skin condition analysis"""
    
    print("=== Skin Condition Analysis ===")
    print("Loading sensitivity and F1 score data...")
    
    # Load all data from the correct files
    sensitivity_data, f1_data = load_all_data()
    
    if len(sensitivity_data) == 0:
        print("ERROR: No sensitivity data loaded!")
        return
    
    print(f"Models found: {sorted(sensitivity_data['Model'].unique())}")
    print(f"Dataset configurations: {sorted(sensitivity_data['Dataset_Config'].unique())}")
    print(f"Conditions: {sorted(sensitivity_data['Condition'].unique())}")
    
    # Create visualizations
    print("\n=== Creating Model Performance Charts ===")
    create_model_performance_charts(sensitivity_data, f1_data, output_directory)
    
    print("\n=== Creating Dataset Performance Charts ===")
    # For dataset comparison, we always use Baseline model from the baseline datasets file
    create_dataset_performance_charts(sensitivity_data, f1_data, 'Baseline', output_directory)
    
    print(f"\n✅ All visualizations saved to: {output_directory}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    # Model disparity summary - use 'All' from baseline datasets file
    all_data = sensitivity_data[sensitivity_data['Dataset_Config'] == 'All']
    if len(all_data) == 0:
        # Fallback to 'All Datasets' if 'All' not found
        all_data = sensitivity_data[sensitivity_data['Dataset_Config'] == 'All Datasets']
    
    if len(all_data) > 0:
        sens_pivot = all_data.pivot_table(values='Value', index='Model', columns='Condition', fill_value=0)
        sens_std = sens_pivot.std(axis=1)
        
        print("\nSensitivity Disparity by Model:")
        for model, std_val in sens_std.items():
            print(f"  {model}: {std_val:.2f}% std deviation")
    else:
        print("\nNo model comparison data found for disparity analysis")
    
    # Dataset disparity summary - use Baseline model dataset comparison data (exclude 'All')
    baseline_data = sensitivity_data[
        (sensitivity_data['Model'] == 'Baseline') & 
        (sensitivity_data['Dataset_Config'] != 'All') &  # Exclude the 'All' entry used for model comparison
        (sensitivity_data['Dataset_Config'] != 'All Datasets')  # Also exclude this if it exists
    ]
    if len(baseline_data) > 0:
        dataset_pivot = baseline_data.pivot_table(values='Value', index='Dataset_Config', columns='Condition', fill_value=0)
        dataset_std = dataset_pivot.std(axis=1)
        
        print(f"\nSensitivity Disparity by Dataset (Baseline Model):")
        for dataset, std_val in dataset_std.items():
            print(f"  {dataset}: {std_val:.2f}% std deviation")
    else:
        print(f"\nNo dataset comparison data found for Baseline model")

if __name__ == "__main__":
    main()