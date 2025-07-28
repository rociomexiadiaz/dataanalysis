import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import re

# Add the parent directory to the Python path to import dataframe module
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from dataframe import parse_combined_log

# Configuration
log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Overall"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def parse_additional_metrics(filepath):
    """Parse additional metrics that may not be in the standard dataframe parser"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split models by 'Python Filename:'
    model_blocks = re.split(r'(?=Python Filename:)', content)
    
    balanced_accuracies = []
    f1_scores = []
    stratified_balanced_accuracies = []
    stratified_f1_scores = []
    
    for block in model_blocks:
        if not block.strip():
            continue
            
        model_match = re.search(r'Python Filename:\s*(\S+)', block)
        if not model_match:
            continue
        model = model_match.group(1).replace('.py', '').replace('train_', '')
        
        datasets_match = re.search(r'Datasets:\s*(.+)', block)
        if not datasets_match:
            continue
        datasets = datasets_match.group(1)
        
        # Parse Overall Balanced Accuracy
        balanced_acc_match = re.search(r'Overall Balanced Accuracy:\s*([\d.]+)%', block)
        if balanced_acc_match:
            balanced_accuracies.append({
                'Model': model,
                'Datasets': datasets,
                'Balanced_Accuracy': float(balanced_acc_match.group(1))
            })
        
        # Parse F1 Scores
        # Overall F1 Score (Macro)
        f1_macro_match = re.search(r'Overall F1 Score \(Macro\):\s*([\d.]+)%', block)
        if f1_macro_match:
            f1_scores.append({
                'Model': model,
                'Datasets': datasets,
                'Metric': 'Macro F1',
                'Value': float(f1_macro_match.group(1))
            })
            
        # Per Condition F1 Scores
        f1_condition_matches = re.findall(r'(\w+):\s*([\d.]+)%', 
                                         re.search(r'Per Condition F1 Score:\n(.*?)(?=\n===|\n\nNumber|$)', 
                                                  block, re.DOTALL).group(1) if re.search(r'Per Condition F1 Score:\n(.*?)(?=\n===|\n\nNumber|$)', block, re.DOTALL) else "")
        
        for condition, f1_score in f1_condition_matches:
            f1_scores.append({
                'Model': model,
                'Datasets': datasets,
                'Condition': condition,
                'Metric': 'Condition F1',
                'Value': float(f1_score)
            })
        
        # Parse Stratified Balanced Accuracy
        skin_tone_sections = re.findall(r'Skin Tone:\s*([\d.]+)\n\s*Overall Balanced Accuracy:\s*([\d.]+)%', block)
        for skin_tone, bal_acc in skin_tone_sections:
            stratified_balanced_accuracies.append({
                'Model': model,
                'Datasets': datasets,
                'Skin_Tone': float(skin_tone),
                'Balanced_Accuracy': float(bal_acc)
            })
        
        # Parse Stratified F1 Scores
        stratified_f1_pattern = r'Skin Tone:\s*([\d.]+)\n.*?Per Condition:\n(.*?)(?=\nSkin Tone:|\n\nNumber|\n===|$)'
        stratified_f1_matches = re.findall(stratified_f1_pattern, block, re.DOTALL)
        
        for skin_tone, conditions_text in stratified_f1_matches:
            condition_f1_matches = re.findall(r'(\w+):\s*([\d.]+)%', conditions_text)
            for condition, f1_score in condition_f1_matches:
                stratified_f1_scores.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Skin_Tone': float(skin_tone),
                    'Condition': condition,
                    'F1_Score': float(f1_score)
                })
    
    return {
        'BalancedAccuracies': pd.DataFrame(balanced_accuracies),
        'F1Scores': pd.DataFrame(f1_scores),
        'StratifiedBalancedAccuracies': pd.DataFrame(stratified_balanced_accuracies),
        'StratifiedF1Scores': pd.DataFrame(stratified_f1_scores)
    }

def clean_model_names(df, column='Model'):
    """Clean and standardize model names"""
    model_mapping = {
        'train_Baseline': 'Baseline',
        'Baseline': 'Baseline',
        'train_VAE': 'VAE',
        'VAE': 'VAE',
        'train_TABE': 'TABE',
        'TABE': 'TABE',
        'train_FairDisCo': 'FairDisCo',
        'FairDisCo': 'FairDisCo'
    }
    
    df[column] = df[column].replace(model_mapping)
    return df

# FST color mapping
fst_color_map = {
    1.0: '#F5D5A0',
    2.0: '#E4B589',
    3.0: '#D1A479',
    4.0: '#C0874F',
    5.0: '#A56635',
    6.0: '#4C2C27'
}

def create_sensitivity_heatmap(condition_sensitivities, save_path):
    """Create heatmap for sensitivity by model and condition"""
    
    # Filter for Top-1 sensitivity (include all models)
    sensitivity_data = condition_sensitivities[
        condition_sensitivities['Metric'] == 'Top-1 Sensitivity'
    ].copy()
    
    sensitivity_data = clean_model_names(sensitivity_data)
    
    # Create pivot table
    sensitivity_pivot = sensitivity_data.pivot_table(
        values='Value', 
        index='Model', 
        columns='Condition', 
        fill_value=0
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        sensitivity_pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        cbar_kws={'label': 'Sensitivity (%)'},
        vmin=0,
        vmax=100
    )
    
    plt.title('Model Performance: Sensitivity by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Skin Condition', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'sensitivity_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: sensitivity_heatmap.png")

def create_f1_heatmap(f1_scores, save_path):
    """Create heatmap for F1 scores by model and condition"""
    
    # Filter for condition F1 scores (include all models)
    f1_data = f1_scores[
        f1_scores['Metric'] == 'Condition F1'
    ].copy()
    
    f1_data = clean_model_names(f1_data)
    
    # Create pivot table
    f1_pivot = f1_data.pivot_table(
        values='Value',
        index='Model',
        columns='Condition',
        fill_value=0
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        f1_pivot,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        cbar_kws={'label': 'F1 Score (%)'},
        vmin=0,
        vmax=100
    )
    
    plt.title('Model Performance: F1 Score by Condition', fontsize=14, fontweight='bold')
    plt.xlabel('Skin Condition', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'f1_score_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: f1_score_heatmap.png")

def create_balanced_accuracy_bar_chart(balanced_accuracies, save_path):
    """Create bar chart for overall balanced accuracy by model"""
    
    # Include all models
    bal_acc_data = balanced_accuracies.copy()
    bal_acc_data = clean_model_names(bal_acc_data)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    
    models = bal_acc_data['Model'].tolist()
    accuracies = bal_acc_data['Balanced_Accuracy'].tolist()
    
    # Define colors for each model (now including FairDisCo)
    colors = ['#2E86C1', '#28B463', '#F39C12', '#E74C3C']  # Blue, Green, Orange, Red
    
    bars = plt.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Balanced Accuracy (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.title('Overall Balanced Accuracy by Model', fontsize=14, fontweight='bold')
    plt.ylim(0, max(accuracies) * 1.1)
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'balanced_accuracy_overall.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: balanced_accuracy_overall.png")

def create_balanced_accuracy_by_skin_tone(stratified_balanced_accuracies, save_path):
    """Create bar chart for balanced accuracy by skin tone, with models on x-axis and FST color-coded"""
    
    # Include all models
    strat_data = stratified_balanced_accuracies.copy()
    strat_data = clean_model_names(strat_data)
    
    # Create grouped bar chart
    plt.figure(figsize=(14, 8))
    
    models = sorted(strat_data['Model'].unique())
    skin_tones = sorted(strat_data['Skin_Tone'].unique())
    
    x = np.arange(len(models))
    width = 0.12  # Width of bars (adjusted for more FST categories)
    
    # Create bars for each FST
    for i, fst in enumerate(skin_tones):
        fst_data = strat_data[strat_data['Skin_Tone'] == fst]
        
        # Ensure we have data for all models
        accuracies = []
        for model in models:
            model_data = fst_data[fst_data['Model'] == model]
            if len(model_data) > 0:
                accuracies.append(model_data['Balanced_Accuracy'].iloc[0])
            else:
                accuracies.append(0)
        
        bars = plt.bar(x + i*width, accuracies, width, 
                      label=f'FST {int(fst)}', 
                      color=fst_color_map[fst], 
                      alpha=0.8, 
                      edgecolor='black', 
                      linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.0f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Balanced Accuracy (%)', fontsize=12)
    plt.title('Balanced Accuracy by Model and Fitzpatrick Skin Type', fontsize=14, fontweight='bold')
    
    # Set x-tick labels to model names
    plt.xticks(x + width * (len(skin_tones) - 1) / 2, models)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'balanced_accuracy_by_skin_tone.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: balanced_accuracy_by_skin_tone.png")

def main():
    """Main function to run overall analysis"""
    
    print("=== Overall Model Performance Analysis ===")
    print("Loading and parsing data from combined_all.txt...")
    
    # Load data from the "All Datasets" configuration
    log_path = os.path.join(log_directory, 'combined_all.txt')
    
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found!")
        return
    
    # Parse standard metrics
    df_dict = parse_combined_log(log_path)
    
    # Parse additional metrics
    additional_metrics = parse_additional_metrics(log_path)
    
    print(f"Loaded data for models: {df_dict['ConditionSensitivities']['Model'].unique()}")
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    
    # 1. Sensitivity Heatmap
    print("Creating sensitivity heatmap...")
    create_sensitivity_heatmap(df_dict['ConditionSensitivities'], output_directory)
    
    # 2. F1 Score Heatmap
    print("Creating F1 score heatmap...")
    create_f1_heatmap(additional_metrics['F1Scores'], output_directory)
    
    # 3. Overall Balanced Accuracy Bar Chart
    print("Creating overall balanced accuracy bar chart...")
    create_balanced_accuracy_bar_chart(additional_metrics['BalancedAccuracies'], output_directory)
    
    # 4. Balanced Accuracy by Skin Tone
    print("Creating balanced accuracy by skin tone chart...")
    create_balanced_accuracy_by_skin_tone(additional_metrics['StratifiedBalancedAccuracies'], output_directory)
    
    print(f"\n✅ All visualizations saved to: {output_directory}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    
    models = additional_metrics['BalancedAccuracies']['Model'].unique()
    models = clean_model_names(pd.DataFrame({'Model': models}))['Model'].unique()
    
    print("Models analyzed:", models)
    
    for model in models:
        model_balanced_acc = additional_metrics['BalancedAccuracies'][
            clean_model_names(additional_metrics['BalancedAccuracies'])['Model'] == model
        ]['Balanced_Accuracy'].iloc[0]
        
        print(f"{model}: {model_balanced_acc:.2f}% balanced accuracy")

if __name__ == "__main__":
    main()