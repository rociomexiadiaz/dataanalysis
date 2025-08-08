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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\DermieComparison"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def parse_dermie_experiment_metrics(filepath):
    """Parse Dermie experiment metrics from the combined log file"""
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Extract different experiment sections
    experiments = ['NO_DERMIE', 'WITH_DERMIE']
    models = ['BASELINE', 'FAIRDISCO', 'TABE', 'VAE']
    
    balanced_accuracies = []
    stratified_balanced_accuracies = []
    condition_sensitivities = []
    stratified_sensitivities = []
    
    # Split content by experiment sections
    for model in models:
        # Find the section for this model type
        model_pattern = rf'{model} DERMIE COMPARISON EXPERIMENT.*?(?={"|".join([m for m in models if m != model])} DERMIE COMPARISON EXPERIMENT|$)'
        model_section = re.search(model_pattern, content, re.DOTALL)
        
        if not model_section:
            continue
            
        model_content = model_section.group(0)
        
        # Extract NO_DERMIE and WITH_DERMIE experiments
        for experiment in experiments:
            exp_pattern = rf'Experiment: {experiment}.*?(?=Experiment: |================================================================================|$)'
            exp_match = re.search(exp_pattern, model_content, re.DOTALL)
            
            if not exp_match:
                continue
                
            exp_block = exp_match.group(0)
            model_name = model.replace('BASELINE', 'Baseline').replace('FAIRDISCO', 'FairDisCo')
            
            # Parse Overall Balanced Accuracy
            balanced_acc_match = re.search(r'Overall Balanced Accuracy:\s*([\d.]+)%', exp_block)
            if balanced_acc_match:
                balanced_accuracies.append({
                    'Model': model_name,
                    'Experiment': experiment,
                    'Balanced_Accuracy': float(balanced_acc_match.group(1))
                })
            
            # Parse Stratified Balanced Accuracy by Skin Tone
            skin_tone_ba_matches = re.findall(r'Skin Tone:\s*([\d.]+)\s+Overall Balanced Accuracy:\s*([\d.]+)%', exp_block)
            for skin_tone, ba_value in skin_tone_ba_matches:
                stratified_balanced_accuracies.append({
                    'Model': model_name,
                    'Experiment': experiment,
                    'Skin_Tone': float(skin_tone),
                    'Balanced_Accuracy': float(ba_value)
                })
            
            # Parse Overall Condition Sensitivities
            condition_sens_matches = re.findall(r'Condition:\s*(\w+),\s*Top-1 Sensitivity:\s*([\d.]+)%', exp_block)
            for condition, sensitivity in condition_sens_matches:
                condition_sensitivities.append({
                    'Model': model_name,
                    'Experiment': experiment,
                    'Condition': condition,
                    'Metric': 'Top-1 Sensitivity',
                    'Value': float(sensitivity)
                })
            
            # Parse Stratified Sensitivities by Skin Tone and Condition
            skin_tone_sections = re.findall(
                r'Skin Tone:\s*([\d.]+)\s+((?:\s+Condition:.*?Top-1 Sensitivity:.*?%.*?\n?)+)', 
                exp_block
            )
            
            for skin_tone, conditions_text in skin_tone_sections:
                condition_matches = re.findall(
                    r'Condition:\s*(\w+),\s*Top-1 Sensitivity:\s*([\d.]+)%', 
                    conditions_text
                )
                
                for condition, sensitivity in condition_matches:
                    stratified_sensitivities.append({
                        'Model': model_name,
                        'Experiment': experiment,
                        'Skin_Tone': float(skin_tone),
                        'Condition': condition,
                        'Metric': 'Top-1 Sensitivity',
                        'Value': float(sensitivity)
                    })
    
    return {
        'BalancedAccuracies': pd.DataFrame(balanced_accuracies),
        'StratifiedBalancedAccuracies': pd.DataFrame(stratified_balanced_accuracies),
        'ConditionSensitivities': pd.DataFrame(condition_sensitivities),
        'StratifiedSensitivities': pd.DataFrame(stratified_sensitivities)
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

def create_overall_balanced_accuracy_comparison(balanced_accuracies, save_path):
    """Create comparison of overall balanced accuracy with/without Dermie"""
    
    bal_acc_data = balanced_accuracies.copy()
    bal_acc_data = clean_model_names(bal_acc_data)
    
    models = sorted(bal_acc_data['Model'].unique())
    experiments = ['NO_DERMIE', 'WITH_DERMIE']
    
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(models))
    width = 0.35
    
    no_dermie_values = []
    with_dermie_values = []
    
    for model in models:
        no_dermie_data = bal_acc_data[(bal_acc_data['Model'] == model) & (bal_acc_data['Experiment'] == 'NO_DERMIE')]
        with_dermie_data = bal_acc_data[(bal_acc_data['Model'] == model) & (bal_acc_data['Experiment'] == 'WITH_DERMIE')]
        
        no_dermie_val = no_dermie_data['Balanced_Accuracy'].iloc[0] if len(no_dermie_data) > 0 else 0
        with_dermie_val = with_dermie_data['Balanced_Accuracy'].iloc[0] if len(with_dermie_data) > 0 else 0
        
        no_dermie_values.append(no_dermie_val)
        with_dermie_values.append(with_dermie_val)
    
    bars1 = plt.bar(x - width/2, no_dermie_values, width, label='No Dermie', 
                    color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = plt.bar(x + width/2, with_dermie_values, width, label='With Dermie', 
                    color='#28B463', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars1, no_dermie_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for bar, val in zip(bars2, with_dermie_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Overall Balanced Accuracy (%)', fontsize=12)
    plt.title('Effect of Dermie Dataset on Overall Balanced Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.ylim(0, max(max(no_dermie_values), max(with_dermie_values)) * 1.15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dermie_overall_balanced_accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dermie_overall_balanced_accuracy_comparison.png")

def create_balanced_accuracy_by_skin_tone_comparison(stratified_balanced_accuracies, save_path):
    """Create comparison of balanced accuracy by skin tone with/without Dermie"""
    
    strat_data = stratified_balanced_accuracies.copy()
    strat_data = clean_model_names(strat_data)
    
    models = sorted(strat_data['Model'].unique())
    skin_tones = sorted(strat_data['Skin_Tone'].unique())
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        if i >= len(axes):
            break
            
        ax = axes[i]
        model_data = strat_data[strat_data['Model'] == model]
        
        x = np.arange(len(skin_tones))
        width = 0.35
        
        no_dermie_values = []
        with_dermie_values = []
        
        for st in skin_tones:
            no_dermie_data = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'NO_DERMIE')]
            with_dermie_data = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'WITH_DERMIE')]
            
            no_dermie_val = no_dermie_data['Balanced_Accuracy'].iloc[0] if len(no_dermie_data) > 0 else 0
            with_dermie_val = with_dermie_data['Balanced_Accuracy'].iloc[0] if len(with_dermie_data) > 0 else 0
            
            no_dermie_values.append(no_dermie_val)
            with_dermie_values.append(with_dermie_val)
        
        bars1 = ax.bar(x - width/2, no_dermie_values, width, label='No Dermie', 
                      color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, with_dermie_values, width, label='With Dermie', 
                      color='#28B463', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars1, no_dermie_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        for bar, val in zip(bars2, with_dermie_values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Fitzpatrick Skin Type', fontsize=10)
        ax.set_ylabel('Balanced Accuracy (%)', fontsize=10)
        ax.set_title(f'{model} - Balanced Accuracy by FST', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'FST {int(st)}' for st in skin_tones])
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, 100)
    
    # Hide unused subplot
    for i in range(len(models), len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Effect of Dermie Dataset on Balanced Accuracy by Skin Tone', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dermie_balanced_accuracy_by_skin_tone.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dermie_balanced_accuracy_by_skin_tone.png")

def create_condition_sensitivity_comparison(condition_sensitivities, save_path):
    """Create comparison of condition sensitivity with/without Dermie"""
    
    sens_data = condition_sensitivities.copy()
    sens_data = clean_model_names(sens_data)
    
    models = sorted(sens_data['Model'].unique())
    conditions = sorted(sens_data['Condition'].unique())
    
    # Create overall condition sensitivity comparison
    plt.figure(figsize=(12, 7))
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, condition in enumerate(conditions):
        plt.subplot(1, len(conditions), i+1)
        
        no_dermie_values = []
        with_dermie_values = []
        
        for model in models:
            model_cond_data = sens_data[(sens_data['Model'] == model) & (sens_data['Condition'] == condition)]
            
            no_dermie_data = model_cond_data[model_cond_data['Experiment'] == 'NO_DERMIE']
            with_dermie_data = model_cond_data[model_cond_data['Experiment'] == 'WITH_DERMIE']
            
            no_dermie_val = no_dermie_data['Value'].iloc[0] if len(no_dermie_data) > 0 else 0
            with_dermie_val = with_dermie_data['Value'].iloc[0] if len(with_dermie_data) > 0 else 0
            
            no_dermie_values.append(no_dermie_val)
            with_dermie_values.append(with_dermie_val)
        
        x_pos = np.arange(len(models))
        bars1 = plt.bar(x_pos - width/2, no_dermie_values, width, label='No Dermie', 
                       color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = plt.bar(x_pos + width/2, with_dermie_values, width, label='With Dermie', 
                       color='#28B463', alpha=0.8, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, val in zip(bars1, no_dermie_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        for bar, val in zip(bars2, with_dermie_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.xlabel('Model', fontsize=10)
        plt.ylabel('Top-1 Sensitivity (%)', fontsize=10)
        plt.title(f'{condition.capitalize()} Sensitivity', fontsize=12, fontweight='bold')
        plt.xticks(x_pos, models, rotation=45, ha='right')
        if i == 0:
            plt.legend()
        plt.grid(True, axis='y', alpha=0.3)
        plt.ylim(0, 100)
    
    plt.suptitle('Effect of Dermie Dataset on Condition Sensitivity', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dermie_condition_sensitivity_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dermie_condition_sensitivity_comparison.png")

def create_stratified_sensitivity_plots(stratified_sensitivities, save_path):
    """Create stratified sensitivity plots by condition and skin tone"""
    
    strat_sens_data = stratified_sensitivities.copy()
    strat_sens_data = clean_model_names(strat_sens_data)
    
    conditions = sorted(strat_sens_data['Condition'].unique())
    
    for condition in conditions:
        cond_data = strat_sens_data[strat_sens_data['Condition'] == condition]
        models = sorted(cond_data['Model'].unique())
        skin_tones = sorted(cond_data['Skin_Tone'].unique())
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, model in enumerate(models):
            if i >= len(axes):
                break
                
            ax = axes[i]
            model_data = cond_data[cond_data['Model'] == model]
            
            x = np.arange(len(skin_tones))
            width = 0.35
            
            no_dermie_values = []
            with_dermie_values = []
            
            for st in skin_tones:
                no_dermie_data = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'NO_DERMIE')]
                with_dermie_data = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'WITH_DERMIE')]
                
                no_dermie_val = no_dermie_data['Value'].iloc[0] if len(no_dermie_data) > 0 else 0
                with_dermie_val = with_dermie_data['Value'].iloc[0] if len(with_dermie_data) > 0 else 0
                
                no_dermie_values.append(no_dermie_val)
                with_dermie_values.append(with_dermie_val)
            
            bars1 = ax.bar(x - width/2, no_dermie_values, width, label='No Dermie', 
                          color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=0.5)
            bars2 = ax.bar(x + width/2, with_dermie_values, width, label='With Dermie', 
                          color='#28B463', alpha=0.8, edgecolor='black', linewidth=0.5)
            
            # Add value labels
            for bar, val in zip(bars1, no_dermie_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            for bar, val in zip(bars2, with_dermie_values):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{val:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Fitzpatrick Skin Type', fontsize=10)
            ax.set_ylabel('Sensitivity (%)', fontsize=10)
            ax.set_title(f'{model} - {condition.capitalize()} Sensitivity by FST', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'FST {int(st)}' for st in skin_tones])
            ax.legend()
            ax.grid(True, axis='y', alpha=0.3)
            ax.set_ylim(0, 100)
        
        # Hide unused subplots
        for i in range(len(models), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle(f'Effect of Dermie Dataset on {condition.capitalize()} Sensitivity by Skin Tone', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'dermie_{condition}_sensitivity_by_skin_tone.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: dermie_{condition}_sensitivity_by_skin_tone.png")

def create_dermie_effect_heatmap(stratified_balanced_accuracies, stratified_sensitivities, save_path):
    """Create heatmap showing the effect of Dermie (difference in performance)"""
    
    # Calculate effect for balanced accuracy
    strat_ba_data = stratified_balanced_accuracies.copy()
    strat_ba_data = clean_model_names(strat_ba_data)
    
    models = sorted(strat_ba_data['Model'].unique())
    skin_tones = sorted(strat_ba_data['Skin_Tone'].unique())
    
    # Balanced accuracy effects
    ba_effects = []
    for model in models:
        model_effects = []
        model_data = strat_ba_data[strat_ba_data['Model'] == model]
        
        for st in skin_tones:
            no_dermie_val = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'NO_DERMIE')]['Balanced_Accuracy']
            with_dermie_val = model_data[(model_data['Skin_Tone'] == st) & (model_data['Experiment'] == 'WITH_DERMIE')]['Balanced_Accuracy']
            
            if len(no_dermie_val) > 0 and len(with_dermie_val) > 0:
                effect = with_dermie_val.iloc[0] - no_dermie_val.iloc[0]
            else:
                effect = 0
            
            model_effects.append(effect)
        ba_effects.append(model_effects)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(ba_effects, cmap='RdBu_r', aspect='auto', vmin=-20, vmax=20)
    
    plt.title('Dermie Dataset Effect on Balanced Accuracy\n(With Dermie - No Dermie)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Fitzpatrick Skin Type', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(range(len(skin_tones)), [f'FST {int(st)}' for st in skin_tones])
    plt.yticks(range(len(models)), models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(skin_tones)):
            text = plt.text(j, i, f'{ba_effects[i][j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, shrink=0.8)
    cbar.set_label('Balanced Accuracy Difference (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'dermie_effect_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: dermie_effect_heatmap.png")

def main():
    """Main function to run Dermie comparison analysis"""
    
    print("=== Dermie Dataset Effect Analysis ===")
    print("Loading and parsing data from combined_all.txt...")
    
    # Load data
    log_path = os.path.join(log_directory, 'combined_experiment.txt')
    
    if not os.path.exists(log_path):
        print(f"ERROR: {log_path} not found!")
        return
    
    # Parse Dermie experiment metrics
    dermie_metrics = parse_dermie_experiment_metrics(log_path)
    
    print(f"Loaded Dermie comparison data for models: {dermie_metrics['BalancedAccuracies']['Model'].unique()}")
    
    # Create visualizations
    print("\n=== Creating Dermie Comparison Visualizations ===")
    
    # 1. Overall Balanced Accuracy Comparison
    print("Creating overall balanced accuracy comparison...")
    create_overall_balanced_accuracy_comparison(dermie_metrics['BalancedAccuracies'], output_directory)
    
    # 2. Balanced Accuracy by Skin Tone Comparison
    print("Creating balanced accuracy by skin tone comparison...")
    create_balanced_accuracy_by_skin_tone_comparison(dermie_metrics['StratifiedBalancedAccuracies'], output_directory)
    
    # 3. Condition Sensitivity Comparison
    print("Creating condition sensitivity comparison...")
    create_condition_sensitivity_comparison(dermie_metrics['ConditionSensitivities'], output_directory)
    
    # 4. Stratified Sensitivity Plots
    print("Creating stratified sensitivity plots...")
    create_stratified_sensitivity_plots(dermie_metrics['StratifiedSensitivities'], output_directory)
    
    # 5. Dermie Effect Heatmap
    print("Creating Dermie effect heatmap...")
    create_dermie_effect_heatmap(dermie_metrics['StratifiedBalancedAccuracies'], 
                                dermie_metrics['StratifiedSensitivities'], output_directory)
    
    print(f"\n✅ All Dermie comparison visualizations saved to: {output_directory}")
    
    # Print summary statistics
    print("\n=== Dermie Effect Summary Statistics ===")
    
    models = dermie_metrics['BalancedAccuracies']['Model'].unique()
    
    for model in models:
        model_data = dermie_metrics['BalancedAccuracies'][dermie_metrics['BalancedAccuracies']['Model'] == model]
        
        no_dermie_ba = model_data[model_data['Experiment'] == 'NO_DERMIE']['Balanced_Accuracy'].iloc[0]
        with_dermie_ba = model_data[model_data['Experiment'] == 'WITH_DERMIE']['Balanced_Accuracy'].iloc[0]
        
        effect = with_dermie_ba - no_dermie_ba
        
        print(f"{model}:")
        print(f"  No Dermie: {no_dermie_ba:.2f}%")
        print(f"  With Dermie: {with_dermie_ba:.2f}%")
        print(f"  Effect: {effect:+.2f}%")

if __name__ == "__main__":
    main()