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
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Generalisation"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

def parse_generalization_results(filepath):
    """
    Parse generalization experiment results for balanced accuracy, Top-3 and Top-5 sensitivity
    """
    with open(filepath, 'r') as file:
        content = file.read()
    
    balanced_accuracy_results = []
    top3_sensitivity_results = []
    top5_sensitivity_results = []
    
    # Split by experiment sections (Python Filename)
    sections = re.split(r'Python Filename:\s*([^\r\n]+)', content)[1:]  # Skip first empty element
    
    # Process pairs of (filename, content)
    for i in range(0, len(sections), 2):
        if i + 1 < len(sections):
            filename = sections[i].strip()
            section_content = sections[i + 1]
            
            # Extract model name
            model_match = re.search(r'train_(\w+)_generalisation\.py', filename)
            if not model_match:
                continue
            model_name = model_match.group(1)
            
            print(f"Processing model: {model_name}")
            
            # Split by fold sections
            fold_sections = re.split(r'=== Fold: (\w+) ===', section_content)[1:]
            
            # Process pairs of (fold_name, fold_content)
            for j in range(0, len(fold_sections), 2):
                if j + 1 < len(fold_sections):
                    fold_name = fold_sections[j]
                    fold_content = fold_sections[j + 1]
                    
                    # Extract stratified balanced accuracy
                    strat_bal_acc_match = re.search(
                        r'=== STRATIFIED BALANCED ACCURACY ===(.*?)(?==== |$)', 
                        fold_content, 
                        re.DOTALL
                    )
                    
                    if strat_bal_acc_match:
                        strat_section = strat_bal_acc_match.group(1)
                        
                        # Extract skin tone balanced accuracies
                        skin_tone_matches = re.findall(
                            r'Skin Tone: ([\d.]+)\s+Overall Balanced Accuracy: ([\d.]+)%', 
                            strat_section
                        )
                        
                        for skin_tone, accuracy in skin_tone_matches:
                            balanced_accuracy_results.append({
                                'Model': model_name,
                                'Test_Dataset': fold_name,
                                'Skin_Tone': float(skin_tone),
                                'Balanced_Accuracy': float(accuracy)
                            })
                    
                    # Extract stratified Top-3 sensitivity
                    strat_top3_sens_match = re.search(
                        r'=== STRATIFIED TOP-3 SENSITIVITY ===(.*?)(?==== |$)', 
                        fold_content, 
                        re.DOTALL
                    )
                    
                    if strat_top3_sens_match:
                        sens_section = strat_top3_sens_match.group(1)
                        
                        # Parse by skin tone blocks
                        skin_tone_blocks = re.split(r'Skin Tone: ([\d.]+)', sens_section)[1:]
                        
                        # Process pairs of (skin_tone, block_content)
                        for k in range(0, len(skin_tone_blocks), 2):
                            if k + 1 < len(skin_tone_blocks):
                                skin_tone = float(skin_tone_blocks[k])
                                block_content = skin_tone_blocks[k + 1]
                                
                                # Extract condition sensitivities
                                condition_matches = re.findall(
                                    r'Condition: (.+?), Top-3 Sensitivity: ([\d.]+)%', 
                                    block_content
                                )
                                
                                if condition_matches:
                                    # Calculate mean sensitivity across conditions for this skin tone
                                    sensitivities = [float(sens) for _, sens in condition_matches]
                                    mean_sensitivity = np.mean(sensitivities)
                                    
                                    top3_sensitivity_results.append({
                                        'Model': model_name,
                                        'Test_Dataset': fold_name,
                                        'Skin_Tone': skin_tone,
                                        'Mean_Top3_Sensitivity': mean_sensitivity
                                    })
                    
                    # Extract stratified Top-5 sensitivity
                    strat_top5_sens_match = re.search(
                        r'=== STRATIFIED TOP-5 SENSITIVITY ===(.*?)(?==== |$)', 
                        fold_content, 
                        re.DOTALL
                    )
                    
                    if strat_top5_sens_match:
                        sens_section = strat_top5_sens_match.group(1)
                        
                        # Parse by skin tone blocks
                        skin_tone_blocks = re.split(r'Skin Tone: ([\d.]+)', sens_section)[1:]
                        
                        # Process pairs of (skin_tone, block_content)
                        for k in range(0, len(skin_tone_blocks), 2):
                            if k + 1 < len(skin_tone_blocks):
                                skin_tone = float(skin_tone_blocks[k])
                                block_content = skin_tone_blocks[k + 1]
                                
                                # Extract condition sensitivities
                                condition_matches = re.findall(
                                    r'Condition: (.+?), Top-5 Sensitivity: ([\d.]+)%', 
                                    block_content
                                )
                                
                                if condition_matches:
                                    # Calculate mean sensitivity across conditions for this skin tone
                                    sensitivities = [float(sens) for _, sens in condition_matches]
                                    mean_sensitivity = np.mean(sensitivities)
                                    
                                    top5_sensitivity_results.append({
                                        'Model': model_name,
                                        'Test_Dataset': fold_name,
                                        'Skin_Tone': skin_tone,
                                        'Mean_Top5_Sensitivity': mean_sensitivity
                                    })
    
    return (pd.DataFrame(balanced_accuracy_results), 
            pd.DataFrame(top3_sensitivity_results), 
            pd.DataFrame(top5_sensitivity_results))

def plot_generalization_performance(balanced_acc_df, top3_sensitivity_df, top5_sensitivity_df, save_path):
    """
    Create three separate bar charts showing balanced accuracy, Top-3 sensitivity, and Top-5 sensitivity by model and skin tone
    """
    
    # Define skin tone colors (from light to dark)
    skin_tone_colors = {
        1.0: '#F5D5A0',
        2.0: '#E4B589',
        3.0: '#D1A479',
        4.0: '#C0874F',
        5.0: '#A56635',
        6.0: '#4C2C27'
    }
    # ===== PLOT 1: BALANCED ACCURACY =====
    
    if not balanced_acc_df.empty:
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate mean and std for balanced accuracy
        bal_acc_stats = balanced_acc_df.groupby(['Model', 'Skin_Tone'])['Balanced_Accuracy'].agg(['mean', 'std', 'count']).reset_index()
        bal_acc_stats['std'] = bal_acc_stats['std'].fillna(0)
        
        print("Balanced Accuracy Statistics:")
        print(bal_acc_stats)
        
        # Prepare data for plotting
        models = sorted(bal_acc_stats['Model'].unique())
        skin_tones = sorted(bal_acc_stats['Skin_Tone'].unique())
        
        x = np.arange(len(models))
        width = 0.13  # Width of bars
        
        # Plot balanced accuracy bars for each skin tone
        for i, skin_tone in enumerate(skin_tones):
            skin_tone_data = bal_acc_stats[bal_acc_stats['Skin_Tone'] == skin_tone]
            
            # Ensure we have data for all models (fill missing with 0)
            means = []
            stds = []
            
            for model in models:
                model_data = skin_tone_data[skin_tone_data['Model'] == model]
                if not model_data.empty:
                    means.append(model_data['mean'].iloc[0])
                    stds.append(model_data['std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Plot bars with error bars
            bars = ax1.bar(
                x + i * width, 
                means, 
                width, 
                yerr=stds,
                capsize=5,
                label=f'Skin Tone {int(skin_tone)}',
                color=skin_tone_colors[skin_tone],
                alpha=0.8,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Add value labels on bars (only for non-zero values)
            for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                if mean_val > 2:  # Only label bars with reasonable values
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 1,
                           f'{mean_val:.1f}%',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize balanced accuracy plot
        ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Balanced Accuracy (%)', fontsize=12, fontweight='bold')
        ax1.set_title('Cross-Dataset Generalization: Balanced Accuracy by Skin Tone', 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax1.set_xticklabels(models, fontsize=11)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        if not bal_acc_stats.empty:
            ax1.set_ylim(0, max(bal_acc_stats['mean'] + bal_acc_stats['std']) * 1.15)
        
        # Save balanced accuracy plot
        output_path1 = os.path.join(save_path, 'balanced_accuracy_by_skin_tone.png')
        plt.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Balanced Accuracy plot saved to: {output_path1}")
    else:
        print("No Balanced Accuracy Data Available")
    
    # ===== PLOT 2: TOP-3 SENSITIVITY =====
    
    if not top3_sensitivity_df.empty:
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate mean and std for Top-3 sensitivity
        top3_sens_stats = top3_sensitivity_df.groupby(['Model', 'Skin_Tone'])['Mean_Top3_Sensitivity'].agg(['mean', 'std', 'count']).reset_index()
        top3_sens_stats['std'] = top3_sens_stats['std'].fillna(0)
        
        print("\nTop-3 Sensitivity Statistics:")
        print(top3_sens_stats)
        
        # Prepare data for plotting
        models = sorted(top3_sens_stats['Model'].unique())
        skin_tones = sorted(top3_sens_stats['Skin_Tone'].unique())
        
        x = np.arange(len(models))
        width = 0.13  # Width of bars
        
        # Plot Top-3 sensitivity bars for each skin tone
        for i, skin_tone in enumerate(skin_tones):
            skin_tone_data = top3_sens_stats[top3_sens_stats['Skin_Tone'] == skin_tone]
            
            # Ensure we have data for all models (fill missing with 0)
            means = []
            stds = []
            
            for model in models:
                model_data = skin_tone_data[skin_tone_data['Model'] == model]
                if not model_data.empty:
                    means.append(model_data['mean'].iloc[0])
                    stds.append(model_data['std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Plot bars with error bars
            bars = ax2.bar(
                x + i * width, 
                means, 
                width, 
                yerr=stds,
                capsize=5,
                label=f'Skin Tone {int(skin_tone)}',
                color=skin_tone_colors[skin_tone],
                alpha=0.8,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Add value labels on bars
            for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                if mean_val > 5:  # Only label bars with reasonable values
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                           f'{mean_val:.1f}%',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize Top-3 sensitivity plot
        ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Top-3 Sensitivity (%)', fontsize=12, fontweight='bold')
        ax2.set_title('Cross-Dataset Generalization: Mean Top-3 Sensitivity by Skin Tone', 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax2.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax2.set_xticklabels(models, fontsize=11)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        ax2.set_ylim(0, max(top3_sens_stats['mean'] + top3_sens_stats['std']) * 1.15)
        
        # Save Top-3 sensitivity plot
        output_path2 = os.path.join(save_path, 'top3_sensitivity_by_skin_tone.png')
        plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Top-3 Sensitivity plot saved to: {output_path2}")
    else:
        print("No Top-3 Sensitivity Data Available")
    
    # ===== PLOT 3: TOP-5 SENSITIVITY =====
    
    if not top5_sensitivity_df.empty:
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 8))
        
        # Calculate mean and std for Top-5 sensitivity
        top5_sens_stats = top5_sensitivity_df.groupby(['Model', 'Skin_Tone'])['Mean_Top5_Sensitivity'].agg(['mean', 'std', 'count']).reset_index()
        top5_sens_stats['std'] = top5_sens_stats['std'].fillna(0)
        
        print("\nTop-5 Sensitivity Statistics:")
        print(top5_sens_stats)
        
        # Prepare data for plotting
        models = sorted(top5_sens_stats['Model'].unique())
        skin_tones = sorted(top5_sens_stats['Skin_Tone'].unique())
        
        x = np.arange(len(models))
        width = 0.13  # Width of bars
        
        # Plot Top-5 sensitivity bars for each skin tone
        for i, skin_tone in enumerate(skin_tones):
            skin_tone_data = top5_sens_stats[top5_sens_stats['Skin_Tone'] == skin_tone]
            
            # Ensure we have data for all models (fill missing with 0)
            means = []
            stds = []
            
            for model in models:
                model_data = skin_tone_data[skin_tone_data['Model'] == model]
                if not model_data.empty:
                    means.append(model_data['mean'].iloc[0])
                    stds.append(model_data['std'].iloc[0])
                else:
                    means.append(0)
                    stds.append(0)
            
            # Plot bars with error bars
            bars = ax3.bar(
                x + i * width, 
                means, 
                width, 
                yerr=stds,
                capsize=5,
                label=f'Skin Tone {int(skin_tone)}',
                color=skin_tone_colors[skin_tone],
                alpha=0.8,
                edgecolor='white',
                linewidth=1.5
            )
            
            # Add value labels on bars
            for j, (bar, mean_val, std_val) in enumerate(zip(bars, means, stds)):
                if mean_val > 5:  # Only label bars with reasonable values
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + std_val + 2,
                           f'{mean_val:.1f}%',
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Customize Top-5 sensitivity plot
        ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Mean Top-5 Sensitivity (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Cross-Dataset Generalization: Mean Top-5 Sensitivity by Skin Tone', 
                    fontsize=14, fontweight='bold', pad=15)
        
        ax3.set_xticks(x + width * (len(skin_tones) - 1) / 2)
        ax3.set_xticklabels(models, fontsize=11)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.set_axisbelow(True)
        
        ax3.set_ylim(0, max(top5_sens_stats['mean'] + top5_sens_stats['std']) * 1.15)
        
        # Save Top-5 sensitivity plot
        output_path3 = os.path.join(save_path, 'top5_sensitivity_by_skin_tone.png')
        plt.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Top-5 Sensitivity plot saved to: {output_path3}")
    else:
        print("No Top-5 Sensitivity Data Available")
    
    print(f"\nAll plots saved to: {save_path}")

def analyze_performance_patterns(balanced_acc_df, top3_sensitivity_df, top5_sensitivity_df):
    """
    Analyze and print performance patterns for all metrics
    """
    print("\n" + "="*70)
    print("GENERALIZATION PERFORMANCE ANALYSIS")
    print("="*70)
    
    # ===== BALANCED ACCURACY ANALYSIS =====
    print(f"\n" + "="*50)
    print("BALANCED ACCURACY ANALYSIS")
    print("="*50)
    
    # Overall model performance for balanced accuracy
    model_performance_bal = balanced_acc_df.groupby('Model')['Balanced_Accuracy'].agg(['mean', 'std']).round(2)
    print(f"\nOverall Average Balanced Accuracy (across all skin tones and datasets):")
    for model, stats in model_performance_bal.iterrows():
        print(f"  {model}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
    
    # Best performing model for balanced accuracy
    best_model_bal = model_performance_bal['mean'].idxmax()
    print(f"\nBest performing model (Balanced Accuracy): {best_model_bal} ({model_performance_bal.loc[best_model_bal, 'mean']:.2f}%)")
    
    # Skin tone analysis for balanced accuracy
    print(f"\n" + "-"*40)
    print("BALANCED ACCURACY BY SKIN TONE")
    print("-"*40)
    
    for model in balanced_acc_df['Model'].unique():
        model_data = balanced_acc_df[balanced_acc_df['Model'] == model]
        skin_tone_stats = model_data.groupby('Skin_Tone')['Balanced_Accuracy'].agg(['mean', 'std']).round(2)
        
        print(f"\n{model}:")
        for skin_tone, stats in skin_tone_stats.iterrows():
            print(f"  Skin Tone {int(skin_tone)}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
        
        # Calculate fairness metrics
        means = skin_tone_stats['mean']
        max_diff = means.max() - means.min()
        cv = (means.std() / means.mean()) * 100 if means.mean() > 0 else 0
        
        print(f"  Performance Range: {max_diff:.2f}%")
        print(f"  Coefficient of Variation: {cv:.2f}% (lower = more equitable)")
    
    # ===== TOP-3 SENSITIVITY ANALYSIS =====
    if not top3_sensitivity_df.empty:
        print(f"\n" + "="*50)
        print("TOP-3 SENSITIVITY ANALYSIS") 
        print("="*50)
        
        # Overall model performance for sensitivity
        model_performance_sens = top3_sensitivity_df.groupby('Model')['Mean_Top3_Sensitivity'].agg(['mean', 'std']).round(2)
        print(f"\nOverall Average Top-3 Sensitivity (across all skin tones and datasets):")
        for model, stats in model_performance_sens.iterrows():
            print(f"  {model}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
        
        # Best performing model for sensitivity
        best_model_sens = model_performance_sens['mean'].idxmax()
        print(f"\nBest performing model (Top-3 Sensitivity): {best_model_sens} ({model_performance_sens.loc[best_model_sens, 'mean']:.2f}%)")
        
        # Skin tone analysis for sensitivity
        print(f"\n" + "-"*40)
        print("TOP-3 SENSITIVITY BY SKIN TONE")
        print("-"*40)
        
        for model in top3_sensitivity_df['Model'].unique():
            model_data = top3_sensitivity_df[top3_sensitivity_df['Model'] == model]
            skin_tone_stats = model_data.groupby('Skin_Tone')['Mean_Top3_Sensitivity'].agg(['mean', 'std']).round(2)
            
            print(f"\n{model}:")
            for skin_tone, stats in skin_tone_stats.iterrows():
                print(f"  Skin Tone {int(skin_tone)}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
            
            # Calculate fairness metrics
            means = skin_tone_stats['mean']
            max_diff = means.max() - means.min()
            cv = (means.std() / means.mean()) * 100 if means.mean() > 0 else 0
            
            print(f"  Performance Range: {max_diff:.2f}%")
            print(f"  Coefficient of Variation: {cv:.2f}% (lower = more equitable)")
    
    # ===== TOP-5 SENSITIVITY ANALYSIS =====
    if not top5_sensitivity_df.empty:
        print(f"\n" + "="*50)
        print("TOP-5 SENSITIVITY ANALYSIS") 
        print("="*50)
        
        # Overall model performance for Top-5 sensitivity
        model_performance_top5 = top5_sensitivity_df.groupby('Model')['Mean_Top5_Sensitivity'].agg(['mean', 'std']).round(2)
        print(f"\nOverall Average Top-5 Sensitivity (across all skin tones and datasets):")
        for model, stats in model_performance_top5.iterrows():
            print(f"  {model}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
        
        # Best performing model for Top-5 sensitivity
        best_model_top5 = model_performance_top5['mean'].idxmax()
        print(f"\nBest performing model (Top-5 Sensitivity): {best_model_top5} ({model_performance_top5.loc[best_model_top5, 'mean']:.2f}%)")
        
        # Skin tone analysis for Top-5 sensitivity
        print(f"\n" + "-"*40)
        print("TOP-5 SENSITIVITY BY SKIN TONE")
        print("-"*40)
        
        for model in top5_sensitivity_df['Model'].unique():
            model_data = top5_sensitivity_df[top5_sensitivity_df['Model'] == model]
            skin_tone_stats = model_data.groupby('Skin_Tone')['Mean_Top5_Sensitivity'].agg(['mean', 'std']).round(2)
            
            print(f"\n{model}:")
            for skin_tone, stats in skin_tone_stats.iterrows():
                print(f"  Skin Tone {int(skin_tone)}: {stats['mean']:.2f}% (±{stats['std']:.2f}%)")
            
            # Calculate fairness metrics
            means = skin_tone_stats['mean']
            max_diff = means.max() - means.min()
            cv = (means.std() / means.mean()) * 100 if means.mean() > 0 else 0
            
            print(f"  Performance Range: {max_diff:.2f}%")
            print(f"  Coefficient of Variation: {cv:.2f}% (lower = more equitable)")
    
    # ===== CROSS-DATASET CONSISTENCY =====
    print(f"\n" + "="*50)
    print("CROSS-DATASET CONSISTENCY")
    print("="*50)
    
    print(f"\nBalanced Accuracy - Cross-dataset consistency:")
    for model in balanced_acc_df['Model'].unique():
        model_data = balanced_acc_df[balanced_acc_df['Model'] == model]
        dataset_stats = model_data.groupby('Test_Dataset')['Balanced_Accuracy'].mean()
        
        print(f"\n{model} - Balanced Accuracy by Test Dataset:")
        for dataset, perf in dataset_stats.items():
            print(f"  {dataset}: {perf:.2f}%")
        
        dataset_std = dataset_stats.std()
        print(f"  Cross-dataset std: {dataset_std:.2f}%")
    
    if not top3_sensitivity_df.empty:
        print(f"\nTop-3 Sensitivity - Cross-dataset consistency:")
        for model in top3_sensitivity_df['Model'].unique():
            model_data = top3_sensitivity_df[top3_sensitivity_df['Model'] == model]
            dataset_stats = model_data.groupby('Test_Dataset')['Mean_Top3_Sensitivity'].mean()
            
            print(f"\n{model} - Top-3 Sensitivity by Test Dataset:")
            for dataset, perf in dataset_stats.items():
                print(f"  {dataset}: {perf:.2f}%")
            
            dataset_std = dataset_stats.std()
            print(f"  Cross-dataset std: {dataset_std:.2f}%")

    if not top5_sensitivity_df.empty:
        print(f"\nTop-5 Sensitivity - Cross-dataset consistency:")
        for model in top5_sensitivity_df['Model'].unique():
            model_data = top5_sensitivity_df[top5_sensitivity_df['Model'] == model]
            dataset_stats = model_data.groupby('Test_Dataset')['Mean_Top5_Sensitivity'].mean()
            
            print(f"\n{model} - Top-5 Sensitivity by Test Dataset:")
            for dataset, perf in dataset_stats.items():
                print(f"  {dataset}: {perf:.2f}%")
            
            dataset_std = dataset_stats.std()
            print(f"  Cross-dataset std: {dataset_std:.2f}%")

def main():
    """Main function to run generalization analysis"""
    
    # Set up matplotlib style
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    print("Starting generalization analysis...")
    
    # Parse the generalization results file
    log_file = 'combined_gen.txt'  # Your generalization results file
    log_path = os.path.join(log_directory, log_file)
    
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found!")
        print("Please ensure the combined_gen.txt file is in the correct directory.")
        return
    
    print(f"Parsing generalization results from {log_file}...")
    # FIXED: Properly unpack all 3 return values
    balanced_acc_df, top3_sensitivity_df, top5_sensitivity_df = parse_generalization_results(log_path)
    
    if balanced_acc_df.empty and top3_sensitivity_df.empty and top5_sensitivity_df.empty:
        print("Error: No data could be parsed from the file!")
        return
    
    print(f"Successfully parsed:")
    print(f"  - Balanced Accuracy: {len(balanced_acc_df)} data points")
    print(f"  - Top-3 Sensitivity: {len(top3_sensitivity_df)} data points")
    print(f"  - Top-5 Sensitivity: {len(top5_sensitivity_df)} data points")
    
    if not balanced_acc_df.empty:
        print(f"  - Models (Balanced Accuracy): {balanced_acc_df['Model'].unique()}")
        print(f"  - Test datasets: {balanced_acc_df['Test_Dataset'].unique()}")
        print(f"  - Skin tones: {sorted(balanced_acc_df['Skin_Tone'].unique())}")
    
    if not top3_sensitivity_df.empty:
        print(f"  - Models (Top-3 Sensitivity): {top3_sensitivity_df['Model'].unique()}")
    
    if not top5_sensitivity_df.empty:
        print(f"  - Models (Top-5 Sensitivity): {top5_sensitivity_df['Model'].unique()}")
    
    # Create the visualizations (three separate plots)
    print("\nCreating performance visualizations...")
    plot_generalization_performance(balanced_acc_df, top3_sensitivity_df, top5_sensitivity_df, output_directory)
    
    # Analyze patterns for all metrics
    analyze_performance_patterns(balanced_acc_df, top3_sensitivity_df, top5_sensitivity_df)
    
    print(f"\nAnalysis complete! All outputs saved to: {output_directory}")

if __name__ == "__main__":
    main()