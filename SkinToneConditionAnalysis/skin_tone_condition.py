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

# Add this F1 score parsing logic to your parse_combined_log function in the dataframe module

def parse_all_log_files(log_directory='./'):
    """Parse all log files and combine into a single dataset"""
    
    # Parse both files
    combined_data = {
        'AverageSensitivities': [],
        'ConditionSensitivities': [],
        'SkinToneAccuracies': [],
        'StratifiedSensitivities': [],
        'MisclassifiedCounts': [],
        'MisclassificationDetails': [],
        'F1Scores':[]
    }
    
    # Parse combined_all.txt (all models on all datasets)
    all_file_path = os.path.join(log_directory, 'combined_all.txt')
    if os.path.exists(all_file_path):
        print(f"Parsing combined_all.txt...")
        df_dict_all = parse_combined_log(all_file_path)
        
        for key in combined_data.keys():
            if key in df_dict_all and not df_dict_all[key].empty:
                df_copy = df_dict_all[key].copy()
                df_copy['Source'] = 'all_models'
                combined_data[key].append(df_copy)
    
    # Parse combined_baseline_datasets.txt (baseline model on different dataset combinations)
    baseline_file_path = os.path.join(log_directory, 'combined_baseline_datasets.txt')
    if os.path.exists(baseline_file_path):
        print(f"Parsing combined_baseline_datasets.txt...")
        df_dict_baseline = parse_baseline_datasets_log(baseline_file_path)
        
        for key in combined_data.keys():
            if key in df_dict_baseline and not df_dict_baseline[key].empty:
                df_copy = df_dict_baseline[key].copy()
                df_copy['Source'] = 'baseline_datasets'
                combined_data[key].append(df_copy)
    
    # Combine all dataframes
    final_data = {}
    for key in combined_data.keys():
        if combined_data[key]:
            final_data[key] = pd.concat(combined_data[key], ignore_index=True)
        else:
            final_data[key] = pd.DataFrame()
    
    return final_data

def parse_baseline_datasets_log(filepath):
    """Parse combined_baseline_datasets.txt and extract Dataset Combination field"""
    with open(filepath, 'r') as file:
        content = file.read()

    # Split models by 'Python Filename:'
    model_blocks = re.split(r'(?=Python Filename:)', content)

    # Containers
    avg_sensitivities = []
    condition_sensitivities = []
    skin_tone_sensitivities = []
    stratified_sensitivities = []
    misclassifications = []
    misclassifications_by_tone = []
    f1_scores = []  # NEW: Add F1 scores container

    for block in model_blocks:
        if not block.strip():
            continue

        model_match = re.search(r'Python Filename:\s*(\S+)', block)
        if not model_match:
            continue
        model = model_match.group(1).replace('.py', '')
        
        # Extract Dataset Combination instead of Datasets
        dataset_combination_match = re.search(r'Dataset Combination:\s*(.+)', block)
        if not dataset_combination_match:
            continue
        dataset_combination = dataset_combination_match.group(1).strip()
        
        # Also extract the full datasets info for reference
        datasets_match = re.search(r'Datasets:\s*(.+)', block)
        datasets = datasets_match.group(1) if datasets_match else dataset_combination

        # Parse all the same data but use dataset_combination as the Datasets field
        ### 1. Average Top-k Sensitivity
        for k in ['Top-1', 'Top-3', 'Top-5']:
            match = re.search(rf'Average {k} Sensitivity:\s*([\d.]+)%', block)
            if match:
                avg_sensitivities.append({
                    'Model': model,
                    'Datasets': dataset_combination,  # Use Dataset Combination field
                    'Metric': f'{k} Sensitivity',
                    'Value': float(match.group(1))
                })

        ### 2. Top-k Sensitivity per condition
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                condition_sensitivities.append({
                    'Model': model,
                    'Datasets': dataset_combination,  # Use Dataset Combination field
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })

        ### 3. Top-k Sensitivity per skin tone
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for tone_match in re.findall(rf'Skin Tone:\s*([\d.]+), {k} Accuracy:\s*([\d.]+)%', block):
                tone, value = tone_match
                skin_tone_sensitivities.append({
                    'Model': model,
                    'Datasets': dataset_combination,  # Use Dataset Combination field
                    'Skin Tone': float(tone),
                    'Metric': f'{k} Accuracy',
                    'Value': float(value)
                })

        ### 4. Stratified Top-k Sensitivity per skin tone and condition
        stratified_blocks = re.findall(r'Skin Tone:\s*([\d.]+)\n((?:\s+Condition:.*\n?)+)', block)
        for tone, section in stratified_blocks:
            matches = re.findall(r'Condition:\s*(.+?), (Top-[135]) Sensitivity:\s*([\d.]+)%', section)
            for condition, k, value in matches:
                stratified_sensitivities.append({
                    'Model': model,
                    'Datasets': dataset_combination,  # Use Dataset Combination field
                    'Skin Tone': float(tone),
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })

        ### 5. NEW: Parse F1 Score data
        # Overall F1 Scores (Macro and Weighted)
        for f1_type in ['Macro', 'Weighted']:
            match = re.search(rf'Overall F1 Score \({f1_type}\):\s*([\d.]+)%', block)
            if match:
                f1_scores.append({
                    'Model': model,
                    'Datasets': dataset_combination,
                    'Skin Tone': None,  # Overall score, no specific skin tone
                    'Condition': 'Overall',
                    'Metric': f'F1 Score ({f1_type})',
                    'Value': float(match.group(1))
                })

        # Per Condition F1 Scores (from main F1 SCORE section)
        f1_section = re.search(r'=== F1 SCORE ===(.+?)(?==== |$)', block, re.DOTALL)
        if f1_section:
            f1_content = f1_section.group(1)
            condition_matches = re.findall(r'^\s*([^:]+):\s*([\d.]+)%', f1_content, re.MULTILINE)
            for condition, value in condition_matches:
                condition = condition.strip()
                f1_scores.append({
                    'Model': model,
                    'Datasets': dataset_combination,
                    'Skin Tone': None,  # Overall per condition, no specific skin tone
                    'Condition': condition,
                    'Metric': 'F1 Score',
                    'Value': float(value)
                })

        # Stratified F1 Scores by Skin Tone
        stratified_f1_section = re.search(r'=== STRATIFIED F1 SCORE ===(.+?)(?==== |$)', block, re.DOTALL)
        if stratified_f1_section:
            stratified_content = stratified_f1_section.group(1)
            
            # Find all skin tone blocks
            skin_tone_blocks = re.findall(r'Skin Tone:\s*([\d.]+)(.*?)(?=Skin Tone:|$)', stratified_content, re.DOTALL)
            
            for tone, tone_content in skin_tone_blocks:
                tone_val = float(tone)
                
                # Overall F1 scores for this skin tone
                for f1_type in ['Macro', 'Weighted']:
                    match = re.search(rf'Overall F1 Score \({f1_type}\):\s*([\d.]+)%', tone_content)
                    if match:
                        f1_scores.append({
                            'Model': model,
                            'Datasets': dataset_combination,
                            'Skin Tone': tone_val,
                            'Condition': 'Overall',
                            'Metric': f'F1 Score ({f1_type})',
                            'Value': float(match.group(1))
                        })
                
                # Per Condition F1 scores for this skin tone
                per_condition_section = re.search(r'Per Condition:(.*?)(?=\n\nSkin Tone:|$)', tone_content, re.DOTALL)
                if per_condition_section:
                    condition_content = per_condition_section.group(1)
                    condition_matches = re.findall(r'^\s*([^:]+):\s*([\d.]+)%', condition_content, re.MULTILINE)
                    for condition, value in condition_matches:
                        condition = condition.strip()
                        f1_scores.append({
                            'Model': model,
                            'Datasets': dataset_combination,
                            'Skin Tone': tone_val,
                            'Condition': condition,
                            'Metric': 'F1 Score',
                            'Value': float(value)
                        })

    # Convert all to DataFrames
    return {
        'AverageSensitivities': pd.DataFrame(avg_sensitivities),
        'ConditionSensitivities': pd.DataFrame(condition_sensitivities),
        'SkinToneAccuracies': pd.DataFrame(skin_tone_sensitivities),
        'StratifiedSensitivities': pd.DataFrame(stratified_sensitivities),
        'MisclassifiedCounts': pd.DataFrame(misclassifications),
        'MisclassificationDetails': pd.DataFrame(misclassifications_by_tone),
        'F1Scores': pd.DataFrame(f1_scores)  # NEW: Return F1 scores dataframe
    }


# Fitzpatrick Skin Tone color mapping
fst_color_map = {
    1: '#F5D5A0',
    2: '#E4B589',
    3: '#D1A479',
    4: '#C0874F',
    5: '#A56635',
    6: '#4C2C27'
}

def create_skin_tone_plot_with_disparity(data, x_col, title, ylabel, save_path, condition_name, plot_type):
    """Create a skin tone plot with disparity metrics"""
    
    if data.empty:
        print(f"No data available for {title}")
        return
    
    # Filter out FairDisCo models
    data = data[~data['Model'].str.contains('FairDisCo', case=False, na=False)]
    
    if data.empty:
        print(f"No data available for {title} after filtering")
        return
    
    # Group data and calculate stats
    grouped = data.groupby([x_col, 'Skin Tone'])['Value'].agg(['mean', 'std']).reset_index()
    
    # Get unique values for x-axis
    x_values = sorted(grouped[x_col].unique())
    skin_tones = [1, 2, 3, 4, 5, 6]
    
    # Create the main plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Main sensitivity/f1 plot
    x = np.arange(len(x_values))
    width = 0.8 / len(skin_tones)
    
    plotted_tones = set()
    for i, tone in enumerate(skin_tones):
        values = []
        errors = []
        for x_val in x_values:
            row = grouped[(grouped[x_col] == x_val) & (grouped['Skin Tone'] == tone)]
            if not row.empty:
                values.append(row['mean'].values[0])
                errors.append(row['std'].values[0])
            else:
                values.append(0)
                errors.append(0)
        
        bar_positions = x + i * width
        label = f'FST {tone}' if tone not in plotted_tones else None
        bars = ax1.bar(bar_positions, values, width, yerr=errors, capsize=4,
                    label=label, color=fst_color_map[tone], alpha=0.8, edgecolor='black', linewidth=0.5)

        plotted_tones.add(tone)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + 1.0  # Adjust offset as needed (1.0 works in most cases)

            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax1.text(
                    xpos,
                    ypos,
                    f'{val:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black'
                )

    
      
    ax1.set_xlabel(x_col.replace('_', ' ').title())
    ax1.set_ylabel(ylabel)
    ax1.set_title(f'{title} - {condition_name}')
    ax1.set_xticks(x + width * (len(skin_tones) - 1) / 2)
    ax1.set_xticklabels(x_values, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Disparity analysis subplot - only standard deviation
    pivot = grouped.pivot(index='Skin Tone', columns=x_col, values='mean')
    
    # Calculate only standard deviation disparity
    disparity_std = pivot.std(axis=0)
    
    x_disp = np.arange(len(x_values))
    
    bars = ax2.bar(x_disp, disparity_std.values, 
            color='#abdda4', alpha=0.8, edgecolor='black', linewidth=0.5)
    
    for bar, val in zip(bars, disparity_std):
            height = bar.get_height()
            xpos = bar.get_x() + bar.get_width() / 2
            ypos = height + 0.1
            if np.isfinite(xpos) and np.isfinite(ypos) and np.isfinite(val):
                ax2.text(
                    xpos,
                    ypos,
                    f'{val:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color='black'
                )

    
    
    ax2.set_xlabel(x_col.replace('_', ' ').title())
    ax2.set_ylabel('Skin Tone Disparity (Std)')
    ax2.set_title(f'Disparity Analysis - {condition_name}')
    ax2.set_xticks(x_disp)
    ax2.set_xticklabels(x_values, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with condition-specific filename
    filename = f"{condition_name.lower().replace(' ', '_')}_{plot_type}"
    plt.savefig(f'{save_path}/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def generate_condition_plots(df_dict, save_path='./'):
    """Generate the 4 required plots for each condition"""
    
    # Clean up model names
    for df_name in df_dict.keys():
        if 'Model' in df_dict[df_name].columns:
            # Clean up model names
            df_dict[df_name]['Model'] = df_dict[df_name]['Model'].replace({
                'train_Baseline': 'Baseline',
                'train_VAE': 'VAE',
                'train_TABE': 'TABE'
            })
    
    # Use separate dataframes for F1 and sensitivity plots
    f1_data = df_dict['F1Scores']
    stratified_data = df_dict['StratifiedSensitivities']
    
    if f1_data.empty and stratified_data.empty:
        print("No F1 or stratified sensitivity data found!")
        return
    
    # Get conditions from both dataframes
    conditions = set()
    if not f1_data.empty:
        conditions.update(f1_data['Condition'].unique())
    if not stratified_data.empty:
        conditions.update(stratified_data['Condition'].unique())
    
    # Remove 'Overall' from conditions for condition-specific plots
    conditions = [c for c in conditions if c != 'Overall']
    print(f"Found conditions: {conditions}")
    
    # Debug: Print available data sources and datasets
    print("\nDebug info:")
    if not f1_data.empty:
        print("F1 Data:")
        print(f"  Available Sources: {f1_data['Source'].unique() if 'Source' in f1_data.columns else 'No Source column'}")
        print(f"  Available Datasets: {f1_data['Datasets'].unique()}")
        print(f"  Available Models: {f1_data['Model'].unique()}")
        print(f"  Available Metrics: {f1_data['Metric'].unique()}")
        print(f"  Available Conditions: {f1_data['Condition'].unique()}")
    
    if not stratified_data.empty:
        print("Stratified Data:")
        print(f"  Available Sources: {stratified_data['Source'].unique() if 'Source' in stratified_data.columns else 'No Source column'}")
        print(f"  Available Datasets: {stratified_data['Datasets'].unique()}")
        print(f"  Available Models: {stratified_data['Model'].unique()}")
    
    for condition in conditions:
        print(f"\nGenerating plots for condition: {condition}")
        
        # 1. F1 Score by Dataset (using F1Scores dataframe with baseline_datasets source)
        f1_dataset_data = f1_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'baseline_datasets' and "
            f"Metric == 'F1 Score' and "
            f"`Skin Tone`.notna()"
        ).copy()
        
        print(f"  F1 dataset data rows: {len(f1_dataset_data)}")
        if not f1_dataset_data.empty:
            create_skin_tone_plot_with_disparity(
                f1_dataset_data, 'Datasets', 
                'F1 Score by Dataset', 'F1 Score (%)', 
                save_path, condition, 'f1_score_by_dataset_skin_with_disparity.png'
            )
        else:
            print(f"No F1 dataset data found for {condition}")
            # Debug what's available
            debug_f1_baseline = f1_data.query("Source == 'baseline_datasets'").copy()
            print(f"    Total baseline F1 data: {len(debug_f1_baseline)}")
            if not debug_f1_baseline.empty:
                print(f"    Available conditions in baseline F1: {debug_f1_baseline['Condition'].unique()}")
                print(f"    Available metrics in baseline F1: {debug_f1_baseline['Metric'].unique()}")
        
        # 2. F1 Score by Model (using F1Scores dataframe with all_models source)
        f1_model_data = f1_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'all_models' and "
            f"Metric == 'F1 Score' and "
            f"`Skin Tone`.notna()"
        ).copy()
        
        print(f"  F1 model data rows: {len(f1_model_data)}")
        if not f1_model_data.empty:
            create_skin_tone_plot_with_disparity(
                f1_model_data, 'Model', 
                'F1 Score by Model', 'F1 Score (%)', 
                save_path, condition, 'f1_score_by_model_skin_with_disparity.png'
            )
        else:
            print(f"No F1 model data found for {condition}")
            # Debug what's available
            debug_f1_all = f1_data.query("Source == 'all_models'").copy()
            print(f"    Total all_models F1 data: {len(debug_f1_all)}")
            if not debug_f1_all.empty:
                print(f"    Available conditions in all_models F1: {debug_f1_all['Condition'].unique()}")
                print(f"    Available metrics in all_models F1: {debug_f1_all['Metric'].unique()}")
        
        # 3. Sensitivity by Dataset (using StratifiedSensitivities dataframe)
        sens_dataset_data = stratified_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'baseline_datasets' and "
            f"Metric == 'Top-1 Sensitivity'"
        ).copy() if not stratified_data.empty else pd.DataFrame()
        
        print(f"  Sensitivity dataset data rows: {len(sens_dataset_data)}")
        if not sens_dataset_data.empty:
            create_skin_tone_plot_with_disparity(
                sens_dataset_data, 'Datasets', 
                'Sensitivity by Dataset', 'Sensitivity (%)', 
                save_path, condition, 'sensitivity_by_dataset_skin_with_disparity.png'
            )
        else:
            print(f"No dataset sensitivity data found for {condition}")
        
        # 4. Sensitivity by Model (using StratifiedSensitivities dataframe)
        sens_model_data = stratified_data.query(
            f"Condition == '{condition}' and "
            f"Source == 'all_models' and "
            f"Metric == 'Top-1 Sensitivity'"
        ).copy() if not stratified_data.empty else pd.DataFrame()
        
        print(f"  Sensitivity model data rows: {len(sens_model_data)}")
        if not sens_model_data.empty:
            create_skin_tone_plot_with_disparity(
                sens_model_data, 'Model', 
                'Sensitivity by Model', 'Sensitivity (%)', 
                save_path, condition, 'sensitivity_by_model_skin_with_disparity.png'
            )
        else:
            print(f"No model sensitivity data found for {condition}")


def main():
    """Main function to generate condition-specific plots"""
    
    # Set up matplotlib for better plots
    plt.style.use('seaborn-v0_8')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    
    # Directory containing log files
    log_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
    save_path = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\SkinToneConditionAnalysis"
    
    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    print("Parsing all log files...")
    df_dict = parse_all_log_files(log_directory)
    
    print("Data parsing complete!")
    print(f"Available dataframes: {list(df_dict.keys())}")
    
    if 'StratifiedSensitivities' in df_dict:
        print(f"StratifiedSensitivities shape: {df_dict['StratifiedSensitivities'].shape}")
        print("Available conditions:")
        if not df_dict['StratifiedSensitivities'].empty:
            print(df_dict['StratifiedSensitivities']['Condition'].unique())
        else:
            print("No stratified data found!")
    
    # Generate the 4 required plots for each condition
    generate_condition_plots(df_dict, save_path)
    
    print(f"\nAll condition-specific plots saved to: {save_path}")

if __name__ == "__main__":
    main()