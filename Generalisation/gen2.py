import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import defaultdict

# Configuration
# Since code is in DataAnalysis/Generalisation and results are in Generalisation results/
data_directory = r"C:\Users\rmexi\OneDrive - University College London\Project"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Generalisation"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# FST color mapping
fst_color_map = {
    1: '#F5D5A0',
    2: '#E4B589',
    3: '#D1A479',
    4: '#C0874F',
    5: '#A56635',
    6: '#4C2C27'
}

def parse_generalization_results(filepath):
    """
    Parse generalization experiment results from text file
    """
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split by fold sections
    fold_sections = re.split(r'=== Fold: (\w+) ===', content)[1:]  # Skip first empty element
    
    results = []
    
    # Process pairs of (fold_name, fold_content)
    for i in range(0, len(fold_sections), 2):
        if i + 1 < len(fold_sections):
            fold_name = fold_sections[i]
            fold_content = fold_sections[i + 1]
            
            print(f"Processing fold: {fold_name}")
            
            # Parse condition-specific sensitivities
            for k in ['Top-1', 'Top-3', 'Top-5']:
                section_pattern = f'=== {k.upper()} SENSITIVITY ===\n(.*?)(?=\n===|\n\nNumber|\nGradCAM|$)'
                section_match = re.search(section_pattern, fold_content, re.DOTALL)
                
                if section_match:
                    section_content = section_match.group(1)
                    condition_matches = re.findall(r'Condition: (.+?), ' + k + r' Sensitivity: ([\d.]+)%', section_content)
                    
                    for condition, sensitivity in condition_matches:
                        results.append({
                            'Test_Dataset': fold_name,
                            'Metric': f'Condition_Sensitivity',
                            'Condition': condition,
                            'Top_K': k,
                            'Sensitivity': float(sensitivity)
                        })
            
            # Parse skin tone accuracies
            for k in ['Top-1', 'Top-3', 'Top-5']:
                skin_section_pattern = f'=== STRATIFIED {k.upper()} ACCURACY ===\n(.*?)(?=\n===|\n\nNumber|\nGradCAM|$)'
                skin_section_match = re.search(skin_section_pattern, fold_content, re.DOTALL)
                
                if skin_section_match:
                    skin_content = skin_section_match.group(1)
                    skin_matches = re.findall(r'Skin Tone: ([\d.]+), ' + k + r' Accuracy: ([\d.]+)%', skin_content)
                    
                    for skin_tone, accuracy in skin_matches:
                        results.append({
                            'Test_Dataset': fold_name,
                            'Metric': 'Skin_Tone_Accuracy',
                            'Skin_Tone': float(skin_tone),
                            'Top_K': k,
                            'Accuracy': float(accuracy)
                        })
    
    return results

def load_multiple_model_results(base_directory):
    """
    Load results from multiple model experiments
    Structure: ./Generalisation results/model_name/*.txt
    """
    all_results = []
    
    # Path to Generalisation results directory
    gen_results_path = os.path.join(base_directory, 'Generalisation results')
    
    print(f"Looking for results in: {gen_results_path}")
    
    if not os.path.exists(gen_results_path):
        print(f"✗ Generalisation results directory not found: {gen_results_path}")
        return all_results
    
    # List all subdirectories (model names) and exclude unwanted ones
    excluded_models = ['csv', 'fairdisco', 'CSV', 'FairDisco', 'Fairdisco']
    
    model_dirs = [d for d in os.listdir(gen_results_path) 
                  if os.path.isdir(os.path.join(gen_results_path, d)) and d not in excluded_models]
    
    print(f"Found model directories: {model_dirs}")
    
    for model_name in model_dirs:
        model_path = os.path.join(gen_results_path, model_name)
        
        # Find .txt files in this model directory
        txt_files = [f for f in os.listdir(model_path) if f.endswith('.txt')]
        
        if txt_files:
            # Use the first .txt file found
            filepath = os.path.join(model_path, txt_files[0])
            print(f"Loading {model_name} results from: {filepath}")
            
            try:
                results = parse_generalization_results(filepath)
                # Add model name to each result
                for result in results:
                    result['Model'] = model_name
                all_results.extend(results)
                print(f"✓ Loaded {len(results)} results for {model_name}")
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
        else:
            print(f"✗ No .txt files found in {model_path}")
    
    return all_results

# Load results from all models
print("Loading results from all model experiments...")
all_results = load_multiple_model_results(data_directory)

if not all_results:
    print("No results loaded. Please check the directory structure.")
    exit()

# Convert to DataFrame and filter out excluded models
df = pd.DataFrame(all_results)

# Additional filtering to remove any excluded models that might have been loaded
excluded_models_plot = ['csv', 'fairdisco', 'FairDisCo', 'CSV', 'FairDisco', 'Fairdisco']
df = df[~df['Model'].isin(excluded_models_plot)]

print(f"Loaded {len(df)} total results")
print(f"Models after filtering: {df['Model'].unique()}")
print(f"Test datasets: {df['Test_Dataset'].unique()}")
print(f"Metrics: {df['Metric'].unique()}")

# Graph 1: Top-1 Accuracy by Model, colored by FST (aggregated across all datasets)
print("\n=== Creating Graph 1: FST Performance by Model ===")

# Filter skin tone data for Top-1 accuracy
skin_tone_data = df[(df['Metric'] == 'Skin_Tone_Accuracy') & (df['Top_K'] == 'Top-1')]

if len(skin_tone_data) > 0:
    # Aggregate across all datasets for each model and FST
    fst_aggregated = skin_tone_data.groupby(['Model', 'Skin_Tone'])['Accuracy'].mean().reset_index()
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    models = sorted(fst_aggregated['Model'].unique())
    skin_tones = sorted(fst_aggregated['Skin_Tone'].unique())
    
    x = np.arange(len(models))
    width = 0.12  # Width of bars
    
    # Create bars for each FST
    for i, fst in enumerate(skin_tones):
        fst_data = fst_aggregated[fst_aggregated['Skin_Tone'] == fst]
        
        # Ensure we have data for all models
        accuracies = []
        for model in models:
            model_data = fst_data[fst_data['Model'] == model]
            if len(model_data) > 0:
                accuracies.append(model_data['Accuracy'].iloc[0])
            else:
                accuracies.append(0)
        
        # Plot bars with FST-specific colors
        bars = plt.bar(x + i*width, accuracies, width, 
                      label=f'FST {int(fst)}', 
                      color=fst_color_map[int(fst)], 
                      alpha=0.8)
        
        # Add value labels on bars
        for j, (bar, acc) in enumerate(zip(bars, accuracies)):
            if acc > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Top-1 Accuracy by Model and Fitzpatrick Skin Type\n(Aggregated across all test datasets)', fontsize=14)
    plt.xticks(x + width * (len(skin_tones) - 1) / 2, models)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'fst_performance_by_model_aggregated.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: fst_performance_by_model_aggregated.png")
else:
    print("✗ No skin tone data found for Graph 1")

# Graph 2: Top-1 Accuracy by Condition, colored by Model (aggregated across all datasets)
print("\n=== Creating Graph 2: Condition Performance by Model ===")

# Filter condition data for Top-1 sensitivity
condition_data = df[(df['Metric'] == 'Condition_Sensitivity') & (df['Top_K'] == 'Top-1')]

if len(condition_data) > 0:
    # Aggregate across all datasets for each model and condition
    condition_aggregated = condition_data.groupby(['Model', 'Condition'])['Sensitivity'].mean().reset_index()
    
    # Create the plot
    plt.figure(figsize=(16, 10))
    
    conditions = sorted(condition_aggregated['Condition'].unique())
    models = sorted(condition_aggregated['Model'].unique())
    
    x = np.arange(len(conditions))
    width = 0.8 / len(models)  # Adjust width based on number of models
    
    # Define colors for models
    model_colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    # Create bars for each model
    for i, model in enumerate(models):
        model_data = condition_aggregated[condition_aggregated['Model'] == model]
        
        # Ensure we have data for all conditions
        sensitivities = []
        for condition in conditions:
            condition_data_point = model_data[model_data['Condition'] == condition]
            if len(condition_data_point) > 0:
                sensitivities.append(condition_data_point['Sensitivity'].iloc[0])
            else:
                sensitivities.append(0)
        
        # Plot bars with model-specific colors
        bars = plt.bar(x + i*width, sensitivities, width, 
                      label=model, 
                      color=model_colors[i], 
                      alpha=0.8)
        
        # Add value labels on bars (only for non-zero values)
        for j, (bar, sens) in enumerate(zip(bars, sensitivities)):
            if sens > 0:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{sens:.1f}%', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.xlabel('Condition', fontsize=12)
    plt.ylabel('Top-1 Sensitivity (%)', fontsize=12)
    plt.title('Top-1 Sensitivity by Condition and Model\n(Aggregated across all test datasets)', fontsize=14)
    plt.xticks(x + width * (len(models) - 1) / 2, conditions, rotation=45, ha='right')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'condition_performance_by_model_aggregated.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: condition_performance_by_model_aggregated.png")
else:
    print("✗ No condition data found for Graph 2")

# Summary statistics
print(f"\n=== Summary Statistics ===")
print(f"Total results processed: {len(df)}")
print(f"Models analyzed: {list(df['Model'].unique())}")
print(f"Test datasets included: {list(df['Test_Dataset'].unique())}")

if len(skin_tone_data) > 0:
    print(f"\nFST Analysis:")
    print(f"  FST types: {sorted(skin_tone_data['Skin_Tone'].unique())}")
    print(f"  Average performance across all models and FST: {skin_tone_data['Accuracy'].mean():.2f}%")

if len(condition_data) > 0:
    print(f"\nCondition Analysis:")
    print(f"  Conditions analyzed: {len(condition_data['Condition'].unique())}")
    print(f"  Average sensitivity across all models and conditions: {condition_data['Sensitivity'].mean():.2f}%")

# Save aggregated results - removed CSV saving as requested
print(f"\nAll aggregated analysis figures saved to: {output_directory}")
print(f"Generated visualizations showing cross-dataset performance aggregated by model")