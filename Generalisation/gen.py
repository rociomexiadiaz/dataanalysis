import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Configuration
data_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\Generalisation"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

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
            
            # Parse overall accuracies
            overall_match = re.search(r'=== OVERALL ACCURACIES ===\s+Top-1 Accuracy: ([\d.]+)%\s+Top-3 Accuracy: ([\d.]+)%\s+Top-5 Accuracy: ([\d.]+)%', fold_content)
            if overall_match:
                top1_acc, top3_acc, top5_acc = map(float, overall_match.groups())
                
                results.append({
                    'Test_Dataset': fold_name,
                    'Metric': 'Overall_Accuracy',
                    'Top_1': top1_acc,
                    'Top_3': top3_acc,
                    'Top_5': top5_acc
                })
            
            # Parse average sensitivities
            avg_sens_match = re.search(r'Average Top-1 Sensitivity: ([\d.]+)%', fold_content)
            avg_sens_3_match = re.search(r'Average Top-3 Sensitivity: ([\d.]+)%', fold_content)
            avg_sens_5_match = re.search(r'Average Top-5 Sensitivity: ([\d.]+)%', fold_content)
            
            if avg_sens_match and avg_sens_3_match and avg_sens_5_match:
                results.append({
                    'Test_Dataset': fold_name,
                    'Metric': 'Average_Sensitivity',
                    'Top_1': float(avg_sens_match.group(1)),
                    'Top_3': float(avg_sens_3_match.group(1)),
                    'Top_5': float(avg_sens_5_match.group(1))
                })
            
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
            
            # Parse misclassification counts
            misclass_match = re.search(r'Number of misclassified samples: (\d+)', fold_content)
            if misclass_match:
                results.append({
                    'Test_Dataset': fold_name,
                    'Metric': 'Misclassified_Count',
                    'Count': int(misclass_match.group(1))
                })
            
            # Parse top misclassifications
            overall_misclass_section = re.search(r'=== MOST COMMON MISCLASSIFICATIONS \(OVERALL\) ===\n(.*?)(?=\n===|\nGradCAM|$)', fold_content, re.DOTALL)
            if overall_misclass_section:
                misclass_content = overall_misclass_section.group(1)
                misclass_matches = re.findall(r'(\w+(?:\s+\w+)*)\s*â†\'\s*(\w+(?:\s+\w+)*)\s*:\s*(\d+)\s+times', misclass_content)
                
                for true_class, pred_class, count in misclass_matches:
                    results.append({
                        'Test_Dataset': fold_name,
                        'Metric': 'Misclassification',
                        'True_Class': true_class,
                        'Predicted_Class': pred_class,
                        'Count': int(count)
                    })
    
    return results

# Load and parse generalization results
print("Loading cross-dataset generalization results...")

gen_file = os.path.join(r'Generalisation results\Baseline\dermie_experiment_20250705_112007.txt')
gen_results = parse_generalization_results(gen_file)

# Convert to DataFrame
df = pd.DataFrame(gen_results)

print(f"Loaded {len(df)} generalization results")
print(f"Test datasets: {df['Test_Dataset'].unique()}")
print(f"Metrics: {df['Metric'].unique()}")

# 1. Overall Performance Across Datasets
print("\n=== Creating Generalization Visualizations ===")

# Filter overall accuracy and sensitivity results
overall_results = df[df['Metric'].isin(['Overall_Accuracy', 'Average_Sensitivity'])].copy()

if len(overall_results) > 0:
    # Create comprehensive performance plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot 1: Overall Accuracy by Dataset
    accuracy_data = overall_results[overall_results['Metric'] == 'Overall_Accuracy']
    if len(accuracy_data) > 0:
        datasets = accuracy_data['Test_Dataset']
        x = np.arange(len(datasets))
        width = 0.25
        
        ax1.bar(x - width, accuracy_data['Top_1'], width, label='Top-1', alpha=0.8)
        ax1.bar(x, accuracy_data['Top_3'], width, label='Top-3', alpha=0.8)
        ax1.bar(x + width, accuracy_data['Top_5'], width, label='Top-5', alpha=0.8)
        
        ax1.set_xlabel('Test Dataset')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Cross-Dataset Generalization: Overall Accuracy')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, dataset in enumerate(datasets):
            dataset_data = accuracy_data[accuracy_data['Test_Dataset'] == dataset]
            if len(dataset_data) > 0:
                ax1.text(i - width, dataset_data['Top_1'].iloc[0] + 1, f"{dataset_data['Top_1'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
                ax1.text(i, dataset_data['Top_3'].iloc[0] + 1, f"{dataset_data['Top_3'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
                ax1.text(i + width, dataset_data['Top_5'].iloc[0] + 1, f"{dataset_data['Top_5'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Average Sensitivity by Dataset
    sensitivity_data = overall_results[overall_results['Metric'] == 'Average_Sensitivity']
    if len(sensitivity_data) > 0:
        datasets = sensitivity_data['Test_Dataset']
        x = np.arange(len(datasets))
        
        ax2.bar(x - width, sensitivity_data['Top_1'], width, label='Top-1', alpha=0.8)
        ax2.bar(x, sensitivity_data['Top_3'], width, label='Top-3', alpha=0.8)
        ax2.bar(x + width, sensitivity_data['Top_5'], width, label='Top-5', alpha=0.8)
        
        ax2.set_xlabel('Test Dataset')
        ax2.set_ylabel('Average Sensitivity (%)')
        ax2.set_title('Cross-Dataset Generalization: Average Sensitivity')
        ax2.set_xticks(x)
        ax2.set_xticklabels(datasets, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for i, dataset in enumerate(datasets):
            dataset_data = sensitivity_data[sensitivity_data['Test_Dataset'] == dataset]
            if len(dataset_data) > 0:
                ax2.text(i - width, dataset_data['Top_1'].iloc[0] + 1, f"{dataset_data['Top_1'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
                ax2.text(i, dataset_data['Top_3'].iloc[0] + 1, f"{dataset_data['Top_3'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
                ax2.text(i + width, dataset_data['Top_5'].iloc[0] + 1, f"{dataset_data['Top_5'].iloc[0]:.1f}", ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Misclassification Counts
    misclass_counts = df[df['Metric'] == 'Misclassified_Count']
    if len(misclass_counts) > 0:
        ax3.bar(misclass_counts['Test_Dataset'], misclass_counts['Count'], alpha=0.8, color='red')
        ax3.set_xlabel('Test Dataset')
        ax3.set_ylabel('Number of Misclassified Samples')
        ax3.set_title('Cross-Dataset Generalization: Misclassification Counts')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (dataset, count) in enumerate(zip(misclass_counts['Test_Dataset'], misclass_counts['Count'])):
            ax3.text(i, count + max(misclass_counts['Count']) * 0.01, str(count), ha='center', va='bottom')
    
    # Plot 4: Performance Drop from Best
    if len(accuracy_data) > 0:
        best_top1 = accuracy_data['Top_1'].max()
        performance_drops = best_top1 - accuracy_data['Top_1']
        
        bars = ax4.bar(accuracy_data['Test_Dataset'], performance_drops, alpha=0.8, color='orange')
        ax4.set_xlabel('Test Dataset')
        ax4.set_ylabel('Performance Drop from Best (%)')
        ax4.set_title('Cross-Dataset Generalization: Performance Drops')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # Color bars by severity
        for bar, drop in zip(bars, performance_drops):
            if drop > 30:
                bar.set_color('red')
            elif drop > 15:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # Add value labels
        for i, drop in enumerate(performance_drops):
            ax4.text(i, drop + max(performance_drops) * 0.02, f'{drop:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'baseline_cross_dataset_performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: cross_dataset_performance_overview.png")

# 2. Condition-Specific Generalization Analysis
condition_data = df[df['Metric'] == 'Condition_Sensitivity']

if len(condition_data) > 0:
    # Focus on Top-1 sensitivity for detailed analysis
    top1_conditions = condition_data[condition_data['Top_K'] == 'Top-1']
    
    if len(top1_conditions) > 0:
        # Create condition-dataset heatmap
        condition_pivot = top1_conditions.pivot_table(values='Sensitivity', 
                                                    index='Condition', 
                                                    columns='Test_Dataset', 
                                                    fill_value=np.nan)
        
        plt.figure(figsize=(12, 10))
        mask = condition_pivot.isnull()
        sns.heatmap(condition_pivot, 
                   annot=True, 
                   fmt='.1f', 
                   cmap='RdYlGn',
                   mask=mask,
                   cbar_kws={'label': 'Top-1 Sensitivity (%)'})
        
        plt.title('Cross-Dataset Generalization: Condition Sensitivity Heatmap')
        plt.xlabel('Test Dataset')
        plt.ylabel('Skin Condition')
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'baseline_condition_generalization_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: condition_generalization_heatmap.png")
        
        # Condition robustness analysis
        condition_stats = top1_conditions.groupby('Condition')['Sensitivity'].agg(['mean', 'std', 'min', 'max']).reset_index()
        condition_stats['range'] = condition_stats['max'] - condition_stats['min']
        condition_stats = condition_stats.sort_values('std', ascending=False)
        
        plt.figure(figsize=(14, 8))
        x = np.arange(len(condition_stats))
        
        bars = plt.bar(x, condition_stats['std'], alpha=0.8)
        plt.xlabel('Skin Condition')
        plt.ylabel('Standard Deviation (%)')
        plt.title('Cross-Dataset Generalization: Condition Robustness\n(Higher std = less robust across datasets)')
        plt.xticks(x, condition_stats['Condition'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Color by robustness
        for bar, std_val in zip(bars, condition_stats['std']):
            if std_val > 20:
                bar.set_color('red')
            elif std_val > 10:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # Add value labels
        for i, std_val in enumerate(condition_stats['std']):
            plt.text(i, std_val + max(condition_stats['std']) * 0.02, f'{std_val:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'baseline_condition_robustness_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: condition_robustness_analysis.png")

# 3. Skin Tone Generalization Analysis
skin_tone_data = df[df['Metric'] == 'Skin_Tone_Accuracy']

if len(skin_tone_data) > 0:
    # Focus on Top-1 accuracy
    top1_skin_tone = skin_tone_data[skin_tone_data['Top_K'] == 'Top-1']
    
    if len(top1_skin_tone) > 0:
        # Create skin tone generalization plot
        plt.figure(figsize=(14, 8))
        
        datasets = top1_skin_tone['Test_Dataset'].unique()
        skin_tones = sorted(top1_skin_tone['Skin_Tone'].unique())
        
        x = np.arange(len(skin_tones))
        width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            dataset_data = top1_skin_tone[top1_skin_tone['Test_Dataset'] == dataset]
            
            # Ensure we have data for all skin tones
            skin_tone_values = []
            for tone in skin_tones:
                tone_data = dataset_data[dataset_data['Skin_Tone'] == tone]
                if len(tone_data) > 0:
                    skin_tone_values.append(tone_data['Accuracy'].iloc[0])
                else:
                    skin_tone_values.append(0)
            
            plt.bar(x + i*width, skin_tone_values, width, label=dataset, alpha=0.8)
        
        plt.xlabel('Fitzpatrick Skin Type')
        plt.ylabel('Top-1 Accuracy (%)')
        plt.title('Cross-Dataset Generalization: Skin Tone Performance')
        plt.xticks(x + width * (len(datasets) - 1) / 2, [f'FST {int(tone)}' for tone in skin_tones])
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, 'baseline_skin_tone_generalization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: skin_tone_generalization.png")

# 4. High-Risk Misclassification Analysis
misclass_data = df[df['Metric'] == 'Misclassification']

if len(misclass_data) > 0:
    # Define high-risk pairs
    high_risk_pairs = [
        ('melanoma', 'nevus'), ('melanoma', 'melanocytic'), ('melanoma', 'psoriasis'),
        ('melanoma', 'eczema'), ('bcc', 'psoriasis'), ('bcc', 'eczema'),
        ('scc', 'psoriasis'), ('scc', 'eczema')
    ]
    
    def is_high_risk(true_class, pred_class):
        true_norm = true_class.lower().replace(' ', '').replace('_', '')
        pred_norm = pred_class.lower().replace(' ', '').replace('_', '')
        
        for risk_true, risk_pred in high_risk_pairs:
            if risk_true in true_norm and risk_pred in pred_norm:
                return True
        return False
    
    misclass_data['High_Risk'] = misclass_data.apply(
        lambda row: is_high_risk(row['True_Class'], row['Predicted_Class']), axis=1
    )
    
    # High-risk misclassifications by dataset
    high_risk_summary = misclass_data.groupby(['Test_Dataset', 'High_Risk'])['Count'].sum().reset_index()
    high_risk_pivot = high_risk_summary.pivot_table(values='Count', 
                                                   index='Test_Dataset', 
                                                   columns='High_Risk', 
                                                   fill_value=0)
    
    plt.figure(figsize=(12, 6))
    x = np.arange(len(high_risk_pivot.index))
    width = 0.35
    
    if False in high_risk_pivot.columns:
        plt.bar(x - width/2, high_risk_pivot[False], width, label='Regular Misclassifications', alpha=0.8)
    if True in high_risk_pivot.columns:
        plt.bar(x + width/2, high_risk_pivot[True], width, label='High-Risk Misclassifications', 
                color='red', alpha=0.8)
    
    plt.xlabel('Test Dataset')
    plt.ylabel('Number of Misclassifications')
    plt.title('Cross-Dataset Generalization: High-Risk vs Regular Misclassifications')
    plt.xticks(x, high_risk_pivot.index, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'high_risk_misclassifications_by_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: high_risk_misclassifications_by_dataset.png")

# 5. Summary Statistics
print(f"\n=== Cross-Dataset Generalization Summary ===")

if len(overall_results) > 0:
    accuracy_data = overall_results[overall_results['Metric'] == 'Overall_Accuracy']
    sensitivity_data = overall_results[overall_results['Metric'] == 'Average_Sensitivity']
    
    if len(accuracy_data) > 0:
        print("Overall Accuracy Performance:")
        best_dataset = accuracy_data.loc[accuracy_data['Top_1'].idxmax(), 'Test_Dataset']
        worst_dataset = accuracy_data.loc[accuracy_data['Top_1'].idxmin(), 'Test_Dataset']
        
        print(f"  Best generalization: {best_dataset} ({accuracy_data['Top_1'].max():.2f}% Top-1)")
        print(f"  Worst generalization: {worst_dataset} ({accuracy_data['Top_1'].min():.2f}% Top-1)")
        print(f"  Performance range: {accuracy_data['Top_1'].max() - accuracy_data['Top_1'].min():.2f}%")
        print(f"  Average performance: {accuracy_data['Top_1'].mean():.2f}% ± {accuracy_data['Top_1'].std():.2f}%")

if len(condition_data) > 0:
    top1_conditions = condition_data[condition_data['Top_K'] == 'Top-1']
    condition_stats = top1_conditions.groupby('Condition')['Sensitivity'].agg(['mean', 'std']).reset_index()
    
    print(f"\nCondition Robustness Analysis:")
    most_robust = condition_stats.loc[condition_stats['std'].idxmin()]
    least_robust = condition_stats.loc[condition_stats['std'].idxmax()]
    
    print(f"  Most robust condition: {most_robust['Condition']} (std: {most_robust['std']:.2f}%)")
    print(f"  Least robust condition: {least_robust['Condition']} (std: {least_robust['std']:.2f}%)")

if len(misclass_data) > 0:
    print(f"\nMisclassification Analysis:")
    total_misclass = misclass_data['Count'].sum()
    high_risk_count = misclass_data[misclass_data['High_Risk'] == True]['Count'].sum()
    
    print(f"  Total misclassifications: {total_misclass}")
    print(f"  High-risk misclassifications: {high_risk_count} ({high_risk_count/total_misclass*100:.1f}%)")
    
    # Most problematic dataset
    dataset_misclass = misclass_data.groupby('Test_Dataset')['Count'].sum().reset_index()
    worst_misclass_dataset = dataset_misclass.loc[dataset_misclass['Count'].idxmax(), 'Test_Dataset']
    print(f"  Most challenging dataset: {worst_misclass_dataset}")

# Save detailed results
df.to_csv(os.path.join(output_directory, 'generalization_results.csv'), index=False)
print("✓ Saved: generalization_results.csv")

print(f"\nAll generalization analysis figures saved to: {output_directory}")
print(f"Generated {len([f for f in os.listdir(output_directory) if f.endswith('.png')])} visualizations")

print(f"\n=== Generalization Analysis Complete ===")
print("This analysis shows how well your model generalizes across different datasets")
print("Key insights: dataset-specific biases, condition robustness, and domain transfer challenges")