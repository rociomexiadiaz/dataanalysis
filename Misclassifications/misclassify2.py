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

def calculate_risk_scores(misclassifications_by_tone):
    """Calculate comprehensive risk score for each model"""
    risk_scores = {}
    
    for model in misclassifications_by_tone['Model'].unique():
        model_data = misclassifications_by_tone[misclassifications_by_tone['Model'] == model]
        
        total_misclass = model_data['Count'].sum()
        high_risk_count = model_data[model_data['High_Risk'] == True]['Count'].sum()
        high_risk_ratio = high_risk_count / total_misclass if total_misclass > 0 else 0
        
        # Weight by severity and frequency
        risk_scores[model] = high_risk_ratio * (1 + np.log(total_misclass + 1))
    
    return pd.Series(risk_scores)

def get_high_risk_distribution(misclassifications_by_tone):
    """Get distribution of high-risk misclassifications across models"""
    high_risk_data = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] == 'All')
    ]
    return high_risk_data.groupby('Model')['Count'].sum()

def get_skin_tone_vulnerability(misclassifications_by_tone):
    """Calculate average high-risk percentage by skin tone across all models"""
    skin_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] != 'All']
    
    vulnerability = {}
    for skin_tone in skin_tone_data['Skin Tone'].unique():
        try:
            skin_tone_int = int(skin_tone)
            tone_data = skin_tone_data[skin_tone_data['Skin Tone'] == skin_tone]
            total_count = tone_data['Count'].sum()
            high_risk_count = tone_data[tone_data['High_Risk'] == True]['Count'].sum()
            vulnerability[skin_tone_int] = (high_risk_count / total_count * 100) if total_count > 0 else 0
        except ValueError:
            continue
    
    return pd.Series(vulnerability).sort_index()

def create_model_comparison_matrix(misclassifications_by_tone):
    """Create a comparison matrix showing relative risk differences between models"""
    models = misclassifications_by_tone['Model'].unique()
    n_models = len(models)
    matrix = np.zeros((n_models, n_models))
    
    # Calculate risk scores for each model
    risk_scores = calculate_risk_scores(misclassifications_by_tone)
    
    for i, model1 in enumerate(models):
        for j, model2 in enumerate(models):
            if i != j:
                matrix[i][j] = risk_scores.get(model1, 0) - risk_scores.get(model2, 0)
    
    return matrix

def create_overall_misclassifications_chart(misclassifications_by_tone, output_directory):
    """Create overall misclassifications chart by model"""
    print("\n=== Creating Overall Misclassifications Chart ===")
    
    # Total misclassifications by model (from detailed data)
    total_misclass = misclassifications_by_tone.groupby('Model')['Count'].sum().reset_index()
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

def create_skin_tone_model_heatmap(misclassifications_by_tone, output_directory):
    """Create heatmap with skin tones on y-axis, models on x-axis, colored by misclassification count"""
    print("\n=== Creating Skin Tone vs Model Heatmap ===")
    
    # Filter out 'All' skin tone data as we want specific skin tones
    skin_tone_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] != 'All']
    
    if len(skin_tone_data) == 0:
        print("No skin tone specific data found, skipping this visualization")
        return
    
    # Sum misclassifications by skin tone and model
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
                cmap='YlOrRd',
                cbar_kws={'label': 'Misclassification Count'},
                linewidths=0.5)
    
    plt.title('Misclassifications by Skin Tone and Model', fontweight='bold', pad=20)
    plt.xlabel('Model')
    plt.ylabel('Skin Tone (Fitzpatrick Scale)')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'skin_tone_model_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: skin_tone_model_heatmap.png")

def create_skin_tone_risk_heatmap(misclassifications_by_tone, output_directory):
    """Create advanced heatmap showing high-risk misclassifications by model and skin tone"""
    print("\n=== Creating Skin Tone Risk Heatmap ===")
    
    high_risk_data = misclassifications_by_tone[
        (misclassifications_by_tone['High_Risk'] == True) &
        (misclassifications_by_tone['Skin Tone'] != 'All')
    ]
    
    if len(high_risk_data) == 0:
        print("No skin tone high-risk data found, skipping this visualization")
        return
    
    # Create pivot table
    skin_tone_pivot = high_risk_data.pivot_table(
        values='Count', index='Model', columns='Skin Tone', fill_value=0
    )
    
    # Clean model names
    skin_tone_pivot.index = skin_tone_pivot.index.str.replace('train_', '')
    
    # Calculate risk intensity (high-risk as % of total misclassifications per model-skin tone)
    all_data = misclassifications_by_tone[misclassifications_by_tone['Skin Tone'] != 'All']
    total_pivot = all_data.pivot_table(
        values='Count', index='Model', columns='Skin Tone', fill_value=0
    )
    total_pivot.index = total_pivot.index.str.replace('train_', '')
    
    intensity_pivot = (skin_tone_pivot / total_pivot * 100).fillna(0)
    
    # Create dual heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Count heatmap
    sns.heatmap(skin_tone_pivot, annot=True, fmt='.0f', cmap='Reds', 
                ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('High-Risk Misclassifications Count', fontweight='bold')
    ax1.set_ylabel('Model')
    ax1.set_xlabel('Skin Tone (Fitzpatrick Scale)')
    
    # Intensity heatmap
    sns.heatmap(intensity_pivot, annot=True, fmt='.1f', cmap='YlOrRd', 
                ax=ax2, cbar_kws={'label': 'Percentage (%)'})
    ax2.set_title('High-Risk Misclassifications Intensity', fontweight='bold')
    ax2.set_ylabel('')
    ax2.set_xlabel('Skin Tone (Fitzpatrick Scale)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'skin_tone_risk_heatmap_dual.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: skin_tone_risk_heatmap_dual.png")

def create_risk_vs_performance_bubble(misclassifications_by_tone, output_directory):
    """Create bubble chart showing risk vs performance trade-off"""
    print("\n=== Creating Risk vs Performance Bubble Chart ===")
    
    # Calculate metrics for each model
    model_metrics = []
    for model in misclassifications_by_tone['Model'].unique():
        model_data = misclassifications_by_tone[misclassifications_by_tone['Model'] == model]
        
        total_misclass = model_data['Count'].sum()
        high_risk_count = model_data[model_data['High_Risk'] == True]['Count'].sum()
        high_risk_pct = (high_risk_count / total_misclass * 100) if total_misclass > 0 else 0
        
        # Calculate skin tone bias (std dev of high-risk across skin tones)
        skin_tone_data = model_data[
            (model_data['High_Risk'] == True) & (model_data['Skin Tone'] != 'All')
        ]
        if len(skin_tone_data) > 0:
            skin_tone_bias = skin_tone_data.groupby('Skin Tone')['Count'].sum().std()
        else:
            skin_tone_bias = 0
        
        model_metrics.append({
            'Model': model.replace('train_', ''),
            'Total_Misclassifications': total_misclass,
            'High_Risk_Percentage': high_risk_pct,
            'Skin_Tone_Bias': skin_tone_bias if pd.notna(skin_tone_bias) else 0
        })
    
    df_metrics = pd.DataFrame(model_metrics)
    
    plt.figure(figsize=(12, 8))
    
    # Create bubble chart
    scatter = plt.scatter(df_metrics['Total_Misclassifications'], 
                         df_metrics['High_Risk_Percentage'],
                         s=df_metrics['Skin_Tone_Bias'] * 50 + 100,  # Min size of 100
                         c=range(len(df_metrics)), cmap='viridis', alpha=0.7)
    
    # Add model labels
    for i, row in df_metrics.iterrows():
        plt.annotate(row['Model'], 
                    (row['Total_Misclassifications'], row['High_Risk_Percentage']),
                    xytext=(5, 5), textcoords='offset points', fontweight='bold')
    
    plt.xlabel('Total Misclassifications')
    plt.ylabel('High-Risk Percentage (%)')
    plt.title('Model Risk vs Performance Trade-off\n(Bubble size = Skin Tone Bias)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'risk_vs_performance_bubble.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: risk_vs_performance_bubble.png")

def create_risk_dashboard(misclassifications_by_tone, output_directory):
    """Create comprehensive risk dashboard for all models"""
    print("\n=== Creating Risk Dashboard ===")
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1])
    
    # 1. Risk Score (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    risk_scores = calculate_risk_scores(misclassifications_by_tone)
    risk_scores.index = risk_scores.index.str.replace('train_', '')
    bars = ax1.bar(risk_scores.index, risk_scores.values, color='red', alpha=0.7)
    ax1.set_title('Risk Score by Model', fontweight='bold')
    ax1.set_ylabel('Risk Score')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, v in enumerate(risk_scores.values):
        ax1.text(i, v + max(risk_scores.values) * 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # 2. High-Risk Distribution (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    high_risk_dist = get_high_risk_distribution(misclassifications_by_tone)
    if len(high_risk_dist) > 0:
        high_risk_dist.index = high_risk_dist.index.str.replace('train_', '')
        wedges, texts, autotexts = ax2.pie(high_risk_dist.values, labels=high_risk_dist.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax2.set_title('High-Risk Distribution', fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No high-risk data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('High-Risk Distribution', fontweight='bold')
    
    # 3. Skin Tone Vulnerability (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    skin_vulnerability = get_skin_tone_vulnerability(misclassifications_by_tone)
    if len(skin_vulnerability) > 0:
        colors = ['#F5D5A0', '#E4B589', '#D1A479', '#C0874F', '#A56635', '#4C2C27']
        available_colors = colors[:len(skin_vulnerability)]
        ax3.bar(range(len(skin_vulnerability)), skin_vulnerability.values, 
                color=available_colors)
        ax3.set_title('Skin Tone Vulnerability', fontweight='bold')
        ax3.set_xticks(range(len(skin_vulnerability)))
        ax3.set_xticklabels([f'FST {i}' for i in skin_vulnerability.index])
        ax3.set_ylabel('Avg High-Risk %')
    else:
        ax3.text(0.5, 0.5, 'No skin tone data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Skin Tone Vulnerability', fontweight='bold')
    
    # 4. Model Comparison Matrix (bottom span)
    ax4 = fig.add_subplot(gs[1:, :])
    comparison_matrix = create_model_comparison_matrix(misclassifications_by_tone)
    models = [m.replace('train_', '') for m in misclassifications_by_tone['Model'].unique()]
    
    im = ax4.imshow(comparison_matrix, cmap='RdYlBu_r', aspect='auto')
    ax4.set_title('Model Risk Comparison Matrix', fontweight='bold', pad=20)
    ax4.set_xticks(range(len(models)))
    ax4.set_yticks(range(len(models)))
    ax4.set_xticklabels(models, rotation=45)
    ax4.set_yticklabels(models)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, orientation='horizontal', pad=0.1)
    cbar.set_label('Risk Difference Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, 'risk_dashboard.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: risk_dashboard.png")

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

# Create the 3 visualizations you want
print("\n=== Creating Visualizations ===")

# 1. Overall misclassifications (your preferred format, without FairDisCo)
create_overall_misclassifications_chart(misclassifications_by_tone, output_directory)

# 2. High-risk vs low-risk bar chart (2 bars per model)
create_high_low_risk_barchart(misclassifications_by_tone, output_directory)

# 3. Skin tone vs model heatmap
create_skin_tone_model_heatmap(misclassifications_by_tone, output_directory)

# Original analysis and summary statistics
print("\n=== Summary Statistics ===")

# Total misclassifications by model
total_misclass = misclassifications_by_tone.groupby('Model')['Count'].sum().reset_index()
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