import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

# Configuration
data_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis"
output_directory = r"C:\Users\rmexi\OneDrive - University College London\Project\DataAnalysis\OOD"

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Fitzpatrick Skin Tone color mapping
fst_color_map = {
    1: '#F5D5A0',
    2: '#E4B589',
    3: '#D1A479',
    4: '#C0874F',
    5: '#A56635',
    6: '#4C2C27'
}

def parse_ood_file(filepath, model_name):
    """
    Parse OOD detection results from text file
    """
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Split by dataset configurations
    dataset_sections = re.split(r'(?=All\n|Minus \w+)', content)
    
    results = []
    
    for section in dataset_sections:
        if not section.strip():
            continue
        
        # Extract dataset configuration name
        lines = section.strip().split('\n')
        if not lines:
            continue
            
        dataset_config = lines[0].strip()
        
        print(f"Processing dataset: {dataset_config}")
        
        # Initialize variables for each section
        current_fst = None
        skin_percentage = None
        diseased_percentage = None
        sample_count = None
        
        for line in lines[1:]:
            line = line.strip()
            
            # Check for FST (Fitzpatrick Skin Type) header
            fst_match = re.match(r'FST ([\d.]+) \(N=(\d+)\):', line)
            if fst_match:
                # Save previous FST results if they exist
                if current_fst is not None and current_fst != "Overall":
                    # Save skin detection result
                    if skin_percentage is not None:
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': current_fst,
                            'Detection_Type': 'Skin_Detection',
                            'In_Distribution_Pct': skin_percentage,
                            'Sample_Count': sample_count
                        })
                    
                    # Save health detection result
                    if diseased_percentage is not None:
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': current_fst,
                            'Detection_Type': 'Health_Detection',
                            'In_Distribution_Pct': diseased_percentage,
                            'Sample_Count': sample_count
                        })
                
                # Reset for new FST
                current_fst = float(fst_match.group(1))
                sample_count = int(fst_match.group(2))
                skin_percentage = None
                diseased_percentage = None
                continue
            
            # Check for overall section
            if line == "=== Overall ===":
                # Save previous FST results if they exist
                if current_fst is not None and current_fst != "Overall":
                    # Save skin detection result
                    if skin_percentage is not None:
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': current_fst,
                            'Detection_Type': 'Skin_Detection',
                            'In_Distribution_Pct': skin_percentage,
                            'Sample_Count': sample_count
                        })
                    
                    # Save health detection result
                    if diseased_percentage is not None:
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': current_fst,
                            'Detection_Type': 'Health_Detection',
                            'In_Distribution_Pct': diseased_percentage,
                            'Sample_Count': sample_count
                        })
                
                # IMPORTANT: Save any existing Overall results before resetting
                elif current_fst == "Overall":
                    # This is the second Overall section, save the skin detection from the first
                    if skin_percentage is not None:
                        print(f"  Saving Overall skin detection for {dataset_config}: {skin_percentage}%")
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': "Overall",
                            'Detection_Type': 'Skin_Detection',
                            'In_Distribution_Pct': skin_percentage,
                            'Sample_Count': sample_count
                        })
                    
                    # Save health detection result if it exists
                    if diseased_percentage is not None:
                        print(f"  Saving Overall health detection for {dataset_config}: {diseased_percentage}%")
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': "Overall",
                            'Detection_Type': 'Health_Detection',
                            'In_Distribution_Pct': diseased_percentage,
                            'Sample_Count': sample_count
                        })
                
                # Set current to overall and reset percentages
                current_fst = "Overall"
                skin_percentage = None
                diseased_percentage = None
                sample_count = None
                print(f"  Found Overall section for {dataset_config}")
                continue
            
            # Parse percentage lines
            if current_fst is not None:
                if "a close-up of human skin:" in line:
                    skin_percentage = float(re.search(r'([\d.]+)%', line).group(1))
                    # For Overall section, also extract sample count
                    if current_fst == "Overall":
                        sample_count_match = re.search(r'N=(\d+)', line)
                        if sample_count_match:
                            sample_count = int(sample_count_match.group(1))
                    print(f"    Found skin detection: {skin_percentage}% (FST: {current_fst})")
                    
                    # SAVE IMMEDIATELY if this is Overall skin detection
                    if current_fst == "Overall":
                        print(f"  Saving Overall skin detection for {dataset_config}: {skin_percentage}%")
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': "Overall",
                            'Detection_Type': 'Skin_Detection',
                            'In_Distribution_Pct': skin_percentage,
                            'Sample_Count': sample_count
                        })
                        
                elif "a close-up of diseased, unhealthy skin" in line:
                    diseased_percentage = float(re.search(r'([\d.]+)%', line).group(1))
                    # For Overall section, also extract sample count
                    if current_fst == "Overall":
                        sample_count_match = re.search(r'N=(\d+)', line)
                        if sample_count_match:
                            sample_count = int(sample_count_match.group(1))
                    print(f"    Found health detection: {diseased_percentage}% (FST: {current_fst})")
                    
                    # SAVE IMMEDIATELY if this is Overall health detection
                    if current_fst == "Overall":
                        print(f"  Saving Overall health detection for {dataset_config}: {diseased_percentage}%")
                        results.append({
                            'Model': model_name,
                            'Dataset_Config': dataset_config,
                            'Skin_Tone': "Overall",
                            'Detection_Type': 'Health_Detection',
                            'In_Distribution_Pct': diseased_percentage,
                            'Sample_Count': sample_count
                        })
        
        # IMPORTANT: Save final FST/Overall results at the end of each dataset section
        if current_fst is not None:
            # Save skin detection result
            if skin_percentage is not None:
                print(f"  Saving final skin detection for {dataset_config}, FST {current_fst}: {skin_percentage}%")
                results.append({
                    'Model': model_name,
                    'Dataset_Config': dataset_config,
                    'Skin_Tone': current_fst,
                    'Detection_Type': 'Skin_Detection',
                    'In_Distribution_Pct': skin_percentage,
                    'Sample_Count': sample_count
                })
            
            # Save health detection result
            if diseased_percentage is not None:
                print(f"  Saving final health detection for {dataset_config}, FST {current_fst}: {diseased_percentage}%")
                results.append({
                    'Model': model_name,
                    'Dataset_Config': dataset_config,
                    'Skin_Tone': current_fst,
                    'Detection_Type': 'Health_Detection',
                    'In_Distribution_Pct': diseased_percentage,
                    'Sample_Count': sample_count
                })
    
    return results

# Load and parse both OOD files
print("Loading OOD detection results...")

# Parse first model (general OOD - CLIP) - only skin detection
ood_file1 = os.path.join(data_directory, 'ood_report.txt')
results1 = parse_ood_file(ood_file1, 'CLIP')

# Parse second model (lesion OOD - LesionCLIP) - only health detection  
ood_file2 = os.path.join(data_directory, 'lesion_ood_report.txt')
results2 = parse_ood_file(ood_file2, 'LesionCLIP')

# Create separate dataframes for each analysis
clip_df = pd.DataFrame(results1)
lesionclip_df = pd.DataFrame(results2)

print("DEBUG - CLIP data before filtering:")
print(f"Unique Detection Types: {clip_df['Detection_Type'].unique()}")
print(f"Unique Skin Tones: {clip_df['Skin_Tone'].unique()}")
print("Sample CLIP records:")
print(clip_df.head(10))

print("\nDEBUG - CLIP Overall records before filtering:")
clip_overall_debug = clip_df[clip_df['Skin_Tone'] == 'Overall']
print(f"Found {len(clip_overall_debug)} Overall records:")
print(clip_overall_debug)

print("\nDEBUG - LesionCLIP data before filtering:")
print(f"Unique Detection Types: {lesionclip_df['Detection_Type'].unique()}")
print(f"Unique Skin Tones: {lesionclip_df['Skin_Tone'].unique()}")
print("Sample LesionCLIP records:")
print(lesionclip_df.head(10))

# Filter CLIP data to only skin detection
clip_df = clip_df[clip_df['Detection_Type'] == 'Skin_Detection']

# Filter LesionCLIP data to only health detection
lesionclip_df = lesionclip_df[lesionclip_df['Detection_Type'] == 'Health_Detection']

print(f"Loaded {len(clip_df)} CLIP skin detection results")
print(f"Loaded {len(lesionclip_df)} LesionCLIP health detection results")
print(f"CLIP dataset configs: {clip_df['Dataset_Config'].unique()}")
print(f"LesionCLIP dataset configs: {lesionclip_df['Dataset_Config'].unique()}")

print("\n=== Creating OOD Analysis Visualizations ===")

# Define dataset order for consistent plotting
dataset_order = ['All', 'Minus dermie', 'Minus PADUFES', 'Minus SCIN', 'Minus Fitz', 'Minus India']

# 1. CLIP Model: Skin Detection Across Datasets (Overall)
clip_overall = clip_df[clip_df['Skin_Tone'] == 'Overall'].copy()

print(f"Found {len(clip_overall)} CLIP overall records")
print(f"CLIP overall data:\n{clip_overall}")

if len(clip_overall) > 0:
    plt.figure(figsize=(14, 8))
    
    # Order datasets logically
    clip_ordered = []
    
    for dataset in dataset_order:
        dataset_data = clip_overall[clip_overall['Dataset_Config'] == dataset]
        if len(dataset_data) > 0:
            clip_ordered.append(dataset_data.iloc[0])
    
    print(f"Found {len(clip_ordered)} ordered datasets for CLIP")
    
    if clip_ordered:
        clip_ordered_df = pd.DataFrame(clip_ordered)
        
        bars = plt.bar(range(len(clip_ordered_df)), clip_ordered_df['In_Distribution_Pct'], 
                       color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.xlabel('Dataset Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Skin Detection Percentage (%)', fontsize=12, fontweight='bold')
        plt.title('CLIP Model: Skin Detection Across Dataset Configurations', fontsize=14, fontweight='bold')
        plt.xticks(range(len(clip_ordered_df)), clip_ordered_df['Dataset_Config'], rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, clip_ordered_df['In_Distribution_Pct'])):
            plt.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, '1_clip_skin_detection_by_dataset.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 1_clip_skin_detection_by_dataset.png")
    else:
        print("⚠️ No CLIP overall data found for plotting")
else:
    print("⚠️ No CLIP overall records found")

# 2. CLIP Model: Skin Detection Stratified by Skin Tone Across Datasets
clip_by_tone = clip_df[clip_df['Skin_Tone'] != 'Overall'].copy()

if len(clip_by_tone) > 0:
    plt.figure(figsize=(16, 10))
    
    # Get unique skin tones and sort them
    skin_tones = sorted(clip_by_tone['Skin_Tone'].unique())
    
    # Set up the bar positions
    x = np.arange(len(dataset_order))
    width = 0.12  # Width of each bar
    
    # Plot bars for each skin tone
    for i, tone in enumerate(skin_tones):
        tone_data = []
        for dataset in dataset_order:
            dataset_tone_data = clip_by_tone[
                (clip_by_tone['Dataset_Config'] == dataset) & 
                (clip_by_tone['Skin_Tone'] == tone)
            ]
            if len(dataset_tone_data) > 0:
                tone_data.append(dataset_tone_data['In_Distribution_Pct'].iloc[0])
            else:
                tone_data.append(0)
        
        # Use FST color
        color = fst_color_map[int(float(tone))]
        bars = plt.bar(x + i * width, tone_data, width, 
                      label=f'FST {int(float(tone))}', 
                      color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, tone_data)):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xlabel('Dataset Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Skin Detection Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('CLIP Model: Skin Detection by Fitzpatrick Skin Type Across Datasets', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(skin_tones) - 1) / 2, dataset_order, rotation=45, ha='right')
    plt.legend(title='Fitzpatrick Skin Types', loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, '2_clip_skin_detection_by_tone_stratified.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 2_clip_skin_detection_by_tone_stratified.png")

# 3. LesionCLIP Model: Health Detection Across Datasets (Overall)
lesionclip_overall = lesionclip_df[lesionclip_df['Skin_Tone'] == 'Overall'].copy()

if len(lesionclip_overall) > 0:
    plt.figure(figsize=(14, 8))
    
    # Order datasets logically
    lesionclip_ordered = []
    for dataset in dataset_order:
        dataset_data = lesionclip_overall[lesionclip_overall['Dataset_Config'] == dataset]
        if len(dataset_data) > 0:
            lesionclip_ordered.append(dataset_data.iloc[0])
    
    if lesionclip_ordered:
        lesionclip_ordered_df = pd.DataFrame(lesionclip_ordered)
        
        bars = plt.bar(range(len(lesionclip_ordered_df)), lesionclip_ordered_df['In_Distribution_Pct'], 
                       color='crimson', alpha=0.8, edgecolor='black', linewidth=1)
        
        plt.xlabel('Dataset Configuration', fontsize=12, fontweight='bold')
        plt.ylabel('Diseased Skin Detection Percentage (%)', fontsize=12, fontweight='bold')
        plt.title('LesionCLIP Model: Diseased Skin Detection Across Dataset Configurations', 
                 fontsize=14, fontweight='bold')
        plt.xticks(range(len(lesionclip_ordered_df)), lesionclip_ordered_df['Dataset_Config'], 
                  rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, lesionclip_ordered_df['In_Distribution_Pct'])):
            plt.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_directory, '3_lesionclip_health_detection_by_dataset.png'), 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 3_lesionclip_health_detection_by_dataset.png")

# 4. LesionCLIP Model: Health Detection Stratified by Skin Tone Across Datasets
lesionclip_by_tone = lesionclip_df[lesionclip_df['Skin_Tone'] != 'Overall'].copy()

if len(lesionclip_by_tone) > 0:
    plt.figure(figsize=(16, 10))
    
    # Get unique skin tones and sort them
    skin_tones = sorted(lesionclip_by_tone['Skin_Tone'].unique())
    
    # Set up the bar positions
    x = np.arange(len(dataset_order))
    width = 0.12  # Width of each bar
    
    # Plot bars for each skin tone
    for i, tone in enumerate(skin_tones):
        tone_data = []
        for dataset in dataset_order:
            dataset_tone_data = lesionclip_by_tone[
                (lesionclip_by_tone['Dataset_Config'] == dataset) & 
                (lesionclip_by_tone['Skin_Tone'] == tone)
            ]
            if len(dataset_tone_data) > 0:
                tone_data.append(dataset_tone_data['In_Distribution_Pct'].iloc[0])
            else:
                tone_data.append(0)
        
        # Use FST color
        color = fst_color_map[int(float(tone))]
        bars = plt.bar(x + i * width, tone_data, width, 
                      label=f'FST {int(float(tone))}', 
                      color=color, alpha=0.9, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for j, (bar, value) in enumerate(zip(bars, tone_data)):
            if value > 0:
                plt.text(bar.get_x() + bar.get_width()/2., value + 0.5,
                        f'{value:.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.xlabel('Dataset Configuration', fontsize=12, fontweight='bold')
    plt.ylabel('Diseased Skin Detection Percentage (%)', fontsize=12, fontweight='bold')
    plt.title('LesionCLIP Model: Diseased Skin Detection by Fitzpatrick Skin Type Across Datasets', 
              fontsize=14, fontweight='bold')
    plt.xticks(x + width * (len(skin_tones) - 1) / 2, dataset_order, rotation=45, ha='right')
    plt.legend(title='Fitzpatrick Skin Types', loc='upper right', ncol=2)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_directory, '4_lesionclip_health_detection_by_tone_stratified.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved: 4_lesionclip_health_detection_by_tone_stratified.png")

# Summary Statistics
print(f"\n=== Summary Statistics ===")

# CLIP Analysis
if len(clip_overall) > 0:
    print("CLIP Model - Skin Detection:")
    for _, row in clip_overall.iterrows():
        print(f"  {row['Dataset_Config']}: {row['In_Distribution_Pct']:.1f}%")

# LesionCLIP Analysis  
if len(lesionclip_overall) > 0:
    print(f"\nLesionCLIP Model - Diseased Skin Detection:")
    for _, row in lesionclip_overall.iterrows():
        print(f"  {row['Dataset_Config']}: {row['In_Distribution_Pct']:.1f}%")

# Bias Analysis
if len(clip_by_tone) > 0:
    print(f"\nCLIP Skin Tone Bias (All Dataset):")
    all_data_clip = clip_by_tone[clip_by_tone['Dataset_Config'] == 'All']
    if len(all_data_clip) > 0:
        sorted_clip = all_data_clip.sort_values('In_Distribution_Pct', ascending=False)
        best = sorted_clip.iloc[0]
        worst = sorted_clip.iloc[-1]
        print(f"  Best: FST {int(best['Skin_Tone'])} ({best['In_Distribution_Pct']:.1f}%)")
        print(f"  Worst: FST {int(worst['Skin_Tone'])} ({worst['In_Distribution_Pct']:.1f}%)")
        print(f"  Gap: {best['In_Distribution_Pct'] - worst['In_Distribution_Pct']:.1f}%")

if len(lesionclip_by_tone) > 0:
    print(f"\nLesionCLIP Skin Tone Bias (All Dataset):")
    all_data_lesion = lesionclip_by_tone[lesionclip_by_tone['Dataset_Config'] == 'All']
    if len(all_data_lesion) > 0:
        sorted_lesion = all_data_lesion.sort_values('In_Distribution_Pct', ascending=False)
        best = sorted_lesion.iloc[0]
        worst = sorted_lesion.iloc[-1]
        print(f"  Best: FST {int(best['Skin_Tone'])} ({best['In_Distribution_Pct']:.1f}%)")
        print(f"  Worst: FST {int(worst['Skin_Tone'])} ({worst['In_Distribution_Pct']:.1f}%)")
        print(f"  Gap: {best['In_Distribution_Pct'] - worst['In_Distribution_Pct']:.1f}%")

print(f"\nAll visualizations saved to: {output_directory}")
print(f"Generated 4 specific visualizations as requested")