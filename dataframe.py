import re
import pandas as pd
from collections import defaultdict

def parse_combined_log(filepath):
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
    balanced_accuracies_by_skin_tone = []  # NEW
    f1_scores = []  # NEW

    def parse_misclass_line(line):
        """Robust misclassification line parser"""
        if 'times' not in line:
            return None
        
        line = line.strip()
        
        # Method 1: Try standard regex with flexible arrow
        pattern = r'^(\w+(?:\s+\w+)*)\s*[^\w\s:]+\s*(\w+(?:\s+\w+)*)\s*:\s*(\d+)\s+times'
        match = re.search(pattern, line)
        if match:
            return match.groups()
        
        # Method 2: Split by colon and work backwards
        if ':' in line:
            before_colon, after_colon = line.rsplit(':', 1)
            
            count_match = re.search(r'(\d+)\s+times', after_colon)
            if count_match:
                count = int(count_match.group(1))
                
                # Find conditions around arrow
                arrow_match = re.search(r'^(\w+(?:\s+\w+)*)\s+[^\w\s]+\s+(\w+(?:\s+\w+)*)$', before_colon.strip())
                if arrow_match:
                    true_class, pred_class = arrow_match.groups()
                    return (true_class, pred_class, str(count))
        
        return None

    for block in model_blocks:
        if not block.strip():
            continue

        model_match = re.search(r'Python Filename:\s*(\S+)', block)
        if not model_match:
            continue
        model = model_match.group(1).replace('.py', '')
        
        datasets_match = re.search(r'Datasets:\s*(.+)', block)
        if not datasets_match:
            continue
        datasets = datasets_match.group(1)

        ### 1. Average Top-k Sensitivity
        for k in ['Top-1', 'Top-3', 'Top-5']:
            match = re.search(rf'Average {k} Sensitivity:\s*([\d.]+)%', block)
            if match:
                avg_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(match.group(1))
                })

        ### 2. Top-k Sensitivity per condition
        for k in ['Top-1', 'Top-3', 'Top-5']:
            for cond_match in re.findall(rf'Condition:\s*(.+?), {k} Sensitivity:\s*([\d.]+)%', block):
                condition, value = cond_match
                condition_sensitivities.append({
                    'Model': model,
                    'Datasets': datasets,
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
                    'Datasets': datasets,
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
                    'Datasets': datasets,
                    'Skin Tone': float(tone),
                    'Condition': condition,
                    'Metric': f'{k} Sensitivity',
                    'Value': float(value)
                })

        ### 5. Number of misclassified samples
        misclassified = re.search(r'Number of misclassified samples:\s*(\d+)', block)
        if misclassified:
            misclassifications.append({
                'Model': model,
                'Datasets': datasets,
                'Misclassified Samples': int(misclassified.group(1))
            })

        ### 6. ROBUST: Most common misclassifications (overall)
        lines = block.split('\n')
        in_overall_section = False
        
        for line in lines:
            if "MOST COMMON MISCLASSIFICATIONS (OVERALL)" in line:
                in_overall_section = True
                continue
            elif line.startswith("===") and in_overall_section:
                in_overall_section = False
                break
            elif in_overall_section:
                result = parse_misclass_line(line)
                if result:
                    true_class, pred_class, count = result
                    misclassifications_by_tone.append({
                        'Model': model,
                        'Datasets': datasets,
                        'Skin Tone': 'All',
                        'True Class': true_class,
                        'Predicted Class': pred_class,
                        'Count': int(count)
                    })

        ### 7. ROBUST: Most common misclassifications by skin tone
        current_skin_tone = None
        in_skin_tone_section = False
        
        for line in lines:
            if "MOST COMMON MISCLASSIFICATIONS BY SKIN TONE" in line:
                in_skin_tone_section = True
                continue
            elif line.startswith("===") and in_skin_tone_section:
                in_skin_tone_section = False
                break
            elif in_skin_tone_section:
                skin_tone_match = re.match(r'^Skin Tone ([\d.]+):', line)
                if skin_tone_match:
                    current_skin_tone = float(skin_tone_match.group(1))
                elif current_skin_tone is not None:
                    result = parse_misclass_line(line)
                    if result:
                        true_class, pred_class, count = result
                        misclassifications_by_tone.append({
                            'Model': model,
                            'Datasets': datasets,
                            'Skin Tone': current_skin_tone,
                            'True Class': true_class,
                            'Predicted Class': pred_class,
                            'Count': int(count)
                        })

        ### 8. NEW: Balanced Accuracy Parsing
        # Parse overall balanced accuracy
        balanced_acc_match = re.search(r'Overall Balanced Accuracy:\s*([\d.]+)%', block)
        if balanced_acc_match:
            balanced_accuracies_by_skin_tone.append({
                'Model': model,
                'Datasets': datasets,
                'Skin Tone': 'Overall',
                'Metric': 'Overall Balanced Accuracy',
                'Value': float(balanced_acc_match.group(1))
            })

        # Parse stratified balanced accuracies by skin tone
        current_skin_tone = None
        in_stratified_balanced_section = False
        
        for line in lines:
            if "STRATIFIED BALANCED ACCURACY" in line:
                in_stratified_balanced_section = True
                continue
            elif line.startswith("===") and in_stratified_balanced_section:
                in_stratified_balanced_section = False
                current_skin_tone = None
                continue
            elif in_stratified_balanced_section:
                skin_tone_match = re.match(r'^Skin Tone:\s*([\d.]+)', line)
                if skin_tone_match:
                    current_skin_tone = float(skin_tone_match.group(1))
                elif current_skin_tone is not None and "Overall Balanced Accuracy:" in line:
                    balanced_acc_val_match = re.search(r'Overall Balanced Accuracy:\s*([\d.]+)%', line)
                    if balanced_acc_val_match:
                        balanced_accuracies_by_skin_tone.append({
                            'Model': model,
                            'Datasets': datasets,
                            'Skin Tone': current_skin_tone,
                            'Metric': 'Overall Balanced Accuracy',
                            'Value': float(balanced_acc_val_match.group(1))
                        })

        ### 9. NEW: F1 Score Parsing
        # Parse overall F1 scores
        f1_macro_match = re.search(r'Overall F1 Score \(Macro\):\s*([\d.]+)%', block)
        if f1_macro_match:
            f1_scores.append({
                'Model': model,
                'Datasets': datasets,
                'Condition': 'Overall',
                'Metric': 'F1 Score (Macro)',
                'Value': float(f1_macro_match.group(1))
            })
        
        f1_weighted_match = re.search(r'Overall F1 Score \(Weighted\):\s*([\d.]+)%', block)
        if f1_weighted_match:
            f1_scores.append({
                'Model': model,
                'Datasets': datasets,
                'Condition': 'Overall',
                'Metric': 'F1 Score (Weighted)',
                'Value': float(f1_weighted_match.group(1))
            })

        # Parse per condition F1 scores
        in_f1_section = False
        for line in lines:
            if "Per Condition F1 Score:" in line:
                in_f1_section = True
                continue
            elif line.startswith("===") and in_f1_section:
                in_f1_section = False
                break
            elif in_f1_section and ":" in line:
                f1_condition_match = re.search(r'\s*(.+?):\s*([\d.]+)%', line)
                if f1_condition_match:
                    condition, value = f1_condition_match.groups()
                    f1_scores.append({
                        'Model': model,
                        'Datasets': datasets,
                        'Condition': condition.strip(),
                        'Metric': 'F1 Score',
                        'Value': float(value)
                    })

        # Parse stratified F1 scores by skin tone
        current_skin_tone = None
        in_stratified_f1_section = False
        in_per_condition_subsection = False
        
        for line in lines:
            if "STRATIFIED F1 SCORE" in line:
                in_stratified_f1_section = True
                continue
            elif line.startswith("===") and in_stratified_f1_section:
                in_stratified_f1_section = False
                current_skin_tone = None
                in_per_condition_subsection = False
                continue
            elif in_stratified_f1_section:
                skin_tone_match = re.match(r'^Skin Tone:\s*([\d.]+)', line)
                if skin_tone_match:
                    current_skin_tone = float(skin_tone_match.group(1))
                    in_per_condition_subsection = False
                elif current_skin_tone is not None:
                    # Parse overall F1 scores for this skin tone
                    f1_macro_stratified_match = re.search(r'Overall F1 Score \(Macro\):\s*([\d.]+)%', line)
                    if f1_macro_stratified_match:
                        f1_scores.append({
                            'Model': model,
                            'Datasets': datasets,
                            'Condition': 'Overall',
                            'Skin_Tone': current_skin_tone,
                            'Metric': 'F1 Score (Macro) Stratified',
                            'Value': float(f1_macro_stratified_match.group(1))
                        })
                    
                    f1_weighted_stratified_match = re.search(r'Overall F1 Score \(Weighted\):\s*([\d.]+)%', line)
                    if f1_weighted_stratified_match:
                        f1_scores.append({
                            'Model': model,
                            'Datasets': datasets,
                            'Condition': 'Overall',
                            'Skin_Tone': current_skin_tone,
                            'Metric': 'F1 Score (Weighted) Stratified',
                            'Value': float(f1_weighted_stratified_match.group(1))
                        })
                    
                    if "Per Condition:" in line:
                        in_per_condition_subsection = True
                    elif in_per_condition_subsection and ":" in line:
                        f1_stratified_match = re.search(r'\s*(.+?):\s*([\d.]+)%', line)
                        if f1_stratified_match:
                            condition, value = f1_stratified_match.groups()
                            f1_scores.append({
                                'Model': model,
                                'Datasets': datasets,
                                'Condition': condition.strip(),
                                'Skin_Tone': current_skin_tone,
                                'Metric': 'F1 Score (Stratified)',
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
        'BalancedAccuraciesBySkinTone': pd.DataFrame(balanced_accuracies_by_skin_tone),  # NEW
        'F1Scores': pd.DataFrame(f1_scores)  # NEW
    }


# Example usage:
if __name__ == "__main__":
    log_path = "combined_all.txt"
    df_dict = parse_combined_log(log_path)

    # Access example:
    print("Available DataFrames:")
    for key, df in df_dict.items():
        print(f"  {key}: {len(df)} records")
    
    if not df_dict['BalancedAccuraciesBySkinTone'].empty:
        print("\nBalanced Accuracies sample:")
        print(df_dict['BalancedAccuraciesBySkinTone'].head())
    
    if not df_dict['F1Scores'].empty:
        print("\nF1 Scores sample:")
        print(df_dict['F1Scores'].head())