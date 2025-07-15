from zip_dataset import *
import torchvision.transforms as transforms
import torch
from torchvision import models
from metricsFunctions import *
from Baseline import *
from TestFunction import *
import matplotlib.pyplot as plt
from xai import *

### SEEDS, DEVICE AND LOG FILE  ###

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs('Logs', exist_ok=True)
log_file = f"Logs/dermie_experiment_{experiment_timestamp}.txt"

def save_experiment_log(data, file_path=log_file):
    with open(file_path, 'w', encoding='utf-8') as f:
        for key, value in data.items():
            f.write(f"{key}: {value}\n")

def save_plot_and_return_path(fig, filename_base):
    filename = f"Logs/{filename_base}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

experiment_data = {}
experiment_data['Python Filename'] = os.path.basename(__file__)
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


### LOAD DATA ###

stratification_strategy = 'Diagnosis'  # 'stratify_col' -> Ensure all conditions and skin tones are in both train and test

dermie_metadata_train, dermie_metadata_test, dermie_metadata_val, images_dermie = load_dataset(project_dir=project_dir,
                                                                                               path_folder=r'Data/dermie_data', 
                                                                                               images_dir='master_data_june_7_2025.zip',
                                                                                               metadata_dir='master_data_june_7_2025.csv',
                                                                                               stratification_strategy=stratification_strategy)

pad_metadata_train, pad_metadata_test, pad_metadata_val, images_pad = load_dataset(project_dir=project_dir,
                                                                                   path_folder=r'Data/padufes', 
                                                                                   images_dir='padufes_images.zip',
                                                                                   metadata_dir='padufes_metadata_clean.csv',
                                                                                   stratification_strategy=stratification_strategy)

scin_metadata_train, scin_metadata_test, scin_metadata_val, images_scin = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/scin', 
                                                                                       images_dir='scin_images.zip',
                                                                                       metadata_dir='scin_metadata_clean.csv',
                                                                                       stratification_strategy=stratification_strategy)

fitz17_metadata_train, fitz17_metadata_test, fitz17_metadata_val, images_fitz17 = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/fitz17k', 
                                                                                       images_dir='fitzpatrick17k_images.zip',
                                                                                       metadata_dir='fitzpatrick17k_metadata_clean.csv',
                                                                                       stratification_strategy=stratification_strategy)

india_metadata_train, india_metadata_test, india_metadata_val, images_india = load_dataset(project_dir=project_dir,
                                                                                       path_folder=r'Data/india_data', 
                                                                                       images_dir='india_images.zip',
                                                                                       metadata_dir='india_metadata_final.csv',
                                                                                       stratification_strategy=stratification_strategy)

experiment_data['Datasets'] = 'Dermie + Padufes + Fitzpatrick17k + India'


### CREATE DATASETS AND DATALOADERS ###

transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomAffine(degrees=10, shear= (-10,10,-10,10)),
])

transformations_val_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def visualize_skin_tone_distribution(metadata_dict, save_path=None):
    """
    Visualize skin tone distribution for each dataset as bar graphs
    
    Args:
        metadata_dict: Dictionary with dataset names as keys and metadata DataFrames as values
        save_path: Optional path to save the figure
    """
    
    fst_color_map = {
        'I': '#F5D5A0',
        'II': '#E4B589',
        'III': '#D1A479',
        'IV': '#C0874F',
        'V': '#A56635',
        'VI': '#4C2C27'
    }
    
    # Define the order of Fitzpatrick skin types
    fst_order = ['I', 'II', 'III', 'IV', 'V', 'VI']
    
    n_datasets = len(metadata_dict)
    
    # Create figure with subplots with more spacing
    fig, axes = plt.subplots(2, 3, figsize=(20, 24))
    fig.subplots_adjust(hspace=0.5, wspace=0.3, top=0.88, bottom=0.03)  # Maximum space between rows
    axes = axes.flatten()
    
    for i, (dataset_name, metadata) in enumerate(metadata_dict.items()):
        if i >= len(axes):
            break
            
        # Count Fitzpatrick skin types
        fst_counts = metadata['Fitzpatrick'].value_counts()
        
        # Ensure all skin types are represented (fill missing with 0)
        fst_data = []
        colors = []
        labels = []
        
        for fst in fst_order:
            count = fst_counts.get(fst, 0)
            if count > 0:  # Only include skin types that exist in the dataset
                fst_data.append(count)
                colors.append(fst_color_map[fst])
                labels.append(fst)
        
        # Create bar plot
        bars = axes[i].bar(labels, fst_data, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        # Customize the plot
        axes[i].set_title(f'{dataset_name} Dataset\n(Total: {len(metadata)} samples)', 
                         fontsize=13, fontweight='bold', pad=15)
        axes[i].set_xlabel('Fitzpatrick Skin Type', fontsize=12)
        axes[i].set_ylabel('Number of Samples', fontsize=12)
        axes[i].grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, fst_data):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(fst_data)*0.01,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # Set y-axis to start from 0
        axes[i].set_ylim(0, max(fst_data) * 1.1)
        
        # Add percentage labels
        total_samples = sum(fst_data)
        percentages = [f'{(count/total_samples)*100:.1f}%' for count in fst_data]
        
        # Add percentage as secondary labels
        for j, (bar, pct) in enumerate(zip(bars, percentages)):
            axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height()/2,
                        pct, ha='center', va='center', fontsize=10, 
                        color='white', fontweight='bold')
    
    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches=None, pad_inches=0)
    
    plt.show()
    return fig

def visualize_combined_distribution(metadata_dict, save_path=None):
    """
    Create a combined view showing all datasets in one plot
    """
    
    fst_color_map = {
        'I': '#F5D5A0',
        'II': '#E4B589',
        'III': '#D1A479',
        'IV': '#C0874F',
        'V': '#A56635',
        'VI': '#4C2C27'
    }
    
    fst_order = ['I', 'II', 'III', 'IV', 'V', 'VI']
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Prepare data for grouped bar chart
    datasets = list(metadata_dict.keys())
    x = np.arange(len(fst_order))
    width = 0.15
    
    for i, (dataset_name, metadata) in enumerate(metadata_dict.items()):
        fst_counts = metadata['Fitzpatrick'].value_counts()
        counts = [fst_counts.get(fst, 0) for fst in fst_order]
        
        # Create bars with slight offset for each dataset
        bars = ax.bar(x + i*width, counts, width, label=dataset_name, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, counts):
            if value > 0:
                ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                       f'{value}', ha='center', va='bottom', fontsize=9)
    
    # Customize the plot
    ax.set_title('Skin Tone Distribution Across All Datasets', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Fitzpatrick Skin Type', fontsize=14)
    ax.set_ylabel('Number of Samples', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(fst_order)
    ax.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return fig

# Add this to your main code after loading all datasets:

# Combine ALL data (train + val + test) for each dataset
metadata_dict = {
    'Dermie': pd.concat([dermie_metadata_train, dermie_metadata_val, dermie_metadata_test], ignore_index=True),
    'PAD': pd.concat([pad_metadata_train, pad_metadata_val, pad_metadata_test], ignore_index=True),
    'Fitzpatrick17k': pd.concat([fitz17_metadata_train, fitz17_metadata_val, fitz17_metadata_test], ignore_index=True),
    'India': pd.concat([india_metadata_train, india_metadata_val, india_metadata_test], ignore_index=True),
    'SCIN': pd.concat([scin_metadata_train, scin_metadata_val, scin_metadata_test], ignore_index=True)
}

# Create individual dataset visualizations
fig1 = visualize_skin_tone_distribution(metadata_dict, 
                                       save_path=f"Logs/skin_tone_distribution_{experiment_timestamp}.png")

# Save plot path to experiment data
experiment_data['Skin Tone Distribution Plot'] = save_plot_and_return_path(fig1, "skin_tone_distribution")

# Print summary statistics
print("\n=== SKIN TONE DISTRIBUTION SUMMARY ===")
for dataset_name, metadata in metadata_dict.items():
    print(f"\n{dataset_name}:")
    fst_counts = metadata['Fitzpatrick'].value_counts().sort_index()
    total = len(metadata)
    for fst, count in fst_counts.items():
        percentage = (count/total)*100
        print(f"  {fst}: {count:4d} samples ({percentage:5.1f}%)")
    print(f"  Total: {total} samples")