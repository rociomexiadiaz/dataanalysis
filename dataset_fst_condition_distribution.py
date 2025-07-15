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


def save_plot_and_return_path(fig, filename_base):
    filename = f"Logs/{filename_base}_{experiment_timestamp}.png"
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filename

train_set = MultipleDatasets([dermie_metadata_train, dermie_metadata_val, dermie_metadata_test], [images_dermie, images_dermie, images_dermie], transform=transformations_val_test) 
fig_train = visualise(train_set)
_ = save_plot_and_return_path(fig_train,'dermie')

train_set = MultipleDatasets([india_metadata_train, india_metadata_val, india_metadata_test], [images_india, images_india, images_india], transform=transformations_val_test) 
fig_train = visualise(train_set)
_ = save_plot_and_return_path(fig_train,'india')

train_set = MultipleDatasets([scin_metadata_train, scin_metadata_val, scin_metadata_test], [images_scin, images_scin, images_scin], transform=transformations_val_test) 
fig_train = visualise(train_set)
_ = save_plot_and_return_path(fig_train,'scin')

train_set = MultipleDatasets([pad_metadata_train, pad_metadata_val, pad_metadata_test], [images_pad, images_pad, images_pad], transform=transformations_val_test) 
fig_train = visualise(train_set)
_ = save_plot_and_return_path(fig_train,'pad')

train_set = MultipleDatasets([fitz17_metadata_train, fitz17_metadata_val, fitz17_metadata_test], [images_fitz17, images_fitz17, images_fitz17], transform=transformations_val_test) 
fig_train = visualise(train_set)
_ = save_plot_and_return_path(fig_train,'fitz17')