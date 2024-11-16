import Preprocessing.generate_samples as generator
import sys
sys.path.append(r'../')
import models.segment as segmentation_models
import torch
import pandas as pd    
import os
import utils.segmentation_utility as helper
from sklearn.model_selection import train_test_split
import data_loader
from torch.utils.data import DataLoader,Dataset
import utils.utility as utils
import pydicom
import torchvision.transforms as T
from natsort import natsorted
import numpy as np
from PIL import Image
import Preprocessing.utility as preprocess
from collections import defaultdict
from tqdm import tqdm
import data_loader
def main(sagittal_t1_path,sagittal_t2_path,axial_t2_path):
    device = 'cuda'
    root = 'Data/'
    sagittal_t1_model = segmentation_models.UNetMobileNetV2(in_channels=1, out_channels=6)
    sagittal_t1_model.load_state_dict(torch.load(sagittal_t1_path))
    sagittal_t1_model.to(device)
    sagittal_t1_model.eval()

    sagittal_t2_model = segmentation_models.UNetMobileNetV2(in_channels=1, out_channels=6)
    sagittal_t2_model.load_state_dict(torch.load(sagittal_t2_path))
    sagittal_t2_model.to(device)
    sagittal_t2_model.eval()
    
    axial_t2_model = segmentation_models.UNetVGG16(in_channels=1, out_channels=3)
    axial_t2_model.load_state_dict(torch.load(axial_t2_path))
    axial_t2_model.to(device)
    axial_t2_model.eval()
    
    extract_df = pd.read_csv(os.path.join(root, "train_series_descriptions.csv"))

    # Extract unique values from the 'series_description' column
    unique_values = extract_df['series_description'].unique()

    # Initialize empty dictionaries to store the DataFrames
    dfs = {}

    # Split the DataFrame based on unique values and store them in the dictionary
    for value in unique_values:
        dfs[value.replace("/", "_").replace(" ", "_")] = extract_df[extract_df['series_description'] == value]

    # Access the DataFrames using their keys
    sagittal_t1_df = dfs['Sagittal_T1']
    axial_t2_df = dfs['Axial_T2']
    sagittal_t2_df = dfs['Sagittal_T2_STIR']
    
    sagittal_t1_dataset = data_loader.extractionDataset(sagittal_t1_df,root)
    sagittal_t2_dataset = data_loader.extractionDataset(sagittal_t2_df,root)
    axial_t2_dataset = data_loader.extractionDataset(axial_t2_df,root)
    
    sagittal_t2_loader = DataLoader(
        sagittal_t2_dataset,batch_size=32,num_workers=7,pin_memory=True,
        persistent_workers=True,prefetch_factor=8
        )
    # generator.extract_sagittal(
    #     sagittal_t1_loader,sagittal_t1_model,
    #     {0:"L1_L2",1:"L2_L3",2:"L3_L4",3:"L4_L5",4:"L5_S1"}
    # )
    
    generator.extract_sagittal(
        sagittal_t2_loader,sagittal_t2_model,
        {0:"L1_L2",1:"L2_L3",2:"L3_L4",3:"L4_L5",4:"L5_S1"}
    )
if __name__ == '__main__':
    sagittal_t1_path = r'weights/sagittal_T1_segmentation/model1/lo_last_neural.pth'
    sagittal_t2_path = r'weights/sagittal_T2_segmentation/model1/lo_last_spinal.pth'
    axial_t2_path = r'weights/axial_T2_segmentation/model2/mask_last_subarticular.pth'
    main(sagittal_t1_path=sagittal_t1_path,sagittal_t2_path=sagittal_t2_path,axial_t2_path=axial_t2_path)