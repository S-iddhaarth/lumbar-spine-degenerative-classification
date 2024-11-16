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

def extract_sagittal(dataset,model,level,lr=5):
    root = r'./Data'
    device = 'cuda'
    out_dir = os.path.join(root,"preprocessed_image")
    os.makedirs(out_dir,exist_ok=True)
    problems = []
    for batch in tqdm(dataset,total=len(dataset)):
        image,study_id,series_id = batch
        segment = defaultdict(list)
        for channel in range(image.shape[1]):
            input_img = image[:,channel,:,:].unsqueeze(1)
            seg,prob = preprocess.extract_regions_from_image(
                model,input_img.to(device),study_id,series_id)
            if prob:
                print(f'problem has occured {prob}')
                problems.extend(prob)
            for i,j in enumerate(seg):
                segment[i].append(j)
        for ch in range(len(segment)):
            img = segment[ch]
            img = torch.cat(img,dim=1)
            for bh in range(image.shape[0]):
                output,study_id_out,series_id_out = img[bh],str(int(study_id[bh])),str(int(series_id[bh]))
                output_file_path = os.path.join(out_dir,study_id_out,series_id_out,f'{level[ch]}')
                os.makedirs(output_file_path,exist_ok=True)
                
                output = output - output.min()
                output = output / output.max()
                output = (output * 255).byte()
                
                for im_ch in range(lr):
                    img_np = output[im_ch].cpu().numpy()
                    img_pil = Image.fromarray(img_np)
                    img_pil.save(os.path.join(output_file_path, f'{im_ch}.png'))
  