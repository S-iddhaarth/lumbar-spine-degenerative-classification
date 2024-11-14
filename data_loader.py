"""
This module contains torch dataloader for RSNA-2024
Lumbar-spine-degenerative-classification challenge
"""
import os
import natsort
import torch
from torch.utils.data import Dataset
import pydicom
import polars as pl
import numpy as np
from Preprocessing.utility import get_z
from tqdm import tqdm
from utils.utility import get_elements
import json
from PIL import Image
import torch
import torchvision.transforms as T

class neural_Dataset(Dataset):
    def __init__(self, df,image_dir,transform=None):
        self.df=df
        self.img_dir = image_dir
        self.transform = transform
        
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((256,256)),
                T.ToTensor(),
            ])
                
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        image,mask,left_right =self.get_images(idx)
        image= self.transform(image) 
        
        image = image.clone().detach().float()
        mask = torch.tensor(mask,dtype=torch.long)
        left_right = torch.tensor(left_right,dtype=torch.long)
        
        return image,mask,left_right
    
    def get_images(self,idx):
        row = self.df.iloc[idx]
        dicom_path = os.path.join(self.img_dir, str(row['study_id']),str(row['series_id']),f"{str(row['instance_number'])}.dcm")
        dicom = pydicom.dcmread(dicom_path)
        image = dicom.pixel_array
        # Normalize and convert to RGB
        image_normalized = np.interp(image, (0, 1810), (0, 255)).astype(np.uint8)
        #rgb_image = np.stack([image_normalized] * 3, axis=-1)
        # Convert NumPy array to PIL Image for cropping
        rgb_image_pil = Image.fromarray(image_normalized)

        width,height =rgb_image_pil.size

        coor=np.array(row.loc['x_level_l1_l2':'y_level_l5_s1'].tolist(),dtype=float)
        coor[0::2],coor[1::2]= coor[0::2]/width ,coor[1::2]/height #coor in 0-1
        
        # Create a segmentation mask for the image (5 spinal levels)
        mask = self.create_segmentation_mask(coor, img_shape=(256, 256))

        left_right=float(row['condition_encoded'])
        
        return rgb_image_pil, mask,left_right
    
    def create_segmentation_mask(self, label, img_shape=(256, 256), box_size=15):
        mask = np.zeros(img_shape, dtype=np.float32)
        width, height = img_shape

        # Iterate over spinal level coordinates (x1, y1, x2, y2, ...)
        for i in range(0, len(label), 2):
            x = int(label[i] * width)  # x coordinate scaled to image size
            y = int(label[i + 1] * height)  # y coordinate scaled to image size

            # Assign class label for each spinal level (1 to 5)
            class_label = (i // 2) + 1

            # Draw a small square around the (x, y) point
            x_min = max(0, x - (box_size-5) // 2)
            x_max = min(width, x + (box_size-5) // 2)
            y_min = max(0, y - box_size // 2)
            y_max = min(height, y + box_size // 2)

            # Fill the mask with the class label
            mask[y_min:y_max, x_min:x_max] = class_label

        return mask


class naive_loader(Dataset):
    def __init__(self,path:str,ch,transform,train=True,augment=None) -> None:
        with open(path,"r") as fl:
            data = json.load(fl)
        if train:
            self.data = data["train"]
        else:
            self.data = data["test"]
        self.data_list = list(self.data.keys())
        self.transform = transform
        self.ch = ch
        self.order = None
        self.augment = augment
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, index) :
        paths,labels = self.data[self.data_list[index]]
        images = []
        for path in paths:
            dirs = natsort.natsorted(os.listdir(path[0]))
            n = len(dirs)
            n = get_elements(n,self.ch)
            di = [pydicom.dcmread(os.path.join(path[0],dirs[i])) for i in n]
            di = [self.transform(Image.fromarray(i.pixel_array.astype(np.int16))) for i in di]
            if self.augment:
                di = [self.augment(Image.fromarray(i.numpy()[0,0,:])) for i in di]
            images.extend(di)
        di = torch.stack(images)
        self.order = list(labels.keys())
        return (di.type(torch.float32).squeeze(dim=1),torch.tensor(list(labels.values()),dtype=torch.float32)
                ,torch.tensor(int(self.data_list[index]),dtype=int))