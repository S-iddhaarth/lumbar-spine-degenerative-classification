"""
This module contains torch dataloader for RSNA-2024
Lumbar-spine-degenerative-classification challenge
"""
import os
import natsort
import torch
from torch.utils.data import Dataset
import pydicom
import numpy as np
from utils.utility import get_elements
import json
from PIL import Image
import torch
import torchvision.transforms as T
from natsort import natsorted
import utils.utility as utils
class segmentation_Dataset(Dataset):
    def __init__(self, df,image_dir,transform=None,spinal=False):
        self.df=df
        self.img_dir = image_dir
        self.transform = transform
        
        if self.transform is None:
            self.transform = T.Compose([
                T.Resize((256,256)),
                T.ToTensor(),
            ])
        self.spinal = spinal
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        image,mask,left_right =self.get_images(idx)
        image= self.transform(image) 
        
        image = image.clone().detach().float()
        mask = torch.tensor(mask,dtype=torch.long)
        
        if not self.spinal:
            left_right = torch.tensor(left_right,dtype=torch.long)
            data = (image,mask,left_right)
        else:
            data = (image,mask)
        return data
    
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
        if not self.spinal:
            left_right=float(row['condition_encoded'])
            data = (rgb_image_pil, mask,left_right)
        else:
            data = (rgb_image_pil,mask,None)
        return data
    
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




class subarticular_Dataset(Dataset):
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
        image,mask,level =self.get_images(idx)
        image= self.transform(image) 
        
        image=image.clone().detach().float()
        mask = torch.tensor(mask,dtype=torch.long)
        level = torch.tensor(level,dtype=torch.long)
        
        return image,mask,level
    
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

        coor=np.array(row.loc['x_level_Right':'y_level_Left'].tolist(),dtype=float)
        coor[0::2],coor[1::2]= coor[0::2]/width ,coor[1::2]/height #coor in 0-1
        
        # Create a segmentation mask for the image (5 spinal levels)
        mask = self.create_segmentation_mask(coor, img_shape=(256, 256))

        level=float(row['level_array'].index(1))
        
        return rgb_image_pil, mask,level
    
    def create_segmentation_mask(self, label, img_shape=(256, 256), box_size=10):
        mask = np.zeros(img_shape, dtype=np.float32)
        width, height = img_shape

        # Iterate over spinal level coordinates (x1, y1, x2, y2, ...)
        for i in range(0, len(label), 2):
            x = int(label[i] * width)  # x coordinate scaled to image size
            y = int(label[i + 1] * height)  # y coordinate scaled to image size

            # Assign class label for each spinal level (1 to 5)
            class_label = (i // 2) + 1

            # Draw a small square around the (x, y) point
            x_min = max(0, x - box_size // 2)
            x_max = min(width, x + box_size // 2)
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
       

class classification_loader(Dataset):
    def __init__(self,df,root,side=None):
        self.df = df
        self.side = side
        self.root = root
        self.normal,self.moderate,self.severe = df.columns[-3],df.columns[-2],df.columns[-1]
        self.transform = T.Compose(
            [T.ToTensor()]
        )
        # self.mapping = {
        #     'L1_L2':[1,0,0,0,0],
        #     'L2_L3':[0,1,0,0,0],
        #     'L3_L4':[0,0,1,0,0],
        #     'L4_L5':[0,0,0,1,0],
        #     'L5_S1':[0,0,0,0,1]
        # }
        
        self.mapping = {
            'L1_L2':0,
            'L2_L3':1,
            'L3_L4':2,
            'L4_L5':3,
            'L5_S1':4
        }
    def __len__(self):
        return len(self.df)
    def __getitem__(self,index):
        series = self.df.iloc[index]
        study_id = str(series["study_id"])
        series_id = str(series["series_id"])
        level = f"{'_'.join(series['level'].split('/'))}"
        if series[self.normal]:
            label = 0
        elif series[self.moderate]:
            label = 1
        else:
            label = 2
        # label = [int(series[self.normal]),int(series[self.moderate]),int(series[self.severe])]
        image_dir = os.path.join(
            self.root,study_id,series_id,level
        )
        images = []
        for i in os.listdir(image_dir):
            images.append(np.array(Image.open(os.path.join(image_dir,i))))
        image = self.transform(np.stack(images,axis=0)).permute(1,0,2)
        # if self.side == "rigth":
            
        #     side = "right"
            
        #     path = os.path.join(
        #         self.root,study_id,series_id,
        #         side,level
        #     )
        # elif self.side =="left":
        #     side = "right"
        #     level = f"{'_'.join(series['level'].split('/'))}.pt"
        #     path = os.path.join(
        #         self.root,study_id,series_id,
        #         side,level
        #     )
        # else:
        #     level = f"{'_'.join(series['level'].split('/'))}.pt"
        #     path = os.path.join(
        #         self.root,study_id,series_id,level
        #     )
        return image,torch.tensor(label,dtype=torch.long),torch.tensor(self.mapping[level])
    
class extractionDataset(Dataset):
    def __init__(self,df,root):
        self.transform = T.Compose([
                T.Resize((256,256)),
                T.ToTensor(),
            ])
        self.root = root
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self,index):
        series = self.data.iloc[index]
        study_id,series_id, series_descriptions = series['study_id'],series['series_id'],series['series_description']
        path = os.path.join(self.root,"train_images",str(study_id),str(series_id))
        dirs = natsorted(os.listdir(path))
        ele = utils.get_elements(len(dirs),5)
        image_list = []
        for i in ele:
            file_path = os.path.join(path,dirs[i])
            dicom = pydicom.dcmread(file_path)
            image = dicom.pixel_array
            image_normalized = np.interp(image, (0, 1810), (0, 255)).astype(np.uint8)
            rgb_image_pil = Image.fromarray(image_normalized)
            image_list.append(self.transform(rgb_image_pil))
        image = torch.cat(image_list)
        return image,study_id,series_id