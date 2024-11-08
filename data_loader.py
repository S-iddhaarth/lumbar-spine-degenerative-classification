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

class DataLoader(Dataset):
    """
    
    """
    def __init__(self, config: dict, transform=None, test=False, resize=None) -> None:
        self.test = False
        self.series = pl.read_csv(config["series"])
        self.transform = transform
        self.data_points = []
        self.resize = resize
        if not test:
            self.label = pl.read_csv(config["labels"])
            self.label = self.label.to_dummies(columns=self.label.columns[1:])
            self.coordinates = pl.read_csv(config["coordinates"])

            for j in tqdm(self.series.iter_rows(named=True), 
                        total=self.series.shape[0], desc="Processing items"):

                path = os.path.join(config["images"], str(j['study_id']), str(j['series_id']))
                clms_df = self.coordinates.filter(
                    (pl.col("study_id") == j['study_id']) &
                    (pl.col("series_id") == j['series_id'])
                ).select(["condition", "level", "instance_number", "x", "y"]) 

                clms = []
                for l in clms_df.iter_rows(named=True):
                    clms.append((
                        ("_".join(l["condition"].split(" ")))
                        + "_" + "_".join(l["level"].split('/'))
                    ).lower())
                clms = [x + y for x in clms for y in ["_Normal/Mild", "_Moderate", "_Severe"]] 

                label = self.label.filter(
                    pl.col("study_id") == j['study_id']
                ).select(clms)
                try:
                    label = np.array(label).reshape(-1, 3)
                except Exception as _:
                    print(f'path - {path}')
                    continue
                data = (path, label, clms_df.select(
                    ["instance_number", "x", "y"]), j["series_description"]
                )
                self.data_points.append(data)
        else: 
            for j in self.series.iter_rows(named=True):
                path = os.path.join(config["images"], str(j['study_id']),
                                    str(j['series_id']),j["series_description"]) 

    def __len__(self):
        return len(self.data_points)
    
    def __getitem__(self, idx):
        pth, label, coords, axis = self.data_points[idx]
        ordered = natsort.natsorted(os.listdir(pth))
        dicoms = [pydicom.read_file(os.path.join(pth, i)) for i in ordered]
        coo = np.zeros_like(label,dtype=np.float16)
        for idx, i in enumerate(coords.iter_rows(named=True)):

            H, W,x,y, z = get_z(dicoms[i["instance_number"]], i["x"], i["y"])
            if self.resize:
                x = x * (self.resize / H)
                y = y * (self.resize / W)
            coo[idx, 0] = x
            coo[idx, 1] = y
            coo[idx, 2] = z
        if self.transform:
            dicoms = [self.transform(i.pixel_array.astype(np.float32)) for i in dicoms]
        else:
            dicoms = [torch.tensor(i.pixel_array) for i in dicoms]
        dicoms = np.stack(dicoms)
        return {"axis": axis, "image": dicoms.squeeze(1),
                "label": torch.tensor(label), "coords": torch.tensor(coo)}
        
class DataLoader_np(Dataset):
    def __init__(self, config: dict, transform=None, test=False, resize=None) -> None:
        self.test = False
        self.series = pl.read_csv(config["series"])
        self.transform = transform
        self.data_points = []
        self.resize = resize
        if not test:
            self.label = pl.read_csv(config["labels"])
            self.label = self.label.to_dummies(columns=self.label.columns[1:])
            self.coordinates = pl.read_csv(config["coordinates"])

            for j in tqdm(self.series.iter_rows(named=True),
                        total=self.series.shape[0], desc="Processing items"):
                path = os.path.join(config["images"], str(j['study_id']), str(j['series_id']))
                clms_df = self.coordinates.filter(
                    (pl.col("study_id") == j['study_id']) &
                    (pl.col("series_id") == j['series_id'])
                ).select(["condition", "level", "instance_number", "x", "y"])
                
                clms = []
                for l in clms_df.iter_rows(named=True):
                    clms.append((
                        ("_".join(l["condition"].split(" ")))
                        + "_" + "_".join(l["level"].split('/'))
                    ).lower())
                clms = [x + y for x in clms for y in ["_Normal/Mild", "_Moderate", "_Severe"]]
                
                label = self.label.filter(
                    pl.col("study_id") == j['study_id']   
                ).select(clms)
                try:
                    label = np.array(label).reshape(-1, 3)
                except Exception as _:
                    continue
                data = (path, label, np.array(
                    clms_df.select(["x", "y","instance_number"])),
                    j["series_description"]
                    )
                self.data_points.append(data)
        else:
            for j in self.series.iter_rows(named=True):
                path = os.path.join(config["images"], str(j['study_id']), str(j['series_id']))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        if not self.test:
            pth, label, coords, axis = self.data_points[idx]
        else:
            pth, axis = self.data_points[idx]
            

        ordered = natsort.natsorted(os.listdir(pth))
        dicoms = [pydicom.read_file(os.path.join(pth, i)) for i in ordered]

        if self.transform:
            dicoms = [self.transform(i.pixel_array.astype(np.float32)) for i in dicoms]
        else: 
            dicoms = [torch.tensor(i.pixel_array) for i in dicoms]
        dicoms = np.stack(dicoms)
        
        if not self.test:
            data = {"axis": axis, "image": dicoms,
                "label": torch.tensor(label), "coords": torch.tensor(coords)}
        else:
            data = {"axis": axis, "image": dicoms,
                }
        return data
    
class naive_loader(Dataset):
    def __init__(self,path:str,ch,transform) -> None:
        with open(path,"r") as fl:
            self.data = json.load(fl)
        self.data_list = list(self.data.keys())
        self.transform = transform
        self.ch = ch
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
            images.extend(di)
        di = torch.stack(images)
        
        return (di.type(torch.float32).squeeze(dim=1),torch.tensor(labels,dtype=float))