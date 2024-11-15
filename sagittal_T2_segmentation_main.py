import os
import pandas as pd

import torch.optim as optim
from torchvision import models 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import utils.segmentation_utility as helper
import data_loader
import utils.visualize as visuals
import models.segment as model
import trainers.segmentation_trainer as trainer
from sklearn.model_selection import train_test_split


def main():
    base_path = 'Data/'

    train_label = os.path.join(base_path,'train_label_coordinates.csv')
    train_series = os.path.join(base_path,'train_series_descriptions.csv')
    train_ = os.path.join(base_path,'train.csv')
    img_train_path = os.path.join(base_path,'train_images')

    test_series=os.path.join(base_path,'test_series_descriptions.csv')
    img_test_path = os.path.join(base_path,'test_images')
    
    train_df = pd.read_csv(train_)
    series_df = pd.read_csv(train_series)
    label_df = pd.read_csv(train_label)
    df = pd.merge(label_df, series_df , on=['study_id','series_id'])
    df = pd.merge(df,train_df,on = 'study_id')
    df_spinal, df_neural,df_subarticular = helper.reconstruct_df(df)

    df_spinal_xy =helper.xy_spinal_neural(df_neural)
    
    spinal_train_df, spinal_val_df=train_test_split(df_spinal_xy,test_size=0.2,random_state=208)
    print(f"size for train:{len(spinal_train_df)}")
    print(f"size for validation:{len(spinal_val_df)}")

    spinal_train_dataset= data_loader.segmentation_Dataset(spinal_train_df,img_train_path,transform=None,spinal=True)
    spinal_train_dataset= data_loader.segmentation_Dataset(spinal_val_df,img_train_path,transform=None,spinal=True)

    spinal_train_loader = DataLoader(spinal_train_dataset,batch_size=32,shuffle=True,num_workers=7,pin_memory=True,prefetch_factor=4 )
    spinal_val_loader=DataLoader(spinal_train_dataset,batch_size=32,shuffle=False,num_workers=7,pin_memory=True,prefetch_factor=4)      
    
    model_mask = model.UNetMobileNetV2(in_channels=1, out_channels=6, pretrained=True)
    model_name='spinal'

    optimizer_mask = optim.AdamW(model_mask.parameters(), lr=0.001)

    scheduler_mask = optim.lr_scheduler.StepLR(optimizer_mask, step_size=10, gamma=0.1) 

    model_mask,train_loss_set, test_loss_set = trainer.train_and_evaluate_v0(model_mask,  
                                                           spinal_train_loader, spinal_val_loader,
                                                           optimizer_mask, 
                                                           scheduler_mask,
                                                           model_name,
                                                           num_epochs=30)
if __name__ == '__main__':
    main()