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
import trainers.axial_segmentation_trainer as trainer
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

    df_subarticular_xy =helper.xy_subarticular(df_subarticular)
    
    subarticular_train_df, subarticular_val_df=train_test_split(df_subarticular_xy,test_size=0.2,random_state=208)
    print(f"size for train:{len(subarticular_train_df)}")
    print(f"size for validation:{len(subarticular_val_df)}")

    subarticular_train_dataset= data_loader.subarticular_Dataset(subarticular_train_df,img_train_path,transform=None)
    subarticular_val_dataset= data_loader.subarticular_Dataset(subarticular_val_df,img_train_path,transform=None)

    subarticular_train_loader = DataLoader(subarticular_train_dataset,batch_size=32,shuffle=True,num_workers=7,pin_memory=True,prefetch_factor=4 )
    subarticular_val_loader = DataLoader(subarticular_val_dataset,batch_size=32,shuffle=False,num_workers=7,pin_memory=True,prefetch_factor=4)      
    
    model_mask = model.UNetVGG16(in_channels=1, out_channels=3,pretrained= models.VGG16_Weights.IMAGENET1K_V1)
    model_name='neural'

    
    model_rl = model.subarticular_resnet(num_classes=5)

    optimizer_mask = optim.Adam(model_mask.parameters(), lr=0.001)

    optimizer_leris = optim.Adam(model_rl.parameters(), lr=0.001)


    scheduler_mask = optim.lr_scheduler.StepLR(optimizer_mask, step_size=10, gamma=0.1) 
    scheduler_leris = optim.lr_scheduler.StepLR(optimizer_leris, step_size=15, gamma=0.1) 

    model_mask,model_rl, train_history_df = trainer.train_and_evaluate_v0(
        model_mask, model_rl,subarticular_train_loader, subarticular_val_loader, 
        optimizer_mask, optimizer_leris,scheduler_mask, scheduler_leris,"weights/axial_T2_segmentation/model2",num_epochs=30)
    


if __name__ == '__main__':
    main()