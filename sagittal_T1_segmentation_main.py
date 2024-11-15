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
import trainers.sagittal_segmentation_trainer as trainer
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

    df_neural_xy =helper.xy_spinal_neural(df_neural)
    df_neural_xy=df_neural_xy[~df_neural_xy['condition'].isin(['Spinal Canal Stenosis']) & ~df_neural_xy['condition'].isna()]
    condition_mapping = {
        'Right Neural Foraminal Narrowing': 0,
        'Left Neural Foraminal Narrowing': 1
    }
    df_neural_xy['condition_encoded'] = df_neural_xy['condition'].map(condition_mapping)
    
    neural_train_df, neural_val_df=train_test_split(df_neural_xy,test_size=0.2,random_state=208)
    print(f"size for train:{len(neural_train_df)}")
    print(f"size for validation:{len(neural_val_df)}")

    neural_train_dataset= data_loader.segmentation_Dataset(neural_train_df,img_train_path,transform=None)
    neural_val_dataset= data_loader.segmentation_Dataset(neural_val_df,img_train_path,transform=None)

    neural_train_loader = DataLoader(neural_train_dataset,batch_size=32,shuffle=True,num_workers=7,pin_memory=True,prefetch_factor=4 )
    neural_val_loader=DataLoader(neural_val_dataset,batch_size=32,shuffle=False,num_workers=7,pin_memory=True,prefetch_factor=4)      
    
    model_mask = model.UNetMobileNetV2(in_channels=1, out_channels=6, pretrained=True)
    model_name='neural'

    optimizer_mask = optim.AdamW(model_mask.parameters(), lr=0.001)

    scheduler_mask = optim.lr_scheduler.StepLR(optimizer_mask, step_size=10, gamma=0.1) 

    model_mask,train_loss_set, test_loss_set = trainer.train_and_evaluate_v0(model_mask,  
                                                           neural_train_loader, neural_val_loader,
                                                           optimizer_mask, 
                                                           scheduler_mask,
                                                           model_name,"weights/sagittal_T1_segmentation/model1",
                                                           num_epochs=30)
if __name__ == '__main__':
    main()