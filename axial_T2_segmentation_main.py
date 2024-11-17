import os
import pandas as pd

import torch.optim as optim
from torchvision import models 
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import utils.segmentation_utility as helper
import data_loader
import utils.visualize as visuals
import models.segment as model
import trainers.axial_segmentation_trainer as trainer
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp


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
    
    model_mask = smp.Unet(
    encoder_name="timm-resnest14d",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=3,                      # model output channels (number of classes in your dataset)
)
#model_rl = neural_resnet(num_classes=2)
    model_name='neural'
    num_epochs = 100
    warmup_steps = 10
    optimizer_mask = optim.AdamW(model_mask.parameters(), lr=0.001)

    #optimizer_leris = optim.Adam(model_rl.parameters(), lr=0.001)


    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_mask, T_max=num_epochs - warmup_steps)

        # Optionally, if using PyTorch 1.10+ for LinearLR
        # Note: PyTorch 1.10+ introduced LinearLR, so ensure your version supports it.
    if torch.__version__ >= '1.10.0':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer_mask, start_factor=1e-4, total_iters=warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer_mask, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
        )
    else:
        # Custom Lambda function for warmup
        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return (epoch + 1) / warmup_steps
            else:
                return 0.5 * (1 + torch.cos((epoch - warmup_steps) / (num_epochs - warmup_steps) * torch.pi))

        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) 

    model_mask,model_rl, train_history_df = trainer.train_and_evaluate_v0(
        model_mask, subarticular_train_loader, subarticular_val_loader, 
        optimizer_mask, main_scheduler, "weights/axial_T2_segmentation/model3",num_epochs=100)
    


if __name__ == '__main__':
    main()