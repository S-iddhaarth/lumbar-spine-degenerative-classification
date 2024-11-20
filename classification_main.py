import pandas as pd
import Preprocessing.utility as preprocess
import gc
import os
from models.classification import ConditionalCNN
from torch.utils.data import DataLoader
from torch import nn
import torch
import data_loader
import trainers.classification_trainer as trainer
def load_meta(root):
    train_label = os.path.join(root,'train_label_coordinates.csv')
    train_series = os.path.join(root,'train_series_descriptions.csv')
    train_ = os.path.join(root,'train.csv')
    train_df = pd.read_csv(train_)
    series_df = pd.read_csv(train_series)
    label_df = pd.read_csv(train_label)
    
    return train_df,series_df,label_df

def main():
    root = 'Data/'
    
    train_df,series_df,label_df = load_meta(root)
    train_annotation = preprocess.generate_train_annotation(label_df,series_df,train_df)
    train_annotation = train_annotation.dropna()
    label_order = ["Normal/Mild", "Moderate", "Severe"]
    train_annotation['label'] = pd.Categorical(train_annotation['label'], categories=label_order, ordered=True)
    train_annotation = pd.get_dummies(train_annotation, columns=['label'], prefix='', prefix_sep='')
    del train_df
    del series_df
    del label_df
    gc.collect

    unique_values = ["Sagittal T1", "Sagittal T2/STIR", "Axial T2"]

    sagittal_t1_df = train_annotation[train_annotation['series_description'] == "Sagittal T1"]
    sagittal_t2_stir_df = train_annotation[train_annotation['series_description'] == "Sagittal T2/STIR"]
    axial_t2_df = train_annotation[train_annotation['series_description'] == "Axial T2"]

    column_to_check = 'condition'

    sagittal_t1_df_left = sagittal_t1_df[sagittal_t1_df[column_to_check].str.split().str[0].str.lower() == "left"]
    sagittal_t1_df_right = sagittal_t1_df[sagittal_t1_df[column_to_check].str.split().str[0].str.lower() == "right"]

    axial_t2_df_left = axial_t2_df[axial_t2_df[column_to_check].str.split().str[0].str.lower() == "left"]
    axial_t2_df_right = axial_t2_df[axial_t2_df[column_to_check].str.split().str[0].str.lower() == "right"]

    print("Sagittal T1 Left DataFrame shape:", sagittal_t1_df_left.shape)
    print("Sagittal T1 Right DataFrame shape:", sagittal_t1_df_right.shape)
    print("Axial T2 Left DataFrame shape:", axial_t2_df_left.shape)
    print("Axial T2 Right DataFrame shape:", axial_t2_df_right.shape)
    print("Sagittal T2 DataFrame shape:",sagittal_t2_stir_df.shape)
    print("total : ", sagittal_t1_df_left.shape[0] + sagittal_t1_df_right.shape[0]
                    + axial_t2_df_left.shape[0] +  axial_t2_df_right.shape[0] + 
                    sagittal_t2_stir_df.shape[0])
    
    sagittal_t1_left = ConditionalCNN()
    sagittal_t1_dataset = data_loader.classification_loader(
        sagittal_t1_df_left,'Data/preprocessed_image',side="left"
    )
    sagittal_t1_dataset[0]
    sagittal_t1_loader = DataLoader(
        sagittal_t1_dataset,batch_size=256,num_workers=7,pin_memory=True,
        persistent_workers=True,prefetch_factor=8,shuffle=True
    )
    sagittal_t1_left.to('cuda')

    criterion = nn.CrossEntropyLoss()

# Define the optimizer
    optimizer = torch.optim.AdamW(sagittal_t1_left.parameters(), lr=0.001)
    
    # Define the number of epochs and warmup steps
    num_epochs = 100
    warmup_steps = 10
    
    # Define the scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_steps)
    
    # Optionally, if using PyTorch 1.10+ for LinearLR
    # Note: PyTorch 1.10+ introduced LinearLR, so ensure your version supports it.
    if torch.__version__ >= '1.10.0':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
        main_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_steps]
        )
    else:
        # Custom Lambda function for warmup
        def lr_lambda(epoch):
            if epoch < warmup_steps:
                return (epoch + 1) / warmup_steps
            else:
                return 0.5 * (1 + torch.cos((epoch - warmup_steps) / (num_epochs - warmup_steps) * torch.pi))
        
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
    trainer.train_and_evaluate(
        sagittal_t1_loader,sagittal_t1_left,criterion,
        main_scheduler,optimizer,100,'cuda',True,"weights/mod1"
    )
    
if __name__ == '__main__':
    main()