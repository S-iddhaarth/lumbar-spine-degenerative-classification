import torch
from torch import nn
import time
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score ,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#training process
def train_and_evaluate_v0(model_mask, model_rl, train_loader, test_loader, optimizer_mask,
                optimizer_leris, scheduler_mask, scheduler_leris,checkpoint, num_epochs=10):
    os.makedirs(checkpoint,exist_ok=True)
    device_mask = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_rl = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_mask.to(device_mask)
    model_rl.to(device_rl)
    print(f'model unet in {device_mask}') #', mdoel left right in {device_rl}')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    class_weights = torch.tensor([0.4, 1.0, 1.0], dtype=torch.float32).to(device_mask)
    criterion_mask = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

    
    optimizer_mask = optimizer_mask #torch.optim.AdamW(model.parameters(), lr=0.00001)
    optimizer_leris = optimizer_leris
    
    scheduler_mask = scheduler_mask #optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) 
    scheduler_leris = scheduler_leris
    
    train_loss_mask_set, test_loss_mask_set, train_loss_rl_set, test_loss_rl_set = [], [], [], []
    #train_coor_set,test_coor_set = [],[]
    best_test_loss = float('inf')
    train_start = time.time()
    
    for epoch in range(num_epochs):
        model_mask.train()
        #model_rl.train()
        total_loss_mask, total_loss_leris = 0.0, 0.0
        
        for inputs, mask,leris in train_loader:
            inputs_mask,input_leris, mask,leris = inputs.to(device_mask), inputs.to(device_rl), mask.to(device_mask),leris.to(device_rl)
            
            optimizer_mask.zero_grad()
            optimizer_leris.zero_grad()
            
            output_mask = model_mask(inputs_mask)
            #output_leris = model_rl(input_leris)
            
            #print(output_leris.shape)  # Should be [batch_size, num_classes]
            #print(leris.shape)  # Should be [batch_size]

            
            #loss 
            loss_mask = criterion_mask(output_mask,mask)
            #loss_leris = criterion(output_leris,leris)
            
            #total loss with backward
            loss_mask.backward()
            #loss_leris.backward()
            optimizer_mask.step()
            #optimizer_leris.step()
            
            #accumulated loss
            total_loss_mask += loss_mask.item()
            #total_loss_leris += loss_leris.item()
        
        avg_train_loss_mask = total_loss_mask / len(train_loader)
        #avg_train_loss_leris = total_loss_leris / len(train_loader)
        train_loss_mask_set.append(avg_train_loss_mask)
        #train_loss_rl_set.append(avg_train_loss_leris)
        scheduler_mask.step()
        #scheduler_leris.step()
        
        model_mask.eval()
        #model_rl.eval()
        total_loss_mask, total_loss_leris, correct_mask, correct_leris, total_mask, total_leris = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad(): 
            for inputs,mask,leris in test_loader:
                inputs_mask,input_leris, mask,leris = inputs.to(device_mask), inputs.to(device_rl), mask.to(device_mask),leris.to(device_rl)
                output_mask = model_mask(inputs_mask)
                #output_leris = model_rl(input_leris)
                
                #loss 
                loss_mask = criterion_mask(output_mask,mask)
                #loss_leris = criterion(output_leris,leris)
                total_loss_mask += loss_mask.item()
                #total_loss_leris += loss_leris.item()
                
                #accuracy
                _, preds_mask = torch.max(output_mask, 1)
                #_, preds_leris = torch.max(output_leris, 1)

                # Calculate accuracy by comparing predictions to ground truth
                correct_mask += (preds_mask == mask).sum().item()
                #correct_leris += (preds_leris == leris).sum().item()
                total_mask += mask.numel()
                #total_leris += leris.numel()
    
                
        avg_test_loss_mask = total_loss_mask / len(test_loader)
        #avg_test_loss_leris = total_loss_leris / len(test_loader)
        test_loss_mask_set.append(avg_test_loss_mask)
        #test_loss_rl_set.append(avg_test_loss_leris)
        accuracy_mask = correct_mask/total_mask *100
        #accuracy_leris = correct_leris/total_leris *100
        
        current_lr = optimizer_mask.param_groups[0]['lr']
            
        if avg_test_loss_mask <best_test_loss:
            best_test_loss = avg_test_loss_mask
            torch.save(model_mask.state_dict(),
                    os.path.join(checkpoint,'mask_best_Subarticular.pth'))
            #torch.save(model_rl.state_dict(),'lr_best_Subarticular.pth')
            print(f'best loss::{best_test_loss}')
        print(
            f'Epoch {epoch+1}/{num_epochs}: \t'
            f'Lr:{current_lr},\t'
            f'train loss mask:{avg_train_loss_mask:.6f} \t'
            f'test loss mask:{avg_test_loss_mask:.6f} \t'
            f'test acc mask:{accuracy_mask:.2f}% \t'
        )
        print('-'*80)
            
    train_end = time.time()
    time_used = train_end-train_start
    print(f'Time used for Training:{time_used} sec ')
    print('-'* 80)
    
    train_history_df=pd.DataFrame({
        'train_loss_mask': train_loss_mask_set,
        'test_loss_mask': test_loss_mask_set ,
    })
    
    torch.save(model_mask.state_dict(), 'mask_last_subarticular.pth')

    return model_mask,model_rl, train_history_df


def train_and_evaluate_v1(model_mask, model_rl, train_loader, test_loader, optimizer_mask, optimizer_leris, scheduler_mask, scheduler_leris, num_epochs=10):
    device_mask = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_rl = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_mask.to(device_mask)
    model_rl.to(device_rl)
    print(f'model unet in {device_mask}, mdoel left right in {device_rl}')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    #class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device_mask)
    #criterion_mask = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

    
    optimizer_mask = optimizer_mask #torch.optim.AdamW(model.parameters(), lr=0.00001)
    optimizer_leris = optimizer_leris
    
    scheduler_mask = scheduler_mask #optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1) 
    scheduler_leris = scheduler_leris
    
    train_loss_mask_set, test_loss_mask_set, train_loss_rl_set, test_loss_rl_set = [], [], [], []
    #train_coor_set,test_coor_set = [],[]
    best_test_loss = float('inf')
    train_start = time.time()
    
    for epoch in range(num_epochs):
        model_mask.train()
        model_rl.train()
        total_loss_mask, total_loss_leris = 0.0, 0.0
        
        for inputs, mask,leris in train_loader:
            inputs_mask,input_leris, mask,leris = inputs.to(device_mask), inputs.to(device_rl), mask.to(device_mask),leris.to(device_rl)
            
            optimizer_mask.zero_grad()
            optimizer_leris.zero_grad()
            
            output_mask = model_mask(inputs_mask)
            output_leris = model_rl(input_leris)
            
            #print(output_leris.shape)  # Should be [batch_size, num_classes]
            #print(leris.shape)  # Should be [batch_size]

            
            #loss 
            loss_mask = criterion(output_mask,mask)
            loss_leris = criterion(output_leris,leris)
            
            #total loss with backward
            loss_mask.backward()
            loss_leris.backward()
            optimizer_mask.step()
            optimizer_leris.step()
            
            #accumulated loss
            total_loss_mask += loss_mask.item()
            total_loss_leris += loss_leris.item()
        
        avg_train_loss_mask = total_loss_mask / len(train_loader)
        avg_train_loss_leris = total_loss_leris / len(train_loader)
        train_loss_mask_set.append(avg_train_loss_mask)
        train_loss_rl_set.append(avg_train_loss_leris)
        scheduler_mask.step()
        scheduler_leris.step()
        
        model_mask.eval()
        model_rl.eval()
        total_loss_mask, total_loss_leris, correct_mask, correct_leris, total_mask, total_leris = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad(): 
            for inputs,mask,leris in test_loader:
                inputs_mask,input_leris, mask,leris = inputs.to(device_mask), inputs.to(device_rl), mask.to(device_mask),leris.to(device_rl)
                output_mask = model_mask(inputs_mask)
                output_leris = model_rl(input_leris)
                
                #loss 
                loss_mask = criterion(output_mask,mask)
                loss_leris = criterion(output_leris,leris)
                total_loss_mask += loss_mask.item()
                total_loss_leris += loss_leris.item()
                
                #accuracy
                _, preds_mask = torch.max(output_mask, 1)
                _, preds_leris = torch.max(output_leris, 1)

                # Calculate accuracy by comparing predictions to ground truth
                correct_mask += (preds_mask == mask).sum().item()
                correct_leris += (preds_leris == leris).sum().item()
                total_mask += mask.numel()
                total_leris += leris.numel()
    
                
        avg_test_loss_mask = total_loss_mask / len(test_loader)
        avg_test_loss_leris = total_loss_leris / len(test_loader)
        test_loss_mask_set.append(avg_test_loss_mask)
        test_loss_rl_set.append(avg_test_loss_leris)
        accuracy_mask = correct_mask/total_mask *100
        accuracy_leris = correct_leris/total_leris *100
        
        current_lr = optimizer_mask.param_groups[0]['lr']
            
        if avg_test_loss_leris <best_test_loss:
            best_test_loss = avg_test_loss_leris
            torch.save(model_mask.state_dict(),'mask_best_Subarticular.pth')
            torch.save(model_rl.state_dict(),'lr_best_Subarticular.pth')
            print(f'best loss::{best_test_loss}')
        print(
            f'Epoch {epoch+1}/{num_epochs}: \t'
            f'Lr:{current_lr},\t'
            f'train loss mask:{avg_train_loss_mask:.6f} \t'
            f'train loss lr:{avg_train_loss_leris:.6f} \t'
            f'test loss mask:{avg_test_loss_mask:.6f} \t'
            f'test loss lr:{avg_test_loss_leris:.6f}\n'
            f'test acc mask:{accuracy_mask:.2f}% \t'
            f'test acc lr:{accuracy_leris:.2f}% \t'
        )
        print('-'*80)
            
    train_end = time.time()
    time_used = train_end-train_start
    print(f'Time used for Training:{time_used} sec ')
    print('-'* 80)
    
    train_history_df=pd.DataFrame({
        'train_loss_mask': train_loss_mask_set,
        'test_loss_mask': test_loss_mask_set ,
        'train_loss_rl': train_loss_rl_set ,
        'test_loss_rl': test_loss_rl_set ,
    })
    
    torch.save(model_mask.state_dict(), 'mask_last_subarticular.pth')
    torch.save(model_rl.state_dict(), 'lr_last_subarticular.pth')
  
    
    return model_mask,model_rl, train_history_df