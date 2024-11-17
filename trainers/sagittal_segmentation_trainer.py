import torch
from torch import nn
import time
import os

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score ,accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import wandb
def train_and_evaluate_v0(model,train_loader,test_loader,optimizer,scheduler,model_name,
                    checkpoint,num_epochs=10,log=True):
    os.makedirs(checkpoint,exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    if log:
            wandb.init(
                job_type="naive_run",
                project="The last dance",
                entity="PaneerShawarma"
            )
    #criterion = nn.CrossEntropyLoss(reduction='mean')
    class_weights = torch.tensor([0.02, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device)
    criterion_mask = nn.CrossEntropyLoss(reduction='mean', weight=class_weights)

    optimizer=optimizer #torch.optim.AdamW(model.parameters(), lr=0.0001)
    schedular=scheduler
    
    train_loss_set,test_loss_set = [],[]
    best_test_loss = float('inf')
    train_start = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0 
        for bach in train_loader:
            inputs, labels = bach[0].to(device), bach[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(f'labels shape: {labels.shape}')
            #print(f'output shape: {outputs.shape}')
            loss = criterion_mask(outputs,labels)
            loss.backward()
            optimizer.step()
            wandb.log({"step loss":loss.item()})
            total_loss += loss.item()
        average_train_loss = total_loss / len(train_loader)

        train_loss_set.append(average_train_loss)
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            total_test_loss, correct, total = 0.0, 0.0, 0.0
            count = 0
            for bach in test_loader:
                inputs,labels = bach[0].to(device),bach[1].to(device)
                outputs = model(inputs)
                _, predicted_mask = torch.max(outputs, 1)
                test_loss=criterion_mask(outputs,labels).item()
                total_test_loss += test_loss
                
                # Get predictions by taking the class with the highest score (argmax)
                _, preds = torch.max(outputs, 1)

                # Calculate accuracy by comparing predictions to ground truth
                correct += (preds == labels).sum().item()
                total += labels.numel()
                # print(f'input - {inputs.shape}, mask - {predicted_mask.shape}, label - {labels.shape}')
                if log:
                    if count == 0 :
                        class_labels = {0: "background", 1: "L1/L2", 2: "L2/L3", 3: "L3/L4",4:"L4/L5",5:"L5/S1"}

                        masked_image = wandb.Image(
                            inputs[0,0].detach().cpu().numpy(),
                            masks={
                                "predictions": {"mask_data": predicted_mask[0].detach().cpu().numpy(), "class_labels": class_labels},
                                "ground_truth": {"mask_data": labels[0].detach().cpu().numpy(), "class_labels": class_labels},
                            },
                        )
                        
                        wandb.log({"img_with_masks": masked_image})
                count += 1
                
        average_test_loss = total_test_loss/len(test_loader)
        test_loss_set.append(average_test_loss)
        val_accuracy = correct / total *100
        current_lr = optimizer.param_groups[0]['lr']
            
        if average_test_loss <best_test_loss:
            best_test_loss = average_test_loss
            torch.save(model.state_dict(), os.path.join(checkpoint,f'lo_best_{model_name}.pth'))
            
        print(f'Epoch {epoch+1}/{num_epochs}: \t Lr:{current_lr},\n train loss:{average_train_loss:.6f} \t test loss:{average_test_loss:.6f} \t val acc:{val_accuracy:.4f}')
        print('-'*80)
        wandb.log({
            "learning rate":current_lr,
            "train loss":average_train_loss,
            "test loss":average_test_loss,
            "val accuracy":val_accuracy
        })
    train_end = time.time()
    time_used = train_end-train_start
    torch.save(model.state_dict(), os.path.join(checkpoint,f'lo_last_{model_name}.pth'))
    print(f'Time used for Training:{time_used} sec ')
    print('-'* 80)
    
    return model, train_loss_set, test_loss_set

def train_and_evaluate_v1(model_mask, model_rl, train_loader, test_loader, optimizer_mask, optimizer_leris, scheduler_mask, scheduler_leris, num_epochs=10):
    device_mask = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_rl = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_mask.to(device_mask)
    model_rl.to(device_rl)
    print(f'model unet in {device_mask}, mdoel left right in {device_rl}')
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    class_weights = torch.tensor([0.4, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32).to(device_mask)
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
        model_rl.train()
        total_loss_mask, total_loss_leris = 0.0, 0.0
        
        for inputs, mask,leris in train_loader:
            inputs_mask,input_leris, mask,leris = inputs.to(device_mask), inputs.to(device_rl), mask.to(device_mask),leris.to(device_rl)
            
            optimizer_mask.zero_grad()
            optimizer_leris.zero_grad()
            
            output_mask = model_mask(inputs_mask)
            output_leris = model_rl(input_leris)
            
            #loss 
            loss_mask = criterion_mask(output_mask,mask)
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
                loss_mask = criterion_mask(output_mask,mask)
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

################### EVALUATION ########################################
def evaluation_mask(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss(reduction='mean')  # For left-right prediction
    
    total_loss,correct_mask = 0.0, 0.0
    total_labels = 0.0
    model.eval()
    for inputs, mask,_ in data_loader:
        inputs, labels = inputs.to(device), mask.to(device)
        
        outputs = model(inputs)
        
        #loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        #accuracy
        _, preds = torch.max(outputs, 1)
        # Calculate accuracy by comparing predictions to ground truth
        correct_mask += (preds == labels).sum().item()
        total_labels += labels.numel() #1k++
        
    average_loss = total_loss / len(data_loader)
    accuracy_mask = correct_mask/total_labels *100
    
    print(f'Total Loss: {average_loss}, \t Total accuracy:{accuracy_mask}')
    print('-' * 80)
    
    


def evaluation_leftRight(model, data_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)
    
    criterion = nn.CrossEntropyLoss(reduction='mean')  # For left-right prediction
    
    all_test_labels = []
    all_test_outputs = []
   
    total_loss,correct_mask = 0.0, 0.0
    total_labels = 0.0
    
    model.eval()
    for inputs, _, lr in data_loader:
        inputs, labels = inputs.to(device), lr.to(device)
        outputs = model(inputs)
        
        #loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        
        #accuracy
        _, preds = torch.max(outputs, 1)
        
        # Calculate accuracy by comparing predictions to ground truth
        correct_mask += (preds == labels).sum().item()
        total_labels += labels.numel() 
        #print(labels.numel()) #16
        
        # Collect labels and predicted probabilities
        all_test_labels.extend(labels.cpu().numpy())
        all_test_outputs.extend(preds.detach().cpu().numpy())
   
    average_loss = total_loss / len(data_loader)
    accuracy_mask = correct_mask/total_labels *100
    print(f'Total Loss: {average_loss}, \t Total accuracy:{accuracy_mask}')
    
    # Convert to numpy arrays
    all_labels = np.array(all_test_labels)
    all_predictions = np.array(all_test_outputs)
    
    # Generate and display confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(2))
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - Left-Right Prediction')
    plt.show()
    print('-' * 80)