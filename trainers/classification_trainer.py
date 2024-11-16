import wandb
import utils.utility as utils
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os
from tqdm import tqdm

def train_and_evaluate(
    dataset, model, criteria, lr_scheduler, optimizer,
    epoch, device, log, checkpoint, frequency=5
):
    
    if log:
        wandb.init(
            job_type="naive_run",
            project="lumbar-spine-no-error",
            entity="PaneerShawarma"
        )

    # Ensure the model is on the correct device
    model.to(device)
    
    for e in range(epoch):
        predictions = []
        labels = []
        running_loss = 0  # Reset running_loss at the start of each epoch
        
        model.train()  # Ensure the model is in training mode
        
        for image, label, condition in tqdm(dataset, total=len(dataset)):
            # Move the inputs and targets to the correct device
            image, label, condition = image.to(device), label.to(device), condition.to(device)
            
            logits = model(image, condition)
            
            # Move the criterion to the same device
            criteria = criteria.to(device)
            loss = criteria(logits, label)
            
            optimizer.zero_grad()  # Make sure gradients are zeroed before backward pass
            loss.backward()
            optimizer.step()

            grad = utils.grad_flow_dict(model.named_parameters())
            grad.update({"step loss": loss.item()})
            
            predictions.append(torch.argmax(logits, dim=1).cpu())
            labels.append(label.cpu())
            
            running_loss += loss.item()
        
        lr_scheduler.step()
        
        y_pred = torch.cat(predictions).detach().numpy()
        y_label = torch.cat(labels).detach().numpy()
        
        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_label, y_pred, average='weighted')
        f1 = f1_score(y_label, y_pred, average='weighted')
        
        # Log the total loss for the current epoch
        print(f'total loss : {running_loss / len(dataset)}')
        
        # Reset running_loss after logging
        running_loss = 0
        
        if log:
            out = {
                "learning rate": optimizer.param_groups[0]['lr'],
                "validation accuracy": accuracy,
                "validation precision": precision,
                "validation recall": recall,
                "validation f1": f1, 
                "total validation loss": running_loss / len(dataset)
            }
            wandb.log(out)
        
        print(f'Validation Accuracy: {accuracy}')
        print(f'Validation Precision: {precision}')
        print(f'Validation Recall: {recall}')
        print(f'Validation F1 Score: {f1}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]}')
        
        if (e + 1) % frequency == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint, f'{e}.pth'))

