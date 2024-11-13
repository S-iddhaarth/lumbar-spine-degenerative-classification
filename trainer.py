import torch
import wandb
import data_loader
from torch.utils.data import DataLoader
import utils.utility
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import Sigmoid
from tqdm import tqdm

class Trainer():
    def __init__(self,config:dict,train:dict,transform)->None:
        self.seed = train["seed"]
        self.log = train["log"]
        if self.seed:
            utils.utility.seed_everything(self.seed)
        self._transfrom = transform
        self.epoch = train["epoch"]
        self.paths = config["paths"]
        self.train_config = config["train_config"]
        self.criteria = train['criteria']
        self.model = train['model']
        
        self.device = train['device']
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.train_config["optimizer"]["learning_rate"],
            betas=self.train_config["optimizer"]["betas"],
            eps=self.train_config["optimizer"]["eps"],
            weight_decay=self.train_config["optimizer"]["weight_decay"],
            amsgrad=self.train_config["optimizer"]["amsgrad"],
            foreach=self.train_config["optimizer"]["foreach"],
            maximize=self.train_config["optimizer"]["maximize"],
            differentiable=self.train_config["optimizer"]["differentiable"],
            fused=self.train_config["optimizer"]["fused"]
        )

        self.train_loader,self.valid_loader = self._load()
        if self.log:
            wandb.init(
                job_type="naive_run",
                config=config,
                project="lumbar-spine-degnerative-classification",
                entity="PaneerShawarma"
            )

    def _load(self):
        
        data = data_loader.naive_loader(
            self.paths["dataset"]["train"]["annotation"],ch=5,transform=self._transfrom
        )
        valid_split = int(len(data)*0.2)
        train_split = len(data)-valid_split
        data = torch.utils.data.random_split(data,[train_split,valid_split])
        train,valid = data[0],data[1]

        trainLoader = DataLoader(
            dataset=train,
            batch_size=self.train_config["data_loader"]["train"]["batch_size"],
            shuffle=self.train_config["data_loader"]["train"]["shuffle"],
            num_workers=self.train_config["data_loader"]["train"]["num_workers"],
            pin_memory=self.train_config["data_loader"]["train"]["pin_memory"],
            drop_last=self.train_config["data_loader"]["train"]["drop_last"],
            timeout=self.train_config["data_loader"]["train"]["timeout"],
            prefetch_factor=self.train_config["data_loader"]["train"]["prefetch_factor"],
            persistent_workers=self.train_config["data_loader"]["train"]["persistent_workers"]
            )

        validLoader = DataLoader(
            dataset=valid,
            batch_size=self.train_config["data_loader"]["valid"]["batch_size"],
            shuffle=self.train_config["data_loader"]["valid"]["shuffle"],
            num_workers=self.train_config["data_loader"]["valid"]["num_workers"],
            pin_memory=self.train_config["data_loader"]["valid"]["pin_memory"],
            drop_last=self.train_config["data_loader"]["valid"]["drop_last"],
            timeout=self.train_config["data_loader"]["valid"]["timeout"],
            prefetch_factor=self.train_config["data_loader"]["valid"]["prefetch_factor"],
            persistent_workers=self.train_config["data_loader"]["valid"]["persistent_workers"],
            )

        return trainLoader,validLoader

    def _run_batch(self,image,label,train=True):
        if train:
            self.optimizer.zero_grad()
            logits = self.model(image)
            sigmoid = Sigmoid()
            probs = sigmoid(logits)
            threshold = 0.5
            preds = (probs >= threshold).float()
            loss = self.criteria(logits,label)
            loss.backward()
            self.optimizer.step()
            
            return loss.detach().item(),preds
        else:
            with torch.no_grad():
                logits = self.model(image)
                sigmoid = Sigmoid()
                probs = sigmoid(logits)
                threshold = 0.5
                preds = (probs >= threshold).float()
                loss = self.criteria(logits,label)
                return loss.detach().item(),preds


    def _run_epoch(self, epoch):
        running_loss = 0
        self.model.train()
        all_preds = []
        all_labels = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        for batch in progress_bar:
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            loss, preds = self._run_batch(image, label)
            all_preds.append(preds.detach().cpu())
            all_labels.append(label.detach().cpu())
            running_loss += loss
            if self.log:
                out = utils.utility.grad_flow_dict(self.model.named_parameters())
                out.update({"train step loss":loss})
                wandb.log(out)

        y_pred = torch.cat(all_preds).numpy()
        y_label = torch.cat(all_labels).numpy()

        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, average='weighted')
        recall = recall_score(y_label, y_pred, average='weighted')
        f1 = f1_score(y_label, y_pred, average='weighted')
        if self.log:
            out = {
            "train accuracy":accuracy,
            "train precision":precision,
            "train recall":recall,
            "train f1":f1,
            "total train loss":running_loss/len(self.train_loader)
        }
            wandb.log(out)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')

        self._save_checkpoint(epoch)
        self._run_inference(epoch)


    def _save_checkpoint(self,epoch):
        os.makedirs(self.paths["checkpoint"],exist_ok=True)
        path = os.path.join(self.paths["checkpoint"],f"{epoch}.pth")
        
        torch.save(self.model.state_dict(),path)
        
    def _run_inference(self,epoch):
        running_loss = 0
        self.model.eval()
        all_logits = []
        all_labels = []
        for batch in self.valid_loader:
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            loss,preds = self._run_batch(image,label,train=False)
            all_logits.append(preds.detach().cpu())
            all_labels.append(label.detach().cpu())
            running_loss += loss
        y_pred = torch.cat(all_logits).numpy()
        y_label = torch.cat(all_labels).numpy()
        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, average='weighted')
        recall = recall_score(y_label, y_pred, average='weighted')
        f1 = f1_score(y_label, y_pred, average='weighted')
        if self.log:
            out = {
            "validation accuracy":accuracy,
            "validation precision":precision,
            "validation recall":recall,
            "validation f1":f1,
            "total validation loss":running_loss/len(self.train_loader)
        }
            wandb.log(out)
        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        
    def train(self):
        for i in range(self.epoch):
            self._run_epoch(i)