import torch
import wandb
import data_loader
from torch.utils.data import DataLoader
import utils.utility
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.nn import Sigmoid
from tqdm import tqdm
from timm.data.auto_augment import rand_augment_transform


class Trainer():
    def __init__(self,config:dict,train:dict,transform,augumentation)->None:
        self.train_solution = None
        self.valid_solution = None
        self.tfm = augumentation
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
                project="lumbar-spine-no-error",
                entity="PaneerShawarma"
            )

    def _load(self):
        
        train = data_loader.naive_loader(
            self.paths["dataset"]["train"]["annotation"],ch=5,transform=self._transfrom,
            augment = self.tfm
        )
        
        valid = data_loader.naive_loader(
            self.paths["dataset"]["train"]["annotation"],ch=5,transform=self._transfrom,
            train=False
        )

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
            shuffle=False,
            num_workers=self.train_config["data_loader"]["valid"]["num_workers"],
            pin_memory=self.train_config["data_loader"]["valid"]["pin_memory"],
            drop_last=self.train_config["data_loader"]["valid"]["drop_last"],
            timeout=self.train_config["data_loader"]["valid"]["timeout"],
            prefetch_factor=self.train_config["data_loader"]["valid"]["prefetch_factor"],
            persistent_workers=self.train_config["data_loader"]["valid"]["persistent_workers"],
            )
        self.valid_solution = utils.utility.generate_ground_truth('Data/',valid.data_list)
        self.train_solution = utils.utility.generate_ground_truth('Data/',train.data_list)
        return trainLoader,validLoader

    def _run_batch(self,image,label,train=True):
        if train:
            self.optimizer.zero_grad()
            logits = self.model(image)
            # sigmoid = Sigmoid()
            # probs = sigmoid(logits)
            # threshold = 0.5
            # preds = (probs >= threshold).float()
            logits = logits.reshape((logits.shape[0],-1,3))
            label = label.reshape((label.shape[0],-1,3))
            loss = 0
            for i in range(logits.shape[1]):
                logits_inter = logits[:,i,:]
                label_inter = label[:,i:i+1,:].squeeze(1)
                loss += self.criteria(logits_inter,label_inter)
            # label = label.permute((0,2,1))
            # logits = torch.nn.functional.softmax(logits,dim=2)
            # logits = logits.permute((0,2,1))
            # loss = self.criteria(logits.to(self.device),label.to(self.device))
            loss.backward()
            self.optimizer.step()
            
            return loss.detach().item(),logits
        else:
            with torch.no_grad():
                logits = self.model(image)
                # sigmoid = Sigmoid()
                # probs = sigmoid(logits)
                # threshold = 0.5
                # preds = (probs >= threshold).float()
                # loss = self.criteria(logits,label)
            
                return 1,logits


    def _run_epoch(self, epoch):
        running_loss = 0
        self.model.train()
        all_preds = []
        all_labels = []
        all_id = []
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", ncols=100)
        for batch in progress_bar:
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            ids = batch[2]
            loss, preds = self._run_batch(image, label)
            all_preds.append(preds.detach().cpu())
            all_labels.append(label.detach().cpu())
            all_id.append(ids)
            running_loss += loss
            if self.log:
                out = utils.utility.grad_flow_dict(self.model.named_parameters())
                out.update({"train step loss":loss})
                wandb.log(out)
        y_pred = torch.cat(all_preds)
        y_id = torch.cat(all_id)
    
        # y_pred = y_pred.permute((0,2,1))
        y_pred = torch.nn.functional.softmax(y_pred,dim=2)
        y_pred = y_pred.reshape((-1,3))
        submission = self.train_solution.copy()[["row_id", "normal_mild", "moderate", "severe"]]
        submission[["normal_mild", "moderate", "severe"]] = y_pred
        y_label = torch.cat(all_labels)
        y_label = y_label.reshape((y_label.shape[0],-1,3))
        # y_label = y_label.permute((0,2,1))
        y_label = y_label.reshape((-1,3)).numpy().astype(int)
        print(utils.utility.score(self.train_solution.copy(),submission,"row_id",1))
        
        y_label = utils.utility.substitute_patterns(y_label)
        y_pred = torch.argmax(y_pred, dim=1).tolist()
        
        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, average='weighted',zero_division=1)
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
        print(f'total train loss: {running_loss/len(self.train_loader)}')

        self._save_checkpoint(epoch)
        # self._run_inference(epoch)


    def _save_checkpoint(self,epoch):
        os.makedirs(self.paths["checkpoint"],exist_ok=True)
        path = os.path.join(self.paths["checkpoint"],f"{epoch}.pth")
        
        torch.save(self.model.state_dict(),path)
        
    def _run_inference(self,epoch):
        running_loss = 0
        self.model.eval()
        all_preds = []
        all_labels = []
        all_id = []
        for batch in self.valid_loader:
            image = batch[0].to(self.device)
            label = batch[1].to(self.device)
            loss,preds = self._run_batch(image,label,train=False)
            all_preds.append(preds.detach().cpu())
            all_labels.append(label.detach().cpu())
            all_id.append(batch[2])
            running_loss += loss
        y_pred = torch.cat(all_preds)
        y_id = torch.cat(all_id)
        y_pred = torch.reshape(y_pred,(y_pred.shape[0],-1,3))
        y_pred = torch.nn.functional.softmax(y_pred,dim=2)
        y_pred = y_pred.reshape((-1,3))
        submission = self.valid_solution.copy()[["row_id", "normal_mild", "moderate", "severe"]]
        submission[["normal_mild", "moderate", "severe"]] = y_pred
        y_label = torch.cat(all_labels)
        y_label = y_label.reshape((y_label.shape[0],-1,3))
        y_label = y_label.reshape((-1,3)).numpy().astype(int)
        metric = utils.utility.score(self.valid_solution.copy(),submission,"row_id",1)
        
        y_label = utils.utility.substitute_patterns(y_label)
        y_pred = torch.argmax(y_pred, dim=1).tolist()
        accuracy = accuracy_score(y_label, y_pred)
        precision = precision_score(y_label, y_pred, average='weighted',zero_division=1)
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