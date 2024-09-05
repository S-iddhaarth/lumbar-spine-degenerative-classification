import torch
import wandb
import dataLoader
from torch.utils.data import DataLoader
import utils.utility

class Trainer():
    def __init__(self,config:dict,train:dict):
        self.seed = train["seed"]
        if self.seed:
            utils.utility.seed_everything(self.seed)

        self.paths = config["paths"]
        self.model_config = config["model_config"]
        self.train_config = config["train_config"]
        self.criteria = train['criteria']
        self.model = train['model']
        self.transform = train['trainsform']

        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.train_config["optimizer"]["learning_rate"],
            betas=self.train_config["optimizer"]["betas"],
            eps=self.train_config["optimizer"]["eps"],
            weight_decay=self.train_config["optimizer"]["weight_decay"],
            amsgrad=self.train_config["optimizer"]["amsgrad"],
            foreach=self.train_config["optimizer"]["foreach"],
            maximize=self.train_config["optimizer"]["maximize"],
            capturable=self.train_config["optimizer"]["captureable"],
            differentiable=self.train_config["optimizer"]["differentiable"],
            fused=self.train_config["optimizer"]["fused"])

        self.train_loader,self.test_loader = self._load()
    def _load(self):
        
        train = dataLoader.DataLoader_np(
            self.paths["dataset"]["train"],self.transform,test=False
        )
        test = dataLoader.DataLoader_np(
            self.paths["dataset"]["test"],self.transform,test=True
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

        testLoader = DataLoader(
            dataset=test,
            batch_size=self.train_config["data_loader"]["test"]["batch_size"],
            shuffle=self.train_config["data_loader"]["test"]["shuffle"],
            num_workers=self.train_config["data_loader"]["test"]["num_workers"],
            pin_memory=self.train_config["data_loader"]["test"]["pin_memory"],
            drop_last=self.train_config["data_loader"]["test"]["drop_last"],
            timeout=self.train_config["data_loader"]["test"]["timeout"],
            prefetch_factor=self.train_config["data_loader"]["test"]["prefetch_factor"],
            persistent_workers=self.train_config["data_loader"]["test"]["persistent_workers"],
            )
        return trainLoader,testLoader
    def _run_batch(self,image,label,coords):
        pass
    def _run_epoch(self,epoch):
        pass
    def _save_checkpoint(self):
        pass
    def _run_inference(self):
        pass
    def train(self):
        pass