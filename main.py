import trainer
import json
import torchvision.models as models
from torch import nn
from torchvision.transforms import transforms
def main():
    with open("config.json","r") as fl:
        config = json.load(fl)

    model = models.resnet18()
    model.conv1 = nn.Conv2d(30, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 100)

    trans = transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])
    train = {
        "seed":100,
        "log":True,
        "epoch":100,
        "model":model,
        "device":"cpu",
        "criteria":nn.CrossEntropyLoss()
    }
    
    training = trainer.Trainer(config,train,trans)
    training.train()

if __name__ == "__main__":
    main()