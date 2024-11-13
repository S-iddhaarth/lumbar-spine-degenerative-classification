import trainer
import json
import torchvision.models as models
from torch import nn
from torchvision.transforms import transforms
def main():
    with open("config.json","r") as fl:
        config = json.load(fl)

    # Load the ConvNeXt model
    model = models.convnext_tiny(pretrained=True)

# Modify the first convolutional layer to accept 15 input channels
# ConvNeXt Tiny's first conv layer is named 'features.0.conv'
    model.features[0][0] = nn.Conv2d(15, model.features[0][0].out_channels, kernel_size=4, stride=4)
    
    # Modify the final fully connected layer to output 100 classes
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 100)

    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train = {
        "seed":100,
        "log":True,
        "epoch":100,
        "model":model,
        "device":"cuda",
        "criteria":nn.BCEWithLogitsLoss()
    }
    
    training = trainer.Trainer(config,train,trans)
    training.train()

if __name__ == "__main__":
    main()