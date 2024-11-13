import trainer
import json
import torchvision.models as models
from torch import nn
from torchvision.transforms import transforms
def main(annotate = False):
    with open("config.json","r") as fl:
        config = json.load(fl)
    if annotate:
        import data_preparation.annotate as annote
        annote.naive_annotate('./',config["paths"]["dataset"]["train"],'./Data/annotation.json')
        annote.remove_more_than_3('./Data/annotation.json','./Data/annotation.json')
        annote.split_annotations('./Data/annotation.json',0.8)
    # Load the ConvNeXt model
    model = models.convnext_tiny()

# Modify the first convolutional layer to accept 15 input channels
# ConvNeXt Tiny's first conv layer is named 'features.0.conv'
    model.features[0][0] = nn.Conv2d(15, model.features[0][0].out_channels, kernel_size=4, stride=4)
    
    # Modify the final fully connected layer to output 100 classes
    num_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(num_features, 75)

    trans = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    train = {
        "seed":100,
        "log":False,
        "epoch":100,
        "model":model,
        "device":"cuda",
        "criteria":nn.BCEWithLogitsLoss()
    }
    
    training = trainer.Trainer(config,train,trans)
    training.train()

if __name__ == "__main__":
    main(False)