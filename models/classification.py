import torch
import torch.nn as nn
import timm

class ConditionalCNN(nn.Module):
    def __init__(self, model_name='convnext_tiny', num_conditions=5, num_classes=3, in_channels=5):
        super(ConditionalCNN, self).__init__()

        # Load the convnext_tiny model from timm
        self.base_model = timm.create_model(model_name, pretrained=True)
        self.num_conditions = num_conditions
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Modify the input layer to accept the desired number of input channels
        self._modify_input_layer()

        # Modify the classifier to output the desired number of classes
        if hasattr(self.base_model, 'fc'):
            in_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()  # Remove the final fully connected layer
        elif hasattr(self.base_model, 'classifier'):
            in_features = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()  # Remove the final classifier layer
        elif hasattr(self.base_model, 'head'):
            in_features = self.base_model.head.fc.in_features
            self.base_model.head.fc = nn.Identity()
        else:
            raise NotImplementedError("Model architecture not supported")
        
        # Define the new classifier that includes the condition
        self.condition_embedding = nn.Embedding(num_conditions, in_features)
        self.classifier = nn.Linear(in_features, num_classes)

    def _modify_input_layer(self):
        # Check and modify the input layer to accept the desired number of input channels (5)
        if hasattr(self.base_model, 'stem'):
            original_conv = self.base_model.stem[0]
            self.base_model.stem[0] = nn.Conv2d(self.in_channels, original_conv.out_channels,
                                                  kernel_size=original_conv.kernel_size,
                                                  stride=original_conv.stride,
                                                  padding=original_conv.padding,
                                                  bias=original_conv.bias is not None)

            # Copy the weights for the first 3 channels
            with torch.no_grad():
                self.base_model.stem[0].weight[:, :3, :, :] = original_conv.weight
                if self.in_channels > 3:
                    nn.init.kaiming_normal_(self.base_model.stem[0].weight[:, 3:, :, :])

        else:
            raise NotImplementedError("Model architecture not supported for input channel modification")

    def forward(self, x, condition):
        features = self.base_model(x)
        features = features.view(features.size(0), -1)

        condition_embedded = self.condition_embedding(condition)
        combined = features + condition_embedded

        output = self.classifier(combined)
        return output
