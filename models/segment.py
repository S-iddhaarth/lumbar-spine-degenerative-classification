from torch import nn
from torchvision import models
import torch
class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batchnorm=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)
        self.bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

#class Conv2Conv

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = Conv2dReLU(in_channels, out_channels)
        self.conv2 = Conv2dReLU(out_channels, out_channels)
        
    def forward(self, x, skip):
        x = torch.cat([x, skip], dim=1)  # Skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class MobileNetV2Encoder(nn.Module):
    def __init__(self, in_channels=1, pretrained=True):
        super(MobileNetV2Encoder, self).__init__()
        mobilenet_v2 = models.mobilenet_v2(pretrained=pretrained)
        mobilenet_v2.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Extract layers from the pretrained MobileNetV2
        self.enc0 = mobilenet_v2.features[0:2]  # 32 channels    [1, 16, 128, 128]
        self.enc1 = mobilenet_v2.features[2:4]  # 24 channels    [1, 24, 64, 64]
        self.enc2 = mobilenet_v2.features[4:7]  # 32 channels    [1, 32, 32, 32]
        self.enc3 = mobilenet_v2.features[7:14]  # 96 channels   [1, 96, 16, 16]
        self.enc4 = mobilenet_v2.features[14:18]  # 320 channels  [1, 320, 8, 8]
        self.bottleneck = mobilenet_v2.features[18:] # 1280 channels  [1, 1280, 8, 8]
        

    def forward(self, x):
        features = []
        x = self.enc0(x)  # Downsample 1
        features.append(x)
        
        x = self.enc1(x)  # Downsample 2
        features.append(x)
        
        x = self.enc2(x)  # Downsample 3
        features.append(x)
        
        x = self.enc3(x)  # Downsample 4
        features.append(x)
        
        x = self.enc4(x)
        features.append(x)
        
        x=self.bottleneck(x)
        return x, features

class UNetMobileNetV2(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, pretrained=True):
        super(UNetMobileNetV2, self).__init__()
        self.encoder = MobileNetV2Encoder(in_channels=in_channels,pretrained=pretrained)

        # Bottleneck output is 1280 from MobileNetV2
        self.bottleneck = Conv2dReLU(320, 1280)
        
        # Upsampling layers
        self.upconv4 = nn.ConvTranspose2d(1280, 320, kernel_size=2, stride=2)  # Upsample to 512
        self.upconv3 = nn.ConvTranspose2d(208, 96, kernel_size=2, stride=2)   # Upsample to 256
        self.upconv2 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2)   # Upsample to 128
        self.upconv1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)     # Upsample to 64
        self.upconv0 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)

        # Decoder
        self.dec4 = DecoderBlock(320+96, 208)  # Decoder for bottleneck and enc3
        self.dec3 = DecoderBlock(96+32, 96)   # Decoder for dec4 and enc2
        self.dec2 = DecoderBlock(96+24, 64)   # Decoder for dec3 and enc1
        self.dec1 = DecoderBlock(64+16, 32)   # Decoder for dec2 and enc0

        # Final segmentation head
        self.segmentation_head = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        bottleneck, features = self.encoder(x)
       
        x = self.upconv4(bottleneck) 
        x = self.dec4(x, features[3])  
        
        x = self.upconv3(x)
        x = self.dec3(x, features[2])  
        
        x = self.upconv2(x)
        x = self.dec2(x, features[1])  
       
        x = self.upconv1(x)
        x = self.dec1(x, features[0]) 
        #print(f'dec4 {x.shape}')
       
        # Final segmentation head
        x = self.upconv0(x) 
        x = self.segmentation_head(x)
        return x



class subarticular_resnet(nn.Module):
    def __init__(self, num_classes=2):
        super(subarticular_resnet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
        # Modify the fully connected layer for 2-class output
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
    

class Conv2dReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UNetVGG16(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, pretrained=True):
        super(UNetVGG16, self).__init__()

        # Load VGG16 model
        vgg16 = models.vgg16(pretrained=pretrained)

        # Modify the first layer for 1 channel input
        vgg16.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3,stride=1, padding=1)

        # Encoder layers
        self.enc1 = nn.Sequential(*vgg16.features[:4])  # 64 channels
        self.enc2 = nn.Sequential(*vgg16.features[4:9])  # 128 channels
        self.enc3 = nn.Sequential(*vgg16.features[9:16])  # 256 channels
        self.enc4 = nn.Sequential(*vgg16.features[16:23])  # 512 channels
        self.enc5 = nn.Sequential(*vgg16.features[23:30])  # 512 channels

        # Bottleneck
        self.bottleneck = Conv2dReLU(512, 512)

        # Decoder layers
        self.dec4 = self.upconv(512, 512)  # 512 -> 512
        self.dec3 = self.upconv(1024, 256)  # 1024 -> 256 (after concatenation)
        self.dec2 = self.upconv(512, 128)   # 512 -> 128 (after concatenation)
        self.dec1 = self.upconv(256, 64)    # 256 -> 64 (after concatenation)

        # Final output layer
        self.final_conv = nn.Conv2d(128, out_channels, kernel_size=1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.5)

    def upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [B, 64, H/2, W/2]
        enc2 = self.enc2(enc1)  # [B, 128, H/4, W/4]
        enc3 = self.enc3(enc2)  # [B, 256, H/8, W/8]
        enc4 = self.enc4(enc3)  # [B, 512, H/16, W/16]
        enc5 = self.enc5(enc4)  # [B, 512, H/32, W/32]
        
        # Bottleneck
        bottleneck = self.dropout(self.bottleneck(enc5))  # [B, 512, H/32, W/32]


        # Decoder
        dec4 = self.dec4(bottleneck)  # [B, 512, H/16, W/16]
        dec4 = torch.cat((dec4, enc4), dim=1)  # [B, 1024, H/16, W/16
        
        dec3 = self.dec3(dec4)  # [B, 256, H/8, W/8]
        dec3 = torch.cat((dec3, enc3), dim=1)  # [B, 512, H/8, W/8]
        
        dec2 = self.dec2(dec3)  # [B, 128, H/4, W/4]
        dec2 = torch.cat((dec2, enc2), dim=1)  # [B, 256, H/4, W/4]
        
        dec1 = self.dec1(dec2)  # [B, 64, H/2, W/2]
        dec1 = torch.cat((dec1, enc1), dim=1)  # [B, 128, H/2, W/2]

        # Final output
        out = self.final_conv(dec1)  # [B, out_channels, H/2, W/2]
        return out
    

