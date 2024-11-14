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