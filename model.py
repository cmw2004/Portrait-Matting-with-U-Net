# model.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Basic conv block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, skip_c):
        super().__init__()
        # in_c: channels of input (decoder feature), out_c: desired output channels after block
        # skip_c: channels from the encoder skip connection
        self.up = nn.ConvTranspose2d(in_c, out_c, 2, stride=2)
        # after upsampling we concatenate the skip (out_c + skip_c)
        self.conv = ConvBlock(out_c + skip_c, out_c)
    def forward(self, x, skip):
        x = self.up(x)
        # ensure spatial sizes match (ConvTranspose may produce off-by-one)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

# Optional attention gate (simple)
class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, 1, bias=False), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.Conv2d(f_int, 1, 1, bias=False), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # ensure spatial sizes match (g may be coarser); upsample g1 to x1 if needed
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class UNet(nn.Module):
    def __init__(self, n_classes=1, base_c=64, encoder='resnet18', use_attention=False, pretrained=True):
        super().__init__()
        self.use_attention = use_attention
        if encoder == 'resnet18':
            res = models.resnet18(pretrained=pretrained)
            self.enc0 = nn.Sequential(res.conv1, res.bn1, res.relu)    # out 64, /2
            self.pool = res.maxpool
            self.enc1 = res.layer1   # 64
            self.enc2 = res.layer2   # 128
            self.enc3 = res.layer3   # 256
            self.enc4 = res.layer4   # 512
        else:
            # fallback: simple conv encoder
            self.enc0 = ConvBlock(3, base_c)
            self.pool = nn.MaxPool2d(2)
            self.enc1 = ConvBlock(base_c, base_c)
            self.enc2 = ConvBlock(base_c, base_c*2)
            self.enc3 = ConvBlock(base_c*2, base_c*4)
            self.enc4 = ConvBlock(base_c*4, base_c*8)

        # decoder
        self.center = ConvBlock(512, 512)
        # specify skip channel sizes from the encoder (resnet18): e4=512, e3=256, e2=128, e1=64
        self.up4 = UpBlock(512, 256, 512)
        self.up3 = UpBlock(256, 128, 256)
        self.up2 = UpBlock(128, 64, 128)
        self.up1 = UpBlock(64, 64, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, n_classes, 1),
            nn.Sigmoid()
        )

        if use_attention:
            # Attention gate params: (gating_channels, skip_channels, inter_channels)
            self.ag4 = AttentionGate(512, 512, 256)
            self.ag3 = AttentionGate(256, 256, 128)
            self.ag2 = AttentionGate(128, 128, 64)
            self.ag1 = AttentionGate(64, 64, 32)

    def forward(self, x):
        e0 = self.enc0(x)        # /2
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        c = self.center(self.pool(e4))
        if self.use_attention:
            d4 = self.up4(c, self.ag4(c, e4))
            d3 = self.up3(d4, self.ag3(d4, e3))
            d2 = self.up2(d3, self.ag2(d3, e2))
            d1 = self.up1(d2, self.ag1(d2, e1))
        else:
            d4 = self.up4(c, e4)
            d3 = self.up3(d4, e3)
            d2 = self.up2(d3, e2)
            d1 = self.up1(d2, e1)
        out = self.final(d1)
        return out
