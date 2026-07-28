import torch
import torch.nn as nn
from compressai.zoo import bmshj2018_hyperprior

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv1(concat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_gate = ChannelAttention(in_channels, reduction)
        self.spatial_gate = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.channel_gate(x)
        return self.spatial_gate(x_out)

class HyperpriorWithCBAM(nn.Module):
    def __init__(self, quality, cbam_position, pretrained=True):
        super(HyperpriorWithCBAM, self).__init__()
        self.base_model = bmshj2018_hyperprior(quality=quality, pretrained=pretrained)
        self.cbam_position = cbam_position
        M = self.base_model.M
        
        if self.cbam_position in ['encoder', 'decoder']:
            self.cbam = CBAM(in_channels=M, reduction=16)
            # 그래디언트 소실을 방지하고 Warmup 학습이 가능하도록 0.01로 초기화
            self.cbam_alpha = nn.Parameter(torch.tensor([0.01]))
        else:
            self.cbam = None
            
    def forward(self, x):
        y = self.base_model.g_a(x)
        if self.cbam_position == 'encoder' and self.cbam is not None:
            y = y + self.cbam_alpha * self.cbam(y) 
            
        z = self.base_model.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.base_model.entropy_bottleneck(z)
        scales_hat = self.base_model.h_s(z_hat)
        y_hat, y_likelihoods = self.base_model.gaussian_conditional(y, scales_hat)
        
        if self.cbam_position == 'decoder' and self.cbam is not None:
            y_hat = y_hat + self.cbam_alpha * self.cbam(y_hat)
            
        x_hat = self.base_model.g_s(y_hat)
        return {"x_hat": x_hat, "likelihoods": {"y": y_likelihoods, "z": z_likelihoods}}

    def aux_loss(self):
        return self.base_model.aux_loss()