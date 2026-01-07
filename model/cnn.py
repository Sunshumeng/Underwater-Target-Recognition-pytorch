import math

import torch
import torch.nn as nn
from torch.nn import Conv2d, ReLU, BatchNorm2d, AdaptiveAvgPool2d, Linear, init, ConvTranspose2d, Dropout
import torch.nn.functional as F


class ChannelBooster(nn.Module):
    def __init__(self, in_c, expansion=5):
        super().__init__()
        mid_c = in_c * expansion
        self.net = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 1),
            nn.GELU(),
            nn.Conv2d(mid_c, in_c, 3, padding=1),
            nn.BatchNorm2d(in_c)
        )

    def forward(self, x):
        return x + self.net(x)



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1, use_gaussian=True):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 是否使用高斯核
        self.use_gaussian = use_gaussian
        if self.use_gaussian:

            self.register_buffer('gaussian_kernel', self._generate_gaussian_kernel(size=3, sigma=1.0))

        assert in_channels >= reduction_ratio, f"Input channels ({in_channels}) must ≥ reduction_ratio ({reduction_ratio})"

        mid_channels = max(in_channels // reduction_ratio, 1)

        self.fc1 = nn.Linear(in_channels, mid_channels, bias=False)
        self.relu = ReLU()
        self.fc2 = nn.Linear(mid_channels, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def _generate_gaussian_kernel(self, size, sigma):
        kernel = torch.zeros(size, size)
        center = size // 2
        variance = sigma ** 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center

                kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * variance)))

        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _apply_gaussian(self, x):

        if not self.use_gaussian:
            return x
            
        batch_size, channels, height, width = x.size()

        if torch.any(torch.lt(torch.tensor([height, width]), 3)):
            return x

        gaussian_kernel = self.gaussian_kernel.expand(channels, 1, -1, -1)
        smoothed = F.conv2d(x, gaussian_kernel, groups=channels, padding=1)
        return smoothed

    def forward(self, x):
        batch_size = x.size(0)
        channel_dim = x.size(1)
        
        # 应用高斯核平滑输入特征
        x_smoothed = self._apply_gaussian(x) if self.use_gaussian else x

        pooled = self.avg_pool(x_smoothed)
        channel_dim = pooled.size(1)
        avg_pooled = pooled.view(batch_size, -1)
        max_pooled = self.max_pool(x_smoothed).view(batch_size, channel_dim)

        if not torch.eq(torch.tensor(avg_pooled.size(1)), torch.tensor(self.fc1.in_features)).item():
            raise ValueError(
                f"Input dimension {avg_pooled.size(1)} does not match fc1's in_features {self.fc1.in_features}")

        avg_out = self.fc2(self.relu(self.fc1(avg_pooled)))
        max_out = self.fc2(self.relu(self.fc1(max_pooled)))

        attention = self.sigmoid(avg_out + max_out).view(batch_size, channel_dim, 1, 1)
        return attention * x, attention  # 返回注意力权重用于可视化, attention


class AudioClassificationModelCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1)

        self.booster1 = ChannelBooster(2)
        self.conv1 = Conv2d(2, 8, kernel_size=3, stride=2, padding=1)
        self.channel_attention1 = ChannelAttention(in_channels=8,
                                                   reduction_ratio=4,
                                                   use_gaussian=True)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm2d(8)
        self.dropout1 = Dropout(p=0.1)


        self.booster2 = ChannelBooster(8)
        self.conv2 = Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.channel_attention2 = ChannelAttention(in_channels=16,
                                                   reduction_ratio=8,
                                                   use_gaussian=True)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm2d(16)
        self.dropout2 = Dropout(p=0.1)


        self.booster3 = ChannelBooster(16)
        self.conv3 = Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.channel_attention3 = ChannelAttention(in_channels=32,
                                                   reduction_ratio=16,
                                                   use_gaussian=True)
        self.relu3 = ReLU()
        self.bn3 = BatchNorm2d(32)
        self.dropout3 = Dropout(p=0.1)

        self.booster4 = ChannelBooster(32)
        self.conv4 = Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.channel_attention4 = ChannelAttention(in_channels=64,
                                                   reduction_ratio=32,
                                                   use_gaussian=True)
        self.relu4 = ReLU()
        self.bn4 = BatchNorm2d(64)
        self.dropout4 = Dropout(p=0.1)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))


        hidden_dim = 96
        self.fc1 = Linear(64, hidden_dim)
        self.fc2 = Linear(hidden_dim, num_classes)

        # ====== 解码器：辅助重建任务 ======
        self.decoder = nn.Sequential(
            ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                            padding=1, output_padding=(0, 1)),
            ReLU(),
            BatchNorm2d(32),

            ConvTranspose2d(32, 16, kernel_size=(3, 5), stride=2, padding=1),
            ReLU(),
            BatchNorm2d(16),

            ConvTranspose2d(16, 8, kernel_size=(5, 7),
                            stride=(1, 2), padding=(2, 3)),
            ReLU(),
            BatchNorm2d(8),

            ConvTranspose2d(8, 1, kernel_size=(3, 5), padding=(1, 2)),


            nn.Upsample(size=(128, 65), mode='bilinear', align_corners=False)
        )

    def extract_features(self, x):
        assert x.ndim == 4, f"Input tensor must be 4D, got {x.shape}"


        x = self.stem(x)

        # Block1
        x = self.booster1(x)
        x = self.conv1(x)
        x, _ = self.channel_attention1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Block2
        x = self.booster2(x)
        x = self.conv2(x)
        x, _ = self.channel_attention2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.dropout2(x)

        # Block3
        x = self.booster3(x)
        x = self.conv3(x)
        x, _ = self.channel_attention3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.dropout3(x)

        # Block4
        x = self.booster4(x)
        x = self.conv4(x)
        x, _ = self.channel_attention4(x)
        x = self.relu4(x)
        x = self.bn4(x)
        x = self.dropout4(x)

        return x

    def forward(self, x):
        # 共享特征
        features = self.extract_features(x)

        # 主任务：分类
        pooled = self.ap(features).view(features.size(0), -1)
        h = F.relu(self.fc1(pooled))
        class_output = self.fc2(h)

        # 辅助任务：重建
        recon_output = self.decoder(features)

        return class_output, recon_output
