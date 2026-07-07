import torch.nn as nn


class PostActivationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride=1, kernel_size=3):
        super(PostActivationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        if first_stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + shortcut)

class PreActivationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride=1, kernel_size=3):
        super(PreActivationBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

        self.shortcut = (
            nn.Identity() if first_stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride, padding=0, bias=False)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return out + shortcut

class NoBatchNormBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first_stride=1, kernel_size=3):
        super(NoBatchNormBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=first_stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=True)

        self.relu = nn.ReLU()
        self.shortcut = (
            nn.Identity() if first_stride == 1 and in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=first_stride, padding=0, bias=True)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.conv1(self.relu(x))
        out = self.conv2(self.relu(out))
        return out + shortcut

class RestNet20(nn.Module):
    def __init__(self,net_type="pre",num_classes=10):
        super(RestNet20, self).__init__()

        if net_type == "pre":
            self.block_type = PreActivationBlock
        elif net_type == "post":
            self.block_type = PostActivationBlock
        elif net_type == "nobn":
            self.block_type = NoBatchNormBlock
        else:
            raise NotImplementedError

        if net_type == "nobn":
            self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        elif net_type == "pre":
            self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True)
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(),
            )

        self.stage1 = self._make_stage(layers=3, in_channels=16, out_channels=16, first_stride=1)
        self.stage2 = self._make_stage(layers=3, in_channels=16, out_channels=32, first_stride=2)
        self.stage3 = self._make_stage(layers=3, in_channels=32, out_channels=64, first_stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(64, num_classes)

    def _make_stage(self, layers, in_channels, out_channels, first_stride=1):
        blocks = []
        for i in range(layers):
            blocks.append(self.block_type(in_channels if i == 0 else out_channels, out_channels, first_stride if i == 0 else 1))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x).flatten(1)
        x = self.fc(x)
        return x