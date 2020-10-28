import torch
import torch.nn as nn

"""
Adaptation set encoder model class definition.
"""

class Encoder(nn.Module):

    def __init__(self):
        super(SetEncoder, self).__init__()
        self.pre_pool_fn = PrePool()
        self.main_pool_fn = MainPool()
        self.post_pool_fn = PostPool()

    def forward(self, x):
        x = self.pre_pool_fn(x)
        x = self.main_pool_fn(x)
        x = self.post_pool_fn(x)
        return x

    def pool(x):
        return torch.mean(x, dim=0, keepdim=True)


class MainPool(nn.Module):

    def __init__(self):
        super(Pool, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=0, keepdim=True)


class PostPool(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PrePool(nn.Module):

    def __init__(self):
        super()PrePool, self).__init__()
        self.output_size = 64
        self.l1 = self.conv_block( 3, 64)
        self.l2 = self.conv_block(64, 64)
        self.l3 = self.conv_block(64, 64)
        self.l4 = self.conv_block(64, 64)
        self.l5 = self.conv_block(64, 64)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @staticmethod
    def conv_block(inp, out):
        return nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



