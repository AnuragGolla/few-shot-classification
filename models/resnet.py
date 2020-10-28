import torch
import torch.nn as nn

"""
Custom resnet model class definition.
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta


class ResNet(nn.Module):

    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=5, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'], param_dict[layer][block]['beta1'],
                                       param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    @property
    def output_size(self):
        return 512


class FilmResNet(ResNet):

    def __init__(self, block, layers):
        ResNet.__init__(self, block, layers)
        self.layers = layers

    def forward(self, x, param_dict):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
                                   param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
                                   param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
                                   param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
                                   param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

def load_resnet18(pretrained=False, fp=None, **kwargs):
    model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(fp)
        model.load_state_dict(ckpt_dict['state_dict'])
    return model



