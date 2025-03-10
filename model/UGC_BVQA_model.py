import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
#vgg注释了 resnet不行的话要改回来from charset_normalizer import models
from torch.hub import load_state_dict_from_url
import torch
import torch.nn as nn
from torchvision import models
from timm import create_model
from timm.models import swin_transformer
# from thop import profile


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}



def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
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


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #layers=[3,4,6,3]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # three stage spatial features (avg + std) + motion  输入通道 中间通道 输出通道
        self.quality = self.quality_regression(4096+2048+1024+2048+256, 128,1)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),          
        )

        return regression_block


    def _forward_impl(self, x, x_3D_features):
        # See note [TorchScript super()]
        # input dimension: batch x frames x 3 x height x width

        x_size = x.shape
        #print(x_size)
        # x_3D: batch x frames x (2048 + 256)
        x_3D_features_size = x_3D_features.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x (2048 + 256)
        x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        # x_AADB_features_size:batch * frames x （256x6）



        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_avg2 = self.avgpool(x)
        x_std2 = global_std_pool2d(x)

        x = self.layer3(x)
        x_avg3 = self.avgpool(x)
        x_std3 = global_std_pool2d(x)

        x = self.layer4(x)
        x_avg4 = self.avgpool(x)
        x_std4 = global_std_pool2d(x)
        x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim = 1)
        #print(x.shape)
        # x: batch * frames x (2048*2 + 1024*2 + 512*2)
        x = torch.flatten(x, 1)
        # x: batch * frames x (2048*2 + 1024*2 + 512*2 + 2048 + 256 +256*6)
        x = torch.cat((x, x_3D_features), dim = 1)
        # x: batch * frames x 1
        x = self.quality(x)
        # x: batch x frames
        x = x.view(x_size[0],x_size[1])
        # x: batch x 1
        x = torch.mean(x, dim = 1)
            
        return x

    def forward(self, x, x_3D_features):
        return self._forward_impl(x, x_3D_features)


class VideoSwinTransformer(nn.Module):
    def __init__(self, pretrained_weights_path=None):
        super(VideoSwinTransformer, self).__init__()

        # 使用 timm 库加载预训练的 Swin Transformer
        self.swin_transformer = create_model('swin_base_patch4_window7_224', pretrained=False)

        # 如果有本地预训练权重，则加载
        if pretrained_weights_path is not None:
            self.load_pretrained_weights(pretrained_weights_path)

        # 修改输出层为回归层
        self.fc = nn.Linear(1024, 1)  # 回归层，输出一个值

    def load_pretrained_weights(self, path):
        # 加载本地预训练权重
        state_dict = torch.load(path)
        self.swin_transformer.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # 输入形状应为 (batch_size, num_frames, num_channels, height, width)
        batch_size, num_frames, num_channels, height, width = x.size()

        # 交换维度以适应 Swin Transformer 的输入格式
        x = x.permute(0, 2, 1, 3, 4)  # (batch_size, num_channels, num_frames, height, width)

        # 展平为适合 Swin Transformer 的格式
        x = x.reshape(-1, num_channels, height, width)  # (batch_size * num_frames, num_channels, height, width)

        # 经过 Swin Transformer
        x = self.swin_transformer(x)  # 输出形状应为 (batch_size * num_frames, 1024)
        print(x.shape)

        # 将输出还原为 (batch_size, num_frames, 1024)
        x = x.view(batch_size, num_frames, -1)  # (batch_size, num_frames, 1024)

        # 对每个视频的特征进行平均池化以生成一个特征
        x = x.mean(dim=1)  # (batch_size, 1024)

        # 经过回归层
        x = self.fc(x)  # 输出 (batch_size, 1)
        return x

class VGG19(nn.Module):
    def __init__(self, pretrained=True, weights_path=None):
        super(VGG19, self).__init__()
        # 加载预训练的VGG19模型
        self.vgg19 = models.vgg19(pretrained=False)
        if pretrained:
            if weights_path:
                state_dict = torch.load(weights_path)
                self.vgg19.load_state_dict(state_dict)
            else:
                self.vgg19 = models.vgg19(pretrained=True)

        # 移除分类器
        self.features = self.vgg19.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.quality = self.quality_regression(100352, 128, 1)



    def forward(self, x):
        x_size = x.shape
        #print(x_size)
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        x = self.features(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        x = self.quality(x)
        x = x.view(x_size[0], x_size[1])
        x = torch.mean(x, dim=1)

        return x

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block


class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True , weights_path=None):
        super(ViTFeatureExtractor, self).__init__()
        # 加载 ViT 模型
        self.model = create_model(model_name, pretrained=False)  # 不下载预训练权重

        # 如果提供了预训练权重的路径，加载本地权重
        if pretrained:
            if weights_path:
                state_dict = torch.load(weights_path)
                self.model.load_state_dict(state_dict, strict=False)

        # 移除最后的分类头，只保留特征提取部分
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])
        self.quality = self.quality_regression(150528, 128, 1)

    def forward(self, x):
        x_size = x.shape

        x = x.view(-1, x_size[2], x_size[3], x_size[4])

        # 获取特征
        x = self.feature_extractor(x)

        x = torch.flatten(x, 1)
        x = self.quality(x)
        x = x.view(x_size[0], x_size[1])
        x = torch.mean(x, dim=1)

        return x

    def quality_regression(self,in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block


class ResNet_train(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet_train, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.k1 = nn.Parameter(torch.Tensor([0.5]))  # 初始化为0.5，但可以是任何值
        self.k2 = nn.Parameter(torch.Tensor([0.5]))
        self.k3 = nn.Parameter(torch.Tensor([0.5]))
        self.k4 = nn.Parameter(torch.Tensor([0.5]))

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # layers=[3,4,6,3]
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # three stage spatial features (avg + std) + motion  输入通道 中间通道 输出通道
        self.quality = self.quality_regression(4096 + 2048 + 1024 , 128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        # input dimension: batch x frames x 3 x height x width

        x_size = x.shape
        #print(x_size)
        # x_3D: batch x frames x (2048 + 256)
        #x_3D_features_size = x_3D_features.shape
        # x: batch * frames x 3 x height x width
        x = x.view(-1, x_size[2], x_size[3], x_size[4])
        # x_3D: batch * frames x (2048 + 256)
        #x_3D_features = x_3D_features.view(-1, x_3D_features_size[2])
        # x_AADB_features_size:batch * frames x （256x6）

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x_avg2 = self.avgpool(x)
        x_std2 = global_std_pool2d(x)

        x = self.layer3(x)
        x_avg3 = self.avgpool(x)
        x_std3 = global_std_pool2d(x)

        x = self.layer4(x)
        x_avg4 = self.avgpool(x)
        x_std4 = global_std_pool2d(x)
        x = torch.cat((x_avg2, x_std2, x_avg3, x_std3, x_avg4, x_std4), dim=1)
       # print(x.shape)
        # x: batch * frames x (2048*2 + 1024*2 + 512*2)
        x = torch.flatten(x, 1)
        # x: batch * frames x (2048*2 + 1024*2 + 512*2 + 2048 + 256 )
        #x = torch.cat((x, x_3D_features), dim=1)
        # x: batch * frames x 1
        x = self.quality(x)
        # x: batch x frames
        x = x.view(x_size[0], x_size[1])
        # x: batch x 1
        x = torch.mean(x, dim=1)

        return x


    def _forward_impl2(self, x1, x2, x3, x4):

        x1_size = x1.shape
        x2_size = x2.shape
        x3_size = x3.shape
        x4_size = x4.shape
        x1 = x1.view(-1, x1_size[2], x1_size[3], x1_size[4])
        x2 = x1.view(-1, x2_size[2], x2_size[3], x2_size[4])
        x3 = x1.view(-1, x3_size[2], x3_size[3], x3_size[4])
        x4 = x1.view(-1, x4_size[2], x4_size[3], x4_size[4])

        #.....x1

        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)

        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1_avg2 = self.avgpool(x1)
        x1_std2 = global_std_pool2d(x1)

        x1 = self.layer3(x1)
        x1_avg3 = self.avgpool(x1)
        x1_std3 = global_std_pool2d(x1)

        x1 = self.layer4(x1)
        x1_avg4 = self.avgpool(x1)
        x1_std4 = global_std_pool2d(x1)
        x1 = torch.cat((x1_avg4, x1_std4), dim=1) #batch*framex512

        x1 = torch.flatten(x1, 1)
        #......x2
        x2 = self.conv1(x2)
        x2 = self.bn1(x2)
        x2 = self.relu(x2)
        x2 = self.maxpool(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2_avg2 = self.avgpool(x2)
        x2_std2 = global_std_pool2d(x2)
        x2 = self.layer3(x2)
        x2_avg3 = self.avgpool(x2)
        x2_std3 = global_std_pool2d(x2)
        x2 = self.layer4(x2)
        x2_avg4 = self.avgpool(x2)
        x2_std4 = global_std_pool2d(x2)
        x2 = torch.cat((x2_avg4, x2_std4), dim=1)  # batch*framex512
        x2 = torch.flatten(x2, 1)
        #.......x3
        x3 = self.conv1(x3)
        x3 = self.bn1(x3)
        x3 = self.relu(x3)
        x3 = self.maxpool(x3)
        x3 = self.layer1(x3)
        x3 = self.layer2(x3)
        x3_avg2 = self.avgpool(x3)
        x3_std2 = global_std_pool2d(x3)
        x3 = self.layer3(x3)
        x3_avg3 = self.avgpool(x3)
        x3_std3 = global_std_pool2d(x3)
        x3 = self.layer4(x3)
        x3_avg4 = self.avgpool(x3)
        x3_std4 = global_std_pool2d(x3)
        x3 = torch.cat((x3_avg4, x3_std4), dim=1)
        x3 = torch.flatten(x3, 1)
        #.....x4
        x4 = self.conv1(x4)
        x4 = self.bn1(x4)
        x4 = self.relu(x4)
        x4 = self.maxpool(x4)
        x4 = self.layer1(x4)
        x4 = self.layer2(x4)
        x4_avg2 = self.avgpool(x4)
        x4_std2 = global_std_pool2d(x4)
        x4 = self.layer3(x4)
        x4_avg3 = self.avgpool(x4)
        x4_std3 = global_std_pool2d(x4)
        x4 = self.layer4(x4)
        x4_avg4 = self.avgpool(x4)
        x4_std4 = global_std_pool2d(x4)
        x4 = torch.cat((x4_avg4, x4_std4), dim=1)
        x4 = torch.flatten(x4, 1)

        x = self.k1 * x1 + self.k2 * x2 + self.k3 * x3 + self.k4 * x4

        return x

    def forward(self, x):
        return self._forward_impl(x)

def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet34'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model

def resnet50_train(pretrained=False, progress=True, **kwargs):

    model = ResNet_train(Bottleneck, [3, 4, 6, 3], **kwargs)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # print('The flops is {:.4f}, and the params is {:.4f}'.format(flops/10e9, params/10e6))
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet50'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model

def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(model, inputs=(input, ))
    # print('The flops is {:.4f}, and the params is {:.4f}'.format(flops/10e9, params/10e6))
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet50'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
    #                **kwargs)
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet101'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
    #                **kwargs)
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnet152'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    #return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
       #            pretrained, progress, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnext50_32x4d'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    # return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
    #                pretrained, progress, **kwargs)
    model = ResNet(Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
    if pretrained:
        model_dict = model.state_dict()
        pre_train_model = model_zoo.load_url(model_urls['resnext101_32x8d'])
        pre_train_model = {k:v for k,v in pre_train_model.items() if k in model_dict}
        model_dict.update(pre_train_model)
        model.load_state_dict(model_dict)
    return model


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)




if __name__ == "__main__":

    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    model = resnet50(pretrained=False).to(device)
    # print(model)
    from thop import profile
    from thop import clever_format

    input = torch.randn(8,8,3,448,448)
    input_3D = torch.randn(8, 8, 2048+256)
    flops, params = profile(model, inputs=(input,input_3D,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops)
    print(params)
