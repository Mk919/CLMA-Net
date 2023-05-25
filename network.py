"""该模块中用于保存网络模型"""
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import spectral
import numpy
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.decomposition import PCA



"""----------------------------------------我的网络3-------------------------------------------------"""


class ChannelAttention_LliuMK3(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(ChannelAttention_LliuMK3, self).__init__()
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // self.rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // self.rotio, in_planes, (1, 1), bias=False))
        self.sigmoid = nn.Sigmoid()
        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x):
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out
        return self.sigmoid(out)


class MultiChannelAttention_LliuMK3(nn.Module):
    def __init__(self, in_planes, rotio=2):
        super(MultiChannelAttention_LliuMK3, self).__init__()
        self.in_planes = in_planes
        self.rotio = rotio

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // rotio, (1, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // rotio, in_planes, (1, 1), bias=False))

        self.conv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.in_planes, kernel_size=(1, 1),
                              stride=(1, 1), padding=(0, 0))
        self.batch_norm = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU()
        # self.up_sample_fc = nn.Sequential(nn.Linear(in_features=in_planes, out_features=in_planes),
        #                                   nn.BatchNorm1d(num_features=in_planes),
        #                                   nn.ReLU()
        #                                   )
        self.sigmoid = nn.Sigmoid()

        self.coefficient1 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient2 = torch.nn.Parameter(torch.Tensor([1.0]))
        self.coefficient3 = torch.nn.Parameter(torch.Tensor([1.0]))

    def forward(self, x, shallow_channel_attention_map):  # 64,103,3,3   64,103,1,1
        avg_out = self.sharedMLP(self.avg_pool(x))
        max_out = self.sharedMLP(self.max_pool(x))

        x2 = shallow_channel_attention_map  # 64,103,1,1 -> 64,1,103,1

        x2 = self.conv(x2)
        x2 = self.batch_norm(x2)
        x2 = self.relu(x2)
        x2 = self.sharedMLP(x2)
        out = self.coefficient1 * avg_out + self.coefficient2 * max_out + self.coefficient3 * x2
        return self.sigmoid(out)


class SpatialAttention_LliuMK3(nn.Module):
    def __init__(self):
        super(SpatialAttention_LliuMK3, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # [16,32,9,9]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [16, 1, 9, 9]
        x_out = torch.cat([avg_out, max_out], dim=1)  # [16,2,9,9]                                    # 按维数1（列）拼接
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)


class MultiSpatialAttention_LliuMK3(nn.Module):
    def __init__(self):
        super(MultiSpatialAttention_LliuMK3, self).__init__()
        self.conv = nn.Conv2d(3, 1, kernel_size=(3, 3), padding=(1, 1), bias=False)
        self.sigmoid = nn.Sigmoid()
        self.conv0 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.batch_norm0 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU()

    def forward(self, x, shallow_spatial_attention_map):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x0 = self.conv0(shallow_spatial_attention_map)
        x0 = self.batch_norm0(x0)
        x0 = self.relu(x0)
        x0 = nn.AvgPool2d(2)(x0)  # 最大池化下采样效果貌似稍微好一点，还需要进一步实验验证
        x = torch.cat([avg_out, max_out, x0], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class CLMA_Net(nn.Module):
    def __init__(self, classes, HSI_Data_Shape_H, HSI_Data_Shape_W, HSI_Data_Shape_C):  # band:103  classes=9
        super(CLMA_Net, self).__init__()
        self.name = 'CLMA_Net'
        self.classes = classes
        self.HSI_Data_Shape_H = HSI_Data_Shape_H
        self.HSI_Data_Shape_W = HSI_Data_Shape_W
        self.band = HSI_Data_Shape_C

        # self.mish = mish()  # 也可以引用一下，等待后续改进
        self.relu = nn.ReLU()

        self.CA1 = ChannelAttention_LliuMK3(in_planes=self.band)

        self.MCA1 = MultiChannelAttention_LliuMK3(in_planes=self.band)
        self.MCA2 = MultiChannelAttention_LliuMK3(in_planes=self.band)

        self.SA1 = SpatialAttention_LliuMK3()

        self.MSA1 = MultiSpatialAttention_LliuMK3()
        self.MSA2 = MultiSpatialAttention_LliuMK3()
        self.conv11 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm11 = nn.BatchNorm2d(self.band)

        self.conv12 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm12 = nn.BatchNorm2d(self.band)

        self.conv13 = nn.Conv2d(in_channels=self.band, out_channels=self.band, kernel_size=(1, 1),
                                stride=(1, 1), padding=(0, 0))
        self.batch_norm13 = nn.BatchNorm2d(self.band)

        self.conv21 = nn.Conv2d(in_channels=self.band, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm21 = nn.BatchNorm2d(64)

        self.conv22 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.batch_norm22 = nn.BatchNorm2d(128)

        self.conv23 = nn.Conv2d(in_channels=128, out_channels=self.band, kernel_size=(3, 3), stride=(1, 1),
                                padding=(1, 1))
        self.batch_norm23 = nn.BatchNorm2d(self.band)

        self.global_pooling = nn.AdaptiveAvgPool2d(1)

        self.finally_fc_classification = nn.Linear(self.band * 2, self.classes)

    def forward(self, patchX, pixelX):  # x:(16,103,9,9)
        """------------------------光谱分支------------------------"""
        patch_size = patchX.shape[-1] // 2
        input_spectral = patchX[:, :, (patch_size-1):(patch_size + 2), (patch_size-1):(patch_size + 2)]  # [64,103,3,3]

        x11 = self.conv11(input_spectral)
        x11 = self.batch_norm11(x11)  # [64,103,3,3]
        x11 = self.relu(x11)
        ca1 = self.CA1(x11)
        x11 = x11 * ca1

        x12 = x11
        x12 = self.conv12(x12)
        x12 = self.batch_norm12(x12)
        x12 = self.relu(x12)
        mca1 = self.MCA1(x12, ca1)
        x12 = x12 * mca1

        x13 = x12
        x13 = self.conv13(x13)
        x13 = self.batch_norm13(x13)
        x13 = self.relu(x13)
        mca2 = self.MCA2(x13, mca1)
        x13 = x13 * mca2

        x13 = self.global_pooling(x13)
        x13 = x13.view(x13.size(0), -1)
        output_spectral = x13

        """------------------------空间分支------------------------"""
        input_spatial = patchX
        x21 = self.conv21(input_spatial)  # (16,32,9,9)<—(16,103,9,9)
        x21 = self.batch_norm21(x21)  # (16,32,9,9)
        x21 = self.relu(x21)  # (16,32,9,9)
        sa1 = self.SA1(x21)
        x21 = x21 * sa1
        x21 = nn.MaxPool2d(2)(x21)

        x22 = self.conv22(x21)  # (16,24,1,9,9)
        x22 = self.batch_norm22(x22)  # (16,24,1,9,9)
        x22 = self.relu(x22)
        msa1 = self.MSA1(x22, sa1)
        x22 = x22 * msa1

        x22 = nn.MaxPool2d(2)(x22)

        x23 = self.conv23(x22)  # (16,24,1,9,9)
        x23 = self.batch_norm23(x23)  # (16,24,1,9,9)
        x23 = self.relu(x23)
        msa2 = self.MSA2(x23, msa1)
        x23 = x23 * msa2

        x23 = nn.MaxPool2d(2)(x23)

        x25 = self.global_pooling(x23)
        x25 = x25.view(x25.size(0), -1)
        output_spatial = x25

        output = torch.cat((output_spectral, output_spatial), dim=1)
        output = self.finally_fc_classification(output)
        output = F.softmax(output, dim=1)

        return output, output


