# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import math
import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
from torch import nn
from torch.utils import model_zoo

from rga_modules import RGA_Module


def weights_init_kaiming(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('Conv') != -1:
		nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
		if m.bias is not None:
			nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0.0)


def weights_init_fc(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		nn.init.normal_(m.weight, std=0.001)
		nn.init.constant_(m.bias, 0.0)
	elif classname.find('BatchNorm') != -1:
		if m.affine:
			nn.init.normal_(m.weight, 1.0, 0.02)
			nn.init.constant_(m.bias, 0.0)



def init_pretrained_weights(model, model_url):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {
        k: v
        for k, v in pretrain_dict.items()
        if k in model_dict and model_dict[k].size() == v.size()
    }
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
							   padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(out_channels * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class RGA_Branch(nn.Module):
	def __init__(self, last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3],
                spa_on=True, cha_on=True, s_ratio=8, c_ratio=8, d_ratio=8, height=256, width=128,
                ):
		super(RGA_Branch, self).__init__()

		self.in_channels = 64

		# Networks
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
		self.layer1 = self._make_layer(block, 64, layers[0])
		self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512, layers[3], stride=last_stride)

		# RGA Modules
		self.rga_att1 = RGA_Module(256, (height//4)*(width//4), use_spatial=spa_on, use_channel=cha_on,
								cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga_att2 = RGA_Module(512, (height//8)*(width//8), use_spatial=spa_on, use_channel=cha_on,
								cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga_att3 = RGA_Module(1024, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
								cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)
		self.rga_att4 = RGA_Module(2048, (height//16)*(width//16), use_spatial=spa_on, use_channel=cha_on,
								cha_ratio=c_ratio, spa_ratio=s_ratio, down_ratio=d_ratio)


	def _make_layer(self, block, channels, blocks, stride=1):
		downsample = None
		if stride != 1 or self.in_channels != channels * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.in_channels, channels * block.expansion,
						  kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(channels * block.expansion),
			)

		layers = []
		layers.append(block(self.in_channels, channels, stride, downsample))
		self.in_channels = channels * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.in_channels, channels))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)
		x = self.layer1(x)
		x = self.rga_att1(x)

		x = self.layer2(x)
		x = self.rga_att2(x)

		x = self.layer3(x)
		x = self.rga_att3(x)

		x = self.layer4(x)
		x = self.rga_att4(x)

		return x

