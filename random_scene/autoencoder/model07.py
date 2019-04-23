from __future__ import print_function
import os
from os.path import join
import argparse
import logging
import math
from statistics import mean, stdev
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import dataloader as dl


class ModelLoss(nn.Module):
    def __init__(self, device, value_edge_weight=0.7, remap_weight=0.2):
        super(ModelLoss, self).__init__()
        self.edge_filter_4 = ModelLoss.generate_filter(depth=4).to(device)
        self.edge_filter_15 = ModelLoss.generate_filter(depth=15).to(device)
        self.value_edge_weight = value_edge_weight
        self.remap_weight = remap_weight

    @staticmethod
    def generate_filter(depth=4):
        f = Variable(torch.FloatTensor([[[[-1 / 8, -1 / 8, -1 / 8],
                                          [-1 / 8,  8 / 8, -1 / 8],
                                          [-1 / 8, -1 / 8, -1 / 8]]]]),
                     requires_grad=False) / float(depth)
        ft = (f,)
        for i in range(depth-1):
            ft = ft + (f,)
        return torch.cat(ft, dim=1)

    def forward(self, input, target, remap_input, remap_target, reduction='mean'):
        value_l1_loss = F.l1_loss(input, target, reduction=reduction)
        input_log_edges = F.conv2d(input, self.edge_filter_4, padding=1)
        target_log_edges = F.conv2d(target, self.edge_filter_4, padding=1)
        edge_l1_loss = F.l1_loss(input_log_edges, target_log_edges, reduction=reduction)
        primary_l1_loss = value_l1_loss * self.value_edge_weight + edge_l1_loss * (1.0 - self.value_edge_weight)

        remap_value_loss = F.l1_loss(remap_input, remap_target, reduction=reduction)
        remap_input_edges = F.conv2d(remap_input, self.edge_filter_15, padding=1)
        remap_target_edges = F.conv2d(remap_target, self.edge_filter_15, padding=1)
        remap_edge_loss = F.l1_loss(remap_input_edges, remap_target_edges, reduction=reduction)
        remap_l1_loss = remap_value_loss * self.value_edge_weight + remap_edge_loss * (1.0 - self.value_edge_weight)

        return primary_l1_loss * (1.0 - self.remap_weight) + remap_l1_loss * self.remap_weight


class UpsampleInterpolate(nn.Module):
    def __init__(self, scale_factor=(2.0,2.0), mode=None):
        super(UpsampleInterpolate, self).__init__()
        if mode is None:
            self.mode = 'nearest'
            if isinstance(scale_factor, tuple):
                for i in range(len(scale_factor)):
                    if abs(scale_factor[i]) < 1.0:
                        self.mode = 'area'
            else:
                if abs(scale_factor) < 1.0:
                    self.mode = 'area'
        else:
            self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class SelfAttention(nn.Module):
    def __init__(self, inplanes, query_planes=None, resample_kernel=1):
        super().__init__()

        self.query_planes = query_planes or inplanes // 8

        self.query = nn.Conv1d(inplanes, self.query_planes, 1)
        self.key = nn.Conv1d(inplanes, self.query_planes, 1)
        self.value = nn.Conv1d(inplanes, inplanes, 1)

        self.gamma = nn.Parameter(torch.Tensor([0.0]).squeeze())

        if resample_kernel > 1:
            self.downsample = nn.MaxPool2d(kernel_size=resample_kernel)
            self.upsample = UpsampleInterpolate(scale_factor=resample_kernel)
        else:
            self.downsample = None
            self.upsample = None

    def forward(self, input):
        x = input
        if self.downsample is not None:
            x = self.downsample(x)

        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attention = F.softmax(query_key, 1)
        attention = torch.bmm(value, attention)
        attention = attention.view(*shape)
        if self.upsample is not None:
            attention = self.upsample(attention)
        out = self.gamma * attention + input

        return out


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    @staticmethod
    def l2normalize(v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = SpectralNorm.l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = SpectralNorm.l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = SpectralNorm.l2normalize(u.data)
        v.data = SpectralNorm.l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample_method=None,
                 attention_kernel=None, activation=nn.ReLU(inplace=True),
                 spectral_normalization=False):
        super(ConvBlock, self).__init__()

        if spectral_normalization:
            if downsample_method == "conv":
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation
                )
                self.downsample = nn.Sequential(
                    SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False)),
                    nn.BatchNorm2d(planes)
                )
            elif downsample_method == "maxpool":
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation,
                    nn.MaxPool2d(kernel_size=2)
                )
                self.downsample = nn.Sequential(
                    SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False)),
                    nn.BatchNorm2d(planes)
                )
            else:
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation
                )
                self.downsample = None

            self.stage2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(planes))
        else:
            if downsample_method == "conv":
                self.stage1 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(planes),
                    activation
                )
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(planes)
                )
            elif downsample_method == "maxpool":
                self.stage1 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(planes),
                    activation,
                    nn.MaxPool2d(kernel_size=2)
                )
                self.downsample = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(planes)
                )
            else:
                self.stage1 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    activation
                )
                self.downsample = None

            self.stage2 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes))

        self.activation = activation
        if attention_kernel is not None:
            self.attention = SelfAttention(planes, resample_kernel=attention_kernel)
        else:
            self.attention = None

    def forward(self, x):
        residual = x

        out = self.stage1(x)
        out = self.stage2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class ConvTransposeBlock(nn.Module):

    def __init__(self, inplanes, planes, upsample_method=None,
                 attention_kernel=None, activation=nn.ReLU(inplace=True),
                 spectral_normalization=False):
        super(ConvTransposeBlock, self).__init__()

        if spectral_normalization:
            if upsample_method == "conv":
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation
                )
            elif upsample_method == "interpolate":
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation,
                    UpsampleInterpolate()
                )
            else:
                self.stage1 = nn.Sequential(
                    SpectralNorm(nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
                    nn.BatchNorm2d(planes),
                    activation
                )

            self.stage2 = nn.Sequential(
                SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
                nn.BatchNorm2d(planes)
            )
        else:
            if upsample_method == "conv":
                self.stage1 = nn.Sequential(
                    nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(planes),
                    activation
                )
            elif upsample_method == "interpolate":
                self.stage1 = nn.Sequential(
                    nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(planes),
                    activation,
                    UpsampleInterpolate()
                )
            else:
                self.stage1 = nn.Sequential(
                    nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.BatchNorm2d(planes),
                    activation
                )

            self.stage2 = nn.Sequential(
                nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes)
            )
        self.activation = activation
        if attention_kernel is not None:
            self.attention = SelfAttention(planes, resample_kernel=attention_kernel)
        else:
            self.attention = None

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.activation(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class Net(nn.Module):
    def __init__(self, downsample_method='conv', upsample_method='conv', input_resolution=(256, 256),
                 output_resolution=None, attention=None, extra_layers=None, activation=nn.ReLU(inplace=True)):
        super(Net, self).__init__()

        if input_resolution < (32, 32):
            raise ValueError('target resolution should have dimensions at least (32,32)')
        if downsample_method not in ['maxpool', 'conv']:
            raise ValueError('expecting downsample method conv or maxpool')
        if upsample_method not in ['interpolate', 'conv']:
            raise ValueError('expecting upsample method interpolate or maxpool')
        if output_resolution is None:
            output_resolution = input_resolution
        if attention is None:
            attention = {}

        # Input: b, 3, 1, 1
        self.position_block0 = nn.Sequential(
            nn.ConvTranspose2d(3, 6, 2, stride=1, padding=0),  # b, 6, 2, 2
            nn.BatchNorm2d(6),
            activation,
            UpsampleInterpolate(scale_factor=(input_resolution[0] / 64,
                                              input_resolution[1] / 64))
        )  # b, 6, 8/sf, 8/sf
        self.position_block1 = Net.build_decoder_block(
            6, 6, upsample_method,
            extra_layers=extra_layers['p1'] if 'p1' in extra_layers else None,
            attention=attention['p1'] if 'p1' in attention else None
        )  # b, 6, 16/sf, 16/sf
        self.position_block2 = Net.build_decoder_block(
            6, 6, upsample_method,
            extra_layers=extra_layers['p2'] if 'p2' in extra_layers else None,
            attention=attention['p2'] if 'p2' in attention else None
        )  # b, 6, 32/sf, 32/sf
        self.position_block3 = Net.build_decoder_block(
            6, 6, upsample_method,
            extra_layers=extra_layers['p3'] if 'p3' in extra_layers else None,
            attention=attention['p3'] if 'p3' in attention else None
        )  # b, 6, 64/sf, 64/sf
        self.position_block4 = Net.build_decoder_block(
            6, 6, upsample_method,
            extra_layers=extra_layers['p4'] if 'p4' in extra_layers else None,
            attention=attention['p4'] if 'p4' in attention else None
        )  # b, 6, 128/sf, 128/sf
        self.position_block5 = Net.build_decoder_block(
            6, 6, upsample_method,
            extra_layers=extra_layers['p5'] if 'p5' in extra_layers else None,
            attention=attention['p5'] if 'p5' in attention else None
        )  # b, 6, 256/sf, 256/sf

        # 6 = RGB Left, RGB Right
        # Input: b, 6, 256/super_res_factor, 256/super_res_factor
        self.encoder_block0 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            activation
        )  # b, 32, 256/sf, 256/sf
        self.encoder_block1 = Net.build_encoder_block(
            32, 32, None,
            extra_layers=extra_layers['e1'] if 'e1' in extra_layers else None,
            attention=attention['e1'] if 'e1' in attention else None
        )  # b, 32, 256/sf, 256/sf
        self.encoder_block2 = Net.build_encoder_block(
            32, 43, downsample_method,
            extra_layers=extra_layers['e2'] if 'e2' in extra_layers else None,
            attention=attention['e2'] if 'e2' in attention else None
        )  # b, 43, 128/sf, 128/sf
        self.encoder_block3 = Net.build_encoder_block(
            43, 57, downsample_method,
            extra_layers=extra_layers['e3'] if 'e3' in extra_layers else None,
            attention=attention['e3'] if 'e3' in attention else None
        )  # b, 57, 64/sf, 64/sf
        self.encoder_block4 = Net.build_encoder_block(
            57, 76, downsample_method,
            extra_layers=extra_layers['e4'] if 'e4' in extra_layers else None,
            attention=attention['e4'] if 'e4' in attention else None
        )  # b, 76, 32/sf, 32/sf
        self.encoder_block5 = Net.build_encoder_block(
            76, 101, downsample_method,
            extra_layers=extra_layers['e5'] if 'e5' in extra_layers else None,
            attention=attention['e5'] if 'e5' in attention else None
        )  # b, 101, 16/sf, 16/sf
        self.encoder_block6 = Net.build_encoder_block(
            101, 135, downsample_method,
            extra_layers=extra_layers['e6'] if 'e6' in extra_layers else None,
            attention=None
        )  # b, 135, 8/sf, 8/sf

        self.super_res_upsample = nn.Sequential(
            UpsampleInterpolate(scale_factor=(output_resolution[0] / input_resolution[0],
                                              output_resolution[1] / input_resolution[1]))
        )

        # Warped decoder
        # Input: b, 135+6, 8/sf, 8/sf
        self.decoder_block1 = Net.build_decoder_block(
            135 + 6, 101, upsample_method,
            extra_layers=extra_layers['d1'] if 'd1' in extra_layers else None,
            attention=attention['d1'] if 'd1' in attention else None
        )  # b, 101, 16/sf, 16/sf
        self.decoder_block2 = Net.build_decoder_block(
            101 * 2 + 6, 76, upsample_method,
            extra_layers=extra_layers['d2'] if 'd2' in extra_layers else None,
            attention=attention['d2'] if 'd2' in attention else None
        )  # b, 76, 32/sf, 32/sf
        self.decoder_block3 = Net.build_decoder_block(
            76 * 2 + 6, 57, upsample_method,
            extra_layers=extra_layers['d3'] if 'd3' in extra_layers else None,
            attention=attention['d3'] if 'd3' in attention else None
        )  # b, 57, 64/sf, 64/sf
        self.decoder_block4 = Net.build_decoder_block(
            57 * 2 + 6, 43, upsample_method,
            extra_layers=extra_layers['d4'] if 'd4' in extra_layers else None,
            attention=attention['d4'] if 'd4' in attention else None
        )  # b, 43, 128/sf, 128/sf
        self.decoder_block5 = Net.build_decoder_block(
            43 * 2 + 6, 32, upsample_method,
            extra_layers=extra_layers['d5'] if 'd5' in extra_layers else None,
            attention=attention['d5'] if 'd5' in attention else None
        )  # b, 32, 256/sf, 256/sf
        self.decoder_block6 = Net.build_decoder_block(
            32 * 2 + 6, 32, None,
            extra_layers=extra_layers['d6'] if 'd6' in extra_layers else None,
            attention=attention['d6'] if 'd6' in attention else None
        )  # b, 32, 256/sf, 256/sf
        # Upsample by sf between 6 and 7
        self.decoder_block7 = Net.build_decoder_block(
            32, 32, None,
            extra_layers=extra_layers['d7'] if 'd7' in extra_layers else None,
            attention=attention['d7'] if 'd7' in attention else None
        )  # b, 32, 256, 256
        self.decoder_output = nn.Sequential(
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )  # b, 4, 256, 256

        # Reprojection decoder
        # Input: b, 135+6, 8/sf, 8/sf
        self.reproj_block1 = Net.build_decoder_block(
            135 + 6, 101, upsample_method,
            extra_layers=extra_layers['r1'] if 'r1' in extra_layers else None,
            attention=attention['r1'] if 'r1' in attention else None
        )  # b, 101, 16/sf, 16/sf
        self.reproj_block2 = Net.build_decoder_block(
            101 * 2 + 6, 76, upsample_method,
            extra_layers=extra_layers['r2'] if 'r2' in extra_layers else None,
            attention=attention['r2'] if 'r2' in attention else None
        )  # b, 76, 32/sf, 32/sf
        self.reproj_block3 = Net.build_decoder_block(
            76 * 2 + 6, 57, upsample_method,
            extra_layers=extra_layers['r3'] if 'r3' in extra_layers else None,
            attention=attention['r3'] if 'r3' in attention else None
        )  # b, 57, 64/sf, 64/sf
        self.reproj_block4 = Net.build_decoder_block(
            57 * 2 + 6, 43, upsample_method,
            extra_layers=extra_layers['r4'] if 'r4' in extra_layers else None,
            attention=attention['r4'] if 'r4' in attention else None
        )  # b, 43, 128/sf, 128/sf
        self.reproj_block5 = Net.build_decoder_block(
            43 * 2 + 6, 32, upsample_method,
            extra_layers=extra_layers['r5'] if 'r5' in extra_layers else None,
            attention=attention['r5'] if 'r5' in attention else None
        )  # b, 32, 256/sf, 256/sf
        self.reproj_block6 = Net.build_decoder_block(
            32 * 2 + 6, 32, None,
            extra_layers=extra_layers['r6'] if 'r6' in extra_layers else None,
            attention=attention['r6'] if 'r6' in attention else None
        )
        r6_planes = 32
        self.reproj_left_output = Net.build_reproj_block(r6_planes, activation, layers=6)
        self.reproj_right_output = Net.build_reproj_block(r6_planes, activation, layers=6)
        self.reproj_up_output = Net.build_reproj_block(r6_planes, activation, layers=6)
        self.reproj_down_output = Net.build_reproj_block(r6_planes, activation, layers=6)
        self.reproj_straight_output = Net.build_reproj_block(r6_planes, activation, layers=6)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def build_encoder_block(inplanes, outplanes, downsample_method, extra_layers, attention):
        layer_list = [('encode0', ConvBlock(inplanes, outplanes, downsample_method=downsample_method))]
        extra_layers = 0 if extra_layers is None else extra_layers
        for i in range(extra_layers):
            layer_list.append(('encode'+str(i+1), ConvBlock(outplanes, outplanes)))
        layer_list.append(('encode'+str(extra_layers+1),ConvBlock(outplanes, outplanes, attention_kernel=attention)))
        return nn.Sequential(OrderedDict(layer_list))

    @staticmethod
    def build_decoder_block(inplanes, outplanes, upsample_method, extra_layers, attention):
        layer_list = [('decode0', ConvTransposeBlock(inplanes, outplanes, upsample_method=upsample_method))]
        extra_layers = 0 if extra_layers is None else extra_layers
        for i in range(extra_layers):
            layer_list.append(('decode'+str(i+1), ConvTransposeBlock(outplanes, outplanes)))
        layer_list.append(('decode'+str(extra_layers+1),ConvTransposeBlock(outplanes, outplanes, attention_kernel=attention)))
        return nn.Sequential(OrderedDict(layer_list))

    @staticmethod
    def build_reproj_block(inplanes, activation, layers=3, reducing=True, reduce_factor=0.9):
        remap_layers = [('reproj_maxpool', nn.MaxPool2d(kernel_size=2))]
        for i in range(layers-1):
            layer_inplanes = round(inplanes * pow(reduce_factor, i)) if reducing else inplanes
            layer_outplanes = round(inplanes * pow(reduce_factor, i+1)) if reducing else inplanes
            remap_layers.append(('reproj_deconv'+str(i+1), ConvTransposeBlock(layer_inplanes, layer_outplanes, activation=activation)))
        if layers > 0:
            layer_inplanes = round(inplanes * pow(reduce_factor, layers-1)) if reducing else inplanes
            remap_layers.append(('reproj_deconv'+str(layers), nn.ConvTranspose2d(layer_inplanes, 3, kernel_size=3, stride=1, padding=1, bias=False)))
        remap_layers.append(('reproj_tanh', nn.Tanh()))
        return nn.Sequential(OrderedDict(remap_layers))

    def forward(self, x, p):
        x1 = self.encoder_block0(x)
        x2 = self.encoder_block1(x1)
        x3 = self.encoder_block2(x2)
        x4 = self.encoder_block3(x3)
        x5 = self.encoder_block4(x4)
        x6 = self.encoder_block5(x5)
        x7 = self.encoder_block6(x6)

        p0 = self.position_block0(p)
        h1 = torch.cat((p0, x7), dim=1)
        y1 = self.decoder_block1(h1)
        w1 = self.reproj_block1(h1)

        p1 = self.position_block1(p0)
        h2 = torch.cat((p1, y1, x6), dim=1)
        y2 = self.decoder_block2(h2)
        i2 = torch.cat((p1, w1, x6), dim=1)
        w2 = self.reproj_block2(i2)

        p2 = self.position_block2(p1)
        h3 = torch.cat((p2, y2, x5), dim=1)
        y3 = self.decoder_block3(h3)
        i3 = torch.cat((p2, w2, x5), dim=1)
        w3 = self.reproj_block3(i3)

        p3 = self.position_block3(p2)
        h4 = torch.cat((p3, y3, x4), dim=1)
        y4 = self.decoder_block4(h4)
        i4 = torch.cat((p3, w3, x4), dim=1)
        w4 = self.reproj_block4(i4)

        p4 = self.position_block4(p3)
        h5 = torch.cat((p4, y4, x3), dim=1)
        y5 = self.decoder_block5(h5)
        i5 = torch.cat((p4, w4, x3), dim=1)
        w5 = self.reproj_block5(i5)

        p5 = self.position_block5(p4)
        h6 = torch.cat((p5, y5, x2), dim=1)
        y6 = self.decoder_block6(h6)
        y6u = self.super_res_upsample(y6)
        i6 = torch.cat((p5, w5, x2), dim=1)
        w6 = self.reproj_block6(i6)
        w6u = self.super_res_upsample(w6)

        r1 = self.reproj_left_output(w6u)
        r2 = self.reproj_right_output(w6u)
        r3 = self.reproj_up_output(w6u)
        r4 = self.reproj_down_output(w6u)
        r5 = self.reproj_straight_output(w6u)

        rx = torch.cat((r1, r2, r3, r4, r5), dim=1)

        y7 = self.decoder_block7(y6u)
        out = self.decoder_output(y7)

        return out, rx, (x1, x2, x3, x4, x5, x6, y1, y2, y3, y4, y5, y6, y7)


class ImageSaver:
    def __init__(self, mean, std, nrows=4, super_res_factor=1.0):
        self.mean = mean
        self.std = std
        self.nrows = nrows
        self.super_res_factor = super_res_factor

    def save(self, output, actual, input, remap_output, remap_actual, path):
        for t, m, s in zip(output.transpose_(dim0=0, dim1=1), self.mean, self.std):
            t.mul_(s).add_(m)
        output.transpose_(dim0=0, dim1=1)
        for t, m, s in zip(actual.transpose_(dim0=0, dim1=1), self.mean, self.std):
            t.mul_(s).add_(m)
        actual.transpose_(dim0=0, dim1=1)
        for t, m, s in zip(input.transpose_(dim0=0, dim1=1), self.mean[0:3] + self.mean[0:3], self.std[0:3] + self.std[0:3]):
            t.mul_(s).add_(m)
        input.transpose_(dim0=0, dim1=1)

        remap_output_list = [remap_output[:, i:i + 3, :, :] for i in range(0, 15, 3)]
        remap_actual_list = [remap_actual[:, i:i + 3, :, :] for i in range(0, 15, 3)]
        for r in remap_output_list + remap_actual_list:
            for t, m, s in zip(r.transpose_(dim0=0, dim1=1), self.mean[0:3] + self.mean[0:3], self.std[0:3] + self.std[0:3]):
                t.mul_(s).add_(m)
            r.transpose_(dim0=0, dim1=1)

        novel_images = torch.cat((output[:, 0:3, :, :], actual[:, 0:3, :, :]), dim=3)
        novel_depths = torch.cat((output[:, 3, :, :], actual[:, 3, :, :]), dim=2).unsqueeze(dim=1)
        novel_depths_rgb = torch.cat((novel_depths, novel_depths, novel_depths), dim=1)

        remap_output_images = torch.cat((remap_output_list[0][:, 0:3, :, :], remap_output_list[1][:, 0:3, :, :],
                                         remap_output_list[2][:, 0:3, :, :], remap_output_list[3][:, 0:3, :, :]), dim=3)
        remap_actual_images = torch.cat((remap_actual_list[0][:, 0:3, :, :], remap_actual_list[1][:, 0:3, :, :],
                                         remap_actual_list[2][:, 0:3, :, :], remap_actual_list[3][:, 0:3, :, :]), dim=3)
        remap_images = torch.cat((remap_output_images, remap_actual_images), dim=2)

        eye_images = torch.cat((input[:, 0:3, :, :], input[:, 3:6, :, :]), dim=3)
        eye_images = F.interpolate(eye_images, scale_factor=self.super_res_factor, mode='bilinear', align_corners=False)
        images = torch.cat((novel_images, novel_depths_rgb, remap_images, eye_images), dim=2).clamp(0, 1)
        save_image(images, path, nrow=self.nrows)


def train(args, model, device, train_loader, criterion, optimizer, epoch, img_saver):
    model.train()
    train_loss = [1.0]
    for batch_idx, data in enumerate(train_loader):
        data_input = torch.cat((data['left'], data['right']), dim=1).to(device)
        data_actual = torch.cat((data['generated'], data['generated_depth']), dim=1).to(device)
        remap_actual = torch.cat((data['reproj_left'], data['reproj_right'], data['reproj_up'], data['reproj_down'],
                                  data['reproj_forward']), dim=1).to(device)
        position = data['position'].to(device)
        optimizer.zero_grad()
        data_output, remap_output, _ = model(data_input, position)
        loss = criterion(data_output, data_actual, remap_output, remap_actual)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        optimizer.step()
        train_loss.append(loss.item())
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.5f} ({:.6f})'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), mean(train_loss), stdev(train_loss)))
            train_loss = []
            if batch_idx % (args.log_interval * 10) == 0:
                img_saver.save(data_output.cpu().data, data_actual.cpu().data, data_input.cpu().data,
                               remap_output.cpu().data, remap_actual.cpu().data,
                               join(args.model_path, 'image_{}_{}.png'.format(epoch, batch_idx)))

    #if epoch % 1 == 0:
    #    img_saver.save(data_output.cpu().data, data_actual.cpu().data, data_input.cpu().data,
    #                   remap_output.cpu().data, remap_actual.cpu().data,
    #                   join(args.model_path, 'image_{}.png'.format(epoch)))


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = []
    with torch.no_grad():
        for data in test_loader:
            data_input = torch.cat((data['left'], data['right']), dim=1).to(device)
            data_actual = torch.cat((data['generated'], data['generated_depth']), dim=1).to(device)
            remap_actual = torch.cat((data['reproj_left'], data['reproj_right'], data['reproj_up'], data['reproj_down'], data['reproj_forward']), dim=1).to(device)
            position = data['position'].to(device)
            data_output, remap_output, _ = model(data_input, position)
            test_loss.append(criterion(data_output, data_actual, remap_output, remap_actual).item())

    logging.info('Test set: Average loss: {:.6f} ({:.6f})\n'.format(mean(test_loss), stdev(test_loss)))


def main(custom_args=None):
    # Training settings
    model_number = "07"
    parser = argparse.ArgumentParser(description='PyTorch Model ' + model_number + ' Experiment')
    parser.add_argument('--batch-size', type=int, default=36, metavar='N',
                        help='input batch size for training (default: 36)')
    parser.add_argument('--test-batch-size', type=int, default=72, metavar='N',
                        help='input batch size for testing (default: 72)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--epoch-start', type=int, default=1, metavar='N',
                        help='epoch number to start counting from')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam beta 1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta 2 (default: 0.999)')
    parser.add_argument('--weight-decay', type=float, default=6e-7, metavar='D',
                        help='Weight decay (default: 6e-7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, default="model"+model_number, metavar='PATH',
                        help='pathname for this models output (default model'+model_number+')')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    parser.add_argument('--dataset-resample', type=float, default=1.0, metavar='S',
                        help='resample/resize the dataset images (default 1.0)')
    parser.add_argument('--dataset-train', type=str, metavar='PATH',
                        help='training dataset path')
    parser.add_argument('--dataset-test', type=str, metavar='PATH',
                        help='testing/validation dataset path')
    parser.add_argument('--super-res-factor', type=float, default=1.0, metavar='S',
                        help='super resolution factor, > 1.0 for upscaling (default 1.0)')
    parser.add_argument('--loss-ve-weight', type=float, default=0.75, metavar='W',
                        help='model loss value/edge weight (default 0.75/0.25); edge weight is 1 - val_weight.')
    parser.add_argument('--loss-remap-weight', type=float, default=0.25, metavar='R',
                        help='model loss remap weight (default 0.25/0.75);')
    parser.add_argument('--clip-max-norm', type=float, default=1.1, metavar='C',
                        help='gradient clip max_norm (default 1.1)')
    parser.add_argument('--attention', type=str, nargs='+',
                        help='attention parameters: layer/block code, resample kernel.')
    parser.add_argument('--extra-layers', type=str, nargs='+',
                        help='extra layers parameters: layer/block code, layer count (0,1,2,3).')
    parser.add_argument('--leaky_relu', type=float, metavar='S',
                        help='use leaky relu activation with given negative slope')
    args = parser.parse_args(args=custom_args)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(join(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    logging.info("Using random seed " + str(args.seed))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    dataset_mean, dataset_std = [0.334, 0.325, 0.320, 0.683], [0.666, 0.675, 0.680, 0.683]

    test_loader, train_loader, input_resolution, output_resolution = get_data_loaders(args, kwargs, mean=dataset_mean,
                                                                                      std=dataset_std)

    model = Net(downsample_method='conv', upsample_method='conv',
                input_resolution=input_resolution, output_resolution=output_resolution,
                extra_layers=parse_extra_layers(args), attention=parse_attention(args), activation=get_activation(args))
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device)

    logging.info("Using Adam optimizer with LR = {}, Beta = ({}, {}), Decay {}".format(args.lr, args.beta1, args.beta2,
                                                                                       args.weight_decay))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    logging.info("Model loss using value/edge weight {}/{}, remap weight {}/{}".format(
        args.loss_ve_weight, (1.0-args.loss_ve_weight), args.loss_remap_weight, (1.0-args.loss_remap_weight)))
    criterion = ModelLoss(device=device, value_edge_weight=args.loss_ve_weight, remap_weight=args.loss_remap_weight)

    img_saver = ImageSaver(mean=dataset_mean, std=dataset_std, nrows=6, super_res_factor=args.super_res_factor)

    for epoch in range(args.epoch_start, args.epochs + args.epoch_start):
        train(args, model, device, train_loader, criterion, optimizer, epoch, img_saver)
        test(args, model, device, test_loader, criterion)
        torch.save(model.state_dict(), join(args.model_path, "model_state_{}.pth".format(epoch)))


def get_data_loaders(args, kwargs, mean=None, std=None):
    if std is None:
        std = [0.5, 0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5, 0.5]
    train_res = int(args.dataset_train.split('_')[-1])
    test_res = int(args.dataset_test.split('_')[-1])
    test_scale = train_res / test_res
    input_resolution = (int(train_res * args.dataset_resample / args.super_res_factor),
                        int(train_res * args.dataset_resample / args.super_res_factor))
    output_resolution = (int(train_res * args.dataset_resample),
                         int(train_res * args.dataset_resample))
    reproj_angle = 30.0
    reproj_scale = 0.5

    logging.info("Building dataset with resample rate {}, super resolution factor {}".format(args.dataset_resample,
                                                                                             args.super_res_factor))
    train_set = dl.RandomSceneDataset(join('..', args.dataset_train), depth=True, reverse=True,
                                      transform=transforms.Compose(
        [dl.ResampleInputImages(args.dataset_resample / args.super_res_factor),
         dl.ResampleGeneratedImages(args.dataset_resample),
         dl.ResampleGeneratedDepth(args.dataset_resample),
         dl.GenerateReprojectedImages(name="forward", scale=reproj_scale, xrot=0.0, yrot=0.0),
         dl.GenerateReprojectedImages(name="left", scale=reproj_scale, xrot=reproj_angle, yrot=0.0),
         dl.GenerateReprojectedImages(name="right", scale=reproj_scale, xrot=-reproj_angle, yrot=0.0),
         dl.GenerateReprojectedImages(name="up", scale=reproj_scale, xrot=0.0, yrot=reproj_angle),
         dl.GenerateReprojectedImages(name="down", scale=reproj_scale, xrot=0.0, yrot=-reproj_angle),
         dl.ToTensor(),
         dl.NormalizeImages(mean=mean[0:3], std=std[0:3]),
         dl.NormalizeDepth(mean=mean[3:4], std=std[3:4])
         ]))
    test_set = dl.RandomSceneDataset(join('..', args.dataset_test), depth=True, reverse=False,
                                     transform=transforms.Compose(
        [dl.ResampleInputImages(args.dataset_resample * test_scale / args.super_res_factor),
         dl.ResampleGeneratedImages(args.dataset_resample * test_scale),
         dl.ResampleGeneratedDepth(args.dataset_resample * test_scale),
         dl.GenerateReprojectedImages(name="forward", scale=reproj_scale, xrot=0.0, yrot=0.0),
         dl.GenerateReprojectedImages(name="left", scale=reproj_scale, xrot=reproj_angle, yrot=0.0),
         dl.GenerateReprojectedImages(name="right", scale=reproj_scale, xrot=-reproj_angle, yrot=0.0),
         dl.GenerateReprojectedImages(name="up", scale=reproj_scale, xrot=0.0, yrot=reproj_angle),
         dl.GenerateReprojectedImages(name="down", scale=reproj_scale, xrot=0.0, yrot=-reproj_angle),
         dl.ToTensor(),
         dl.NormalizeImages(mean=mean[0:3], std=std[0:3]),
         dl.NormalizeDepth(mean=mean[3:4], std=std[3:4])
         ]))

    logging.info("Configuring dataset loader with train batch size {}, test batch size {}".format(args.batch_size,
                                                                                                  args.test_batch_size))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)
    return test_loader, train_loader, input_resolution, output_resolution


def parse_attention(args):
    attention = {}
    valid_attention = ["e"+str(i) for i in range(1,6)] + ["d"+str(i) for i in range(1,8)] + \
                      ["p"+str(i) for i in range(1,6)] + ["r"+str(i) for i in range(1,7)]
    if args.attention is not None:
        if len(args.attention) % 2 == 0:
            for i in range(0, len(args.attention), 2):
                code = args.attention[i]
                if code not in valid_attention:
                    logging.warning("Unknown attention layer/block code '" + code + "'")
                try:
                    kernel = int(args.attention[i + 1])
                    attention[code] = kernel
                except ValueError:
                    logging.error("Error parsing attenuation kernel '" + args.attention[i + 1] + "'")
        else:
            logging.error("Odd number of attention parameters. Ignoring attention configuration.")
    else:
        # attention = {"d1": 1}
        logging.debug("Attenation is none")
    if len(attention) > 0:
        logging.info("Model self attention configuration: {}".format(attention))
    return attention


def parse_extra_layers(args):
    extra_layers = {}
    valid_layers = ["e"+str(i) for i in range(1,7)] + ["d"+str(i) for i in range(1,8)] + ["p"+str(i) for i in range(1,6)]
    if args.extra_layers is not None:
        if len(args.extra_layers) % 2 == 0:
            for i in range(0, len(args.extra_layers), 2):
                code = args.extra_layers[i]
                if code not in valid_layers:
                    logging.warning("Unknown extra_layers layer/block code '" + code + "'")
                try:
                    count = int(args.extra_layers[i + 1])
                    extra_layers[code] = count
                except ValueError:
                    logging.error("Error parsing extra_layers count '" + args.extra_layers[i + 1] + "'")
        else:
            logging.error("Odd number of extra_layers parameters. Ignoring extra_layers configuration.")
    #else:
    #    for code in valid_layers:
    #        extra_layers[code] = 0
        #extra_layers = {"e1": 0, "e2": 0, "e3": 0, "e4": 0, "e5": 0, "e6": 0,
        #                "d1": 0, "d2": 0, "d3": 0, "d4": 0, "d5": 0, "d6": 0, "d7": 0}
    if len(extra_layers) > 0:
        logging.info("Model extra layers configuration: {}".format(extra_layers))
    return extra_layers


def get_activation(args):
    if args.leaky_relu is not None:
        logging.info("Using LeakyReLU activation with neg slope {}".format(args.leaky_relu))
        return nn.LeakyReLU(negative_slope=args.leaky_relu, inplace=True)
    logging.info("Using ReLU activation")
    return nn.ReLU(inplace=True)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(threadName)s] %(message)s')
    main()
