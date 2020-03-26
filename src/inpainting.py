import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb
from scipy.misc import imread

import numpy as np
from PIL import Image


class InpaintNet(nn.Module):
    def __init__(self, residual_blocks=8):
        super(InpaintNet, self).__init__()

        self.weight = torch.load("./src/weight/InpaintingModel_gen.pth", map_location=lambda storage, loc: storage)

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(residual_blocks):
            block = ResnetBlock(256, 2)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
        )

        # if init_weights:
        #     self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = (torch.tanh(x) + 1) / 2

        return x


class EdgeNet(nn.Module):
    def __init__(self, use_spectral_norm=True):
        super(EdgeNet, self).__init__()

        self.weight = torch.load("./src/weight/EdgeModel_gen.pth", map_location=lambda storage, loc: storage)

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            spectral_norm(nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=0), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(256, track_running_stats=False),
            nn.ReLU(True)
        )

        blocks = []
        for _ in range(8):
            block = ResnetBlock(256, 2, use_spectral_norm=use_spectral_norm)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.decoder = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(128, track_running_stats=False),
            nn.ReLU(True),

            spectral_norm(nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1), use_spectral_norm),
            nn.InstanceNorm2d(64, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=7, padding=0),
        )

        # if init_weights:
        #     self.init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(dilation),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=dilation, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            spectral_norm(nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=0, dilation=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)

        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html

        return out


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module

class Preprocessing():
    def __init__(self, image_masked, mask):
        # image_masked = imread("./test/before/test.png") / 255
        # mask = imread("./test/mask/test.png") / 255
        self.image_masked = image_masked
        self.mask = mask
        self.image_masked_gray = rgb2gray(image_masked)
        self.mask_gray = rgb2gray(mask)

        self.image_masked = (self.image_masked * (1 - self.mask)) + self.mask
        # image_masked_gray[mask.astype(np.bool)] = 1.0
        self.image_masked_gray = (self.image_masked_gray * (1 - self.mask_gray)) + self.mask_gray


        pilImg = Image.fromarray(np.uint8(image_masked * 255))#デバッグ-保存用に255倍
        # pilImg.save('yeaaaaaaa.png')

        self.edge_masked = canny(self.image_masked_gray, sigma=2, mask=(1 - self.mask_gray).astype(np.bool)).astype(np.float)
        # pilImg = Image.fromarray(np.uint8(image_masked_gray*255))#デバッグ-保存用に255倍
        # pilImg.save('input.png')

        self.image_masked_gray = self.image_masked_gray.reshape(1,1,512,512)
        self.edge_masked = self.edge_masked.reshape(1,1,512,512)
        self.mask_gray = self.mask_gray.reshape(1,1,512,512)

        self.input_edge = torch.cat((torch.from_numpy(self.image_masked_gray), torch.from_numpy(self.edge_masked), torch.from_numpy(self.mask_gray)), dim=1)
        # print(input_edge.shape)

        #inpait 画像処理
        self.image_masked = self.image_masked.transpose(2,0,1)
        self.image_masked = self.image_masked[np.newaxis,:,:,:]
