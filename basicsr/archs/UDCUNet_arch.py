import functools

import torch
import torch.nn as nn
import torch.nn.functional as F
import basicsr.archs.arch_util as arch_util
from basicsr.utils.registry import ARCH_REGISTRY
import numpy


@ARCH_REGISTRY.register()
class UDCUNet(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=32, depths=[2, 2, 2, 8, 2, 2, 2], DyK_size=3):
        super(UDCUNet, self).__init__()
        self.DyK_size = DyK_size

        ### Condition
        basic_Res = functools.partial(arch_util.ResidualBlockNoBN, nf=nf)

        self.cond_head = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.cond_first = arch_util.make_layer_unet(basic_Res, 2)

        self.CondNet0 = nn.Sequential(nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1))

        self.CondNet1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf * 2, 1))

        self.CondNet2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf * 4, 1))

        self.CondNet3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(nf, nf * 8, 1))

        ## Kernel

        self.k_head = nn.Sequential(nn.Conv2d(in_nc + 5, nf, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.k_first = arch_util.make_layer_unet(basic_Res, 2)

        self.KNet0 = nn.Sequential(nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf * self.DyK_size * self.DyK_size, 1))

        self.KNet1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf * 2 * self.DyK_size * self.DyK_size, 1))

        self.KNet2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf * 4 * self.DyK_size * self.DyK_size, 1))

        self.KNet3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
                                   nn.Conv2d(nf, nf * 8 * self.DyK_size * self.DyK_size, 1))

        ## Base
        self.conv_first = nn.Sequential(nn.Conv2d(in_nc, nf, 3, 1, 1), nn.LeakyReLU(0.2, True))
        basic_block = functools.partial(arch_util.ResBlock_with_SFT, nf=nf, in_nc=nf, out_nc=nf)
        basic_block2 = functools.partial(arch_util.ResBlock_with_SFT, nf=nf * 2, in_nc=nf * 2, out_nc=nf * 2)
        basic_block4 = functools.partial(arch_util.ResBlock_with_SFT, nf=nf * 4, in_nc=nf * 4, out_nc=nf * 4)
        basic_block8 = functools.partial(arch_util.ResBlock_with_SFT, nf=nf * 8, in_nc=nf * 8, out_nc=nf * 8)

        self.enconv_layer0 = arch_util.make_layer(basic_block, depths[0])
        self.down_conv0 = nn.Conv2d(nf, nf * 2, 3, 2, 1)

        self.enconv_layer1 = arch_util.make_layer(basic_block2, depths[1])
        self.down_conv1 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)

        self.enconv_layer2 = arch_util.make_layer(basic_block4, depths[2])
        self.down_conv2 = nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)

        self.Bottom_conv = arch_util.make_layer(basic_block8, depths[3])

        self.up_conv2 = nn.Sequential(nn.Conv2d(nf * 8, nf * 4 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.deconv_layer2 = arch_util.make_layer(basic_block4, depths[4])

        self.up_conv1 = nn.Sequential(nn.Conv2d(nf * 4, nf * 2 * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.deconv_layer1 = arch_util.make_layer(basic_block2, depths[5])

        self.up_conv0 = nn.Sequential(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
        self.deconv_layer0 = arch_util.make_layer(basic_block, depths[6])

        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        psf = numpy.float32(numpy.array([-0.0020, 0.0352, -2.0215e-05, 0.0060, 9.4328e-05]))
        psf = torch.from_numpy(psf).view(1, 5, 1, 1)

        psf = psf.expand(x.shape[0], -1, x.shape[2], x.shape[3]).cuda()
        k_fea = torch.cat((x, psf), 1)
        # print(psf.dtype)
        # print(x.dtype)
        # print(k_fea.dtype)
        k_fea = self.k_first(self.k_head(k_fea))
        kernel0 = self.KNet0(k_fea)
        kernel1 = self.KNet1(k_fea)
        kernel2 = self.KNet2(k_fea)
        kernel3 = self.KNet3(k_fea)

        cond = self.cond_first(self.cond_head(x))
        cond0 = self.CondNet0(cond)
        cond1 = self.CondNet1(cond)
        cond2 = self.CondNet2(cond)
        cond3 = self.CondNet3(cond)

        fea0 = self.conv_first(x)

        fea0, _ = self.enconv_layer0((fea0, cond0))
        down0 = self.down_conv0(fea0)

        fea1, _ = self.enconv_layer1((down0, cond1))
        down1 = self.down_conv1(fea1)

        fea2, _ = self.enconv_layer2((down1, cond2))
        down2 = self.down_conv2(fea2)
        # print(down2.shape)
        # print(kernel3.shape)
        feaB, _ = self.Bottom_conv((down2, cond3))
        feaB = feaB + kernel2d_conv(down2, kernel3, self.DyK_size)

        up2 = self.up_conv2(feaB) + kernel2d_conv(fea2, kernel2, self.DyK_size)
        defea2, _ = self.deconv_layer2((up2, cond2))

        up1 = self.up_conv1(defea2) + kernel2d_conv(fea1, kernel1, self.DyK_size)
        defea1, _ = self.deconv_layer1((up1, cond1))

        up0 = self.up_conv0(defea1) + kernel2d_conv(fea0, kernel0, self.DyK_size)
        defea0, _ = self.deconv_layer0((up0, cond0))

        out = F.relu(x + self.conv_last(defea0))

        return out


def kernel2d_conv(feat_in, kernel, ksize):
    """
    If you have some problems in installing the CUDA FAC layer,
    you can consider replacing it with this Python implementation.
    Thanks @AIWalker-Happy for his implementation.
    """
    channels = feat_in.size(1)
    N, kernels, H, W = kernel.size()
    pad_sz = (ksize - 1) // 2

    feat_in = F.pad(feat_in, (pad_sz, pad_sz, pad_sz, pad_sz), mode="replicate")
    feat_in = feat_in.unfold(2, ksize, 1).unfold(3, ksize, 1)
    feat_in = feat_in.permute(0, 2, 3, 1, 5, 4).contiguous()
    feat_in = feat_in.reshape(N, H, W, channels, -1)

    kernel = kernel.permute(0, 2, 3, 1).reshape(N, H, W, channels, ksize, ksize)
    kernel = kernel.permute(0, 1, 2, 3, 5, 4).reshape(N, H, W, channels, -1)
    feat_out = torch.sum(feat_in * kernel, axis=-1)
    feat_out = feat_out.permute(0, 3, 1, 2).contiguous()
    return feat_out