import sys
import torch
import torch.nn as nn
import torch.nn.parallel
from config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Upsample
import time
from collections import deque


class GLU(nn.Module):

    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)

        assert nc % 2 == 0, 'channels dont divide 2!'

        nc = int(nc / 2)


        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


def child_to_parent(child_c_code, classes_child, classes_parent):
    ratio = classes_child / classes_parent
    arg_parent = torch.argmax(child_c_code, dim=1) / ratio
    parent_c_code = torch.zeros([child_c_code.size(0), classes_parent]).cuda()
    for i in range(child_c_code.size(0)):
        parent_c_code[i][int(arg_parent[i])] = 1
    return parent_c_code


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block


#
def sameBlock(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block



class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(conv3x3(channel_num, channel_num * 2),
                                   nn.BatchNorm2d(channel_num * 2),
                                   GLU(),
                                   conv3x3(channel_num, channel_num),
                                   nn.BatchNorm2d(channel_num))

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class GET_IMAGE(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 3), nn.Tanh())

    def forward(self, h_code):
        return self.img(h_code)


# whatever a vast channels -> 1 channels (which means the  gray space)
class GET_MASK(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.img = nn.Sequential(conv3x3(ngf, 1), nn.Sigmoid())

    def forward(self, h_code):
        return self.img(h_code)

class SUBNETS_3(nn.Module):
    def __init__(self, ngf, num_residual=3):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        self.num_residual = num_residual


        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.samesample = sameBlock(ngf * 2, ngf)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code):

        h_c_code = h_code

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code


class MASK_(nn.Module):
    def __init__(self, ngf, num_residual=0):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.samesample = sameBlock(ngf * 2, ngf)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code):

        h_c_code = h_code

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.samesample(out_code)
        return out_code

class G_NET_OF_ComGAN(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = cfg.GAN.GF_DIM

        self.parent_stage = FEATURE_GENERATOR(ngf * 8)
        self.first_stage = MASK_(ngf // 4)
        self.first_mask = GET_MASK(ngf // 4)

        self.colour_stage = SUBNETS_3(ngf // 4)
        self.colour_image = GET_IMAGE(ngf // 4)

        self.colour_stag_inverse = SUBNETS_3(ngf // 4)
        self.colour_image_inverse = GET_IMAGE(ngf // 4)


    def forward(self, z_code):

        p_temp = self.parent_stage(z_code)
        f_temp = self.first_stage(p_temp)
        fake_img1_mask = self.first_mask(f_temp)


        s_temp = self.colour_stage(p_temp)
        fake_img_color = self.colour_image(s_temp)

        s_temp_inverse = self.colour_stag_inverse(p_temp)
        fake_img_color_inverse = self.colour_image_inverse(s_temp_inverse)

        fake_final_img = fake_img1_mask * fake_img_color + (1. - fake_img1_mask) * fake_img_color_inverse

        return fake_img1_mask, fake_img_color, fake_img_color_inverse, fake_final_img


class G_NET_of_DSComGAN(nn.Module):
    def __init__(self):
        super().__init__()
        ngf = cfg.GAN.GF_DIM
        self.parent_stage = FEATURE_GENERATOR_GLOBAL(ngf * 8)
        # # # \bar X_z # #  #
        self.parent_image = GET_IMAGE(ngf // 4)

        # # # Mask subnetwoks # # # #
        self.second_stage = MASK_SUBNETS(ngf  )
        self.second_mask = GET_MASK( ngf //4 )

        # # # Foreground subnetwoks # # #
        self.colour_stage = FORE_SUBNETS(ngf)
        self.colour_image = GET_IMAGE(ngf // 4)

        # # # Background subnetwoks # # #
        self.colour_stag_inverse = BACK_SUBNETS(ngf )
        self.colour_image_inverse = GET_IMAGE(ngf // 4)

        # # # Share feature # # #
        self.up64 = upBlock_64(64, 32)
        self.up32 = upBlock_32(32, 16)


    def forward(self, z_code, c_code, c_code_inv):
        fake_imgs = []
        mk_imgs = []

        fake_imgs_inverse = []
        mk_imgs_inverse = []

        # features
        p_temp, feature = self.parent_stage(z_code)
        xz_img = self.parent_image(p_temp)

        fake_imgs.append(xz_img)
        fake_imgs_inverse.append(xz_img)

        c_temp = torch.cat([self.up64(feature[0]),p_temp],dim=1)
        c_temp = torch.cat([self.up32(feature[1]),c_temp],dim=1)


        # Mask_ subnets
        m_temp = self.second_stage(c_temp)
        fake_mask = self.second_mask(m_temp)

        fake_imgs.append(fake_mask)
        fake_imgs_inverse.append(fake_mask)

        # Foreground_ subnets
        s_temp = self.colour_stage(c_temp, c_code)
        fake_img3_fore = self.colour_image(s_temp)
        fake_img3 = fake_mask * fake_img3_fore

        fake_imgs.append(fake_img3)
        mk_imgs.append(fake_img3_fore)

        # Background_ subnets
        s_temp_inverse = self.colour_stag_inverse(c_temp, c_code_inv)
        fake_img3_background = self.colour_image(s_temp_inverse)
        fake_img3_inverse = (1 - fake_mask ) * fake_img3_background

        fake_imgs_inverse.append(fake_img3_inverse)
        mk_imgs_inverse.append(fake_img3_background)

        fake_final_img = fake_img3_inverse + fake_img3

        return fake_imgs, mk_imgs, fake_imgs_inverse, mk_imgs_inverse, fake_final_img

class FEATURE_GENERATOR_GLOBAL(nn.Module):
    def __init__(self, ngf):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        in_dim = cfg.GAN.Z_DIM + self.code_len

        self.fc = nn.Sequential(nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())

        # 512*4*4
        self.upsample1 = upBlock(ngf, ngf // 2)
        # 256*8*8
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # 128*16*16
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # 64*32*32
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # 16*64*64
        self.upsample5 = upBlock(ngf // 16, ngf // 32)
        # 16*128*128

    def forward(self, z_input):
        self.feature = []

        in_code =  z_input
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        self.feature.append(out_code)
        out_code = self.upsample4(out_code)
        self.feature.append(out_code)
        out_code = self.upsample5(out_code)
        return out_code, self.feature

class FEATURE_GENERATOR(nn.Module):
    def __init__(self, ngf):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        in_dim = cfg.GAN.Z_DIM + self.code_len

        self.fc = nn.Sequential(nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), nn.BatchNorm1d(ngf * 4 * 4 * 2), GLU())
        # 512*4*4
        self.upsample1 = upBlock(ngf, ngf // 2)
        # 256*8*8
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        # 128*16*16
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        # 64*32*32
        self.upsample4 = upBlock(ngf // 8, ngf // 16)
        # 16*64*64
        self.upsample5 = upBlock(ngf // 16, ngf // 32)

    def forward(self, z_input):
        in_code = z_input
        out_code = self.fc(in_code).view(-1, self.ngf, 4, 4)
        out_code = self.upsample1(out_code)
        out_code = self.upsample2(out_code)
        out_code = self.upsample3(out_code)
        out_code = self.upsample4(out_code)
        out_code = self.upsample5(out_code)
        return out_code


class MASK_SUBNETS(nn.Module):
    def __init__(self, ngf, num_residual=1):
        super().__init__()

        self.ngf = ngf
        self.code_len = 0
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.downsample_1 = sameBlock(ngf * 2, ngf // 4)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code):

        h_c_code = h_code

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.downsample_1(out_code)
        return out_code

class FORE_SUBNETS(nn.Module):
    def __init__(self, ngf, num_residual=2):
        super().__init__()

        self.ngf = ngf
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.downsample_1 = sameBlock(ngf * 2, ngf // 4)

    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code, code):
        h, w = h_code.size(2), h_code.size(3)
        code = code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        h_c_code = torch.cat((code, h_code), 1)

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.downsample_1(out_code)
        return out_code

class BACK_SUBNETS(nn.Module):
    def __init__(self, ngf, num_residual=2):
        super().__init__()

        self.ngf = ngf
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.num_residual = num_residual

        self.jointConv = sameBlock(self.code_len + self.ngf, ngf * 2)
        self.residual = self._make_layer()
        self.downsample_1 = sameBlock(ngf * 2, ngf // 4)
    def _make_layer(self):
        layers = []
        for _ in range(self.num_residual):
            layers.append(ResBlock(self.ngf * 2))
        return nn.Sequential(*layers)

    def forward(self, h_code, code):
        h, w = h_code.size(2), h_code.size(3)
        code = code.view(-1, self.code_len, 1, 1).repeat(1, 1, h, w)
        h_c_code = torch.cat((code, h_code), 1)

        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        out_code = self.downsample_1(out_code)
        return out_code


def upBlock_128(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=8, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block

def upBlock_64(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=4, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block

def upBlock_32(in_planes, out_planes):
    block = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                          conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block
#
def sameBlock(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes * 2),
                          nn.BatchNorm2d(out_planes * 2),
                          GLU())
    return block






# ############## D networks ################################################

class FORE_D(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32
                                         nn.BatchNorm2d(ndf * 2),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16
                                         nn.BatchNorm2d(ndf * 4),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  # 4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         )

        self.code_logits = nn.Sequential(
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  #  4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, self.code_len, kernel_size=4,
                                                   stride=4))
        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1),
                                       nn.Sigmoid())

    def forward(self, x):
        x = self.encode_img(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)

class BACK_D(nn.Module):
    def __init__(self, ndf=64):
        super().__init__()
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.encode_img = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32
                                         nn.BatchNorm2d(ndf * 2),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16
                                         nn.BatchNorm2d(ndf * 4),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  # 4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         )

        self.code_logits = nn.Sequential(
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  # 4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, self.code_len, kernel_size=4,
                                                   stride=4))

        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1),
                                       nn.Sigmoid())

    def forward(self, x):
        x = self.encode_img(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)

class MASK_D(nn.Module):
    def __init__(self, ndf=32):
        super().__init__()
        self.code_len = cfg.FINE_GRAINED_CATEGORIES
        self.encode_mask = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32
                                         nn.BatchNorm2d(ndf * 2),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16
                                         nn.BatchNorm2d(ndf * 4),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),

                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         )

        self.code_logits = nn.Sequential(
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, self.code_len, kernel_size=4,
                                                   stride=4))

        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1),
                                       nn.Sigmoid())
    def forward(self, x):
        x = self.encode_mask(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)

class IMAGE_D(nn.Module):
    def __init__(self, ndf=32):
        super().__init__()
        self.code_len = cfg.SUPER_CATEGORIES
        self.encode_mask = nn.Sequential(nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32
                                         nn.BatchNorm2d(ndf * 2),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16
                                         nn.BatchNorm2d(ndf * 4),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  # 4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         )

        self.code_logits = nn.Sequential(
                                         nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1, bias=False),  # 4
                                         nn.BatchNorm2d(ndf * 8),
                                         nn.LeakyReLU(0.2, inplace=True),
                                         nn.Conv2d(ndf * 8, self.code_len, kernel_size=4,
                                                   stride=4))  # 4

        self.rf_logits = nn.Sequential(nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1),
                                       nn.Sigmoid())  # 4

    def forward(self, x):
        x = self.encode_mask(x)
        return self.code_logits(x).view(-1, self.code_len), self.rf_logits(x)

def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),  # shape -> shape / 2
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# conv3x3 means that the shape keep same, not change
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(conv3x3(in_planes, out_planes),
                          nn.BatchNorm2d(out_planes),
                          nn.LeakyReLU(0.2, inplace=True))
    return block


################################## BI_DIS #######################################

class Gaussian(nn.Module):
    def __init__(self, std):
        super(Gaussian, self).__init__()
        self.std = std

    def forward(self, x):
        n = torch.randn_like(x) * self.std
        return x + n


class ConvT_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, bn=True, activation=None, noise=False, std=None):
        super(ConvT_Block, self).__init__()
        model = [nn.ConvTranspose2d(in_c, out_c, k, s, p)]

        if bn:
            model.append(nn.BatchNorm2d(out_c))

        if activation == 'relu':
            model.append(nn.ReLU())
        elif activation == 'elu':
            model.append(nn.ELU())
        elif activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        elif activation == 'tanh':
            model.append(nn.Tanh())
        elif activation == 'sigmoid':
            model.append(nn.Sigmoid())
        elif activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Conv_Block(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=0, bn=True, activation=None, noise=False, std=None):
        super(Conv_Block, self).__init__()
        model = [nn.Conv2d(in_c, out_c, k, s, p)]

        if bn:
            model.append(nn.BatchNorm2d(out_c))

        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Linear_Block(nn.Module):
    def __init__(self, in_c, out_c, bn=True, activation=None, noise=False, std=None):
        super(Linear_Block, self).__init__()
        model = [nn.Linear(in_c, out_c)]

        if bn:
            model.append(nn.BatchNorm1d(out_c))

        if activation == 'relu':
            model.append(nn.ReLU())
        if activation == 'elu':
            model.append(nn.ELU())
        if activation == 'leaky':
            model.append(nn.LeakyReLU(0.2))
        if activation == 'tanh':
            model.append(nn.Tanh())
        if activation == 'sigmoid':
            model.append(nn.Sigmoid())
        if activation == 'softmax':
            model.append(nn.Softmax(dim=1))

        if noise:
            model.append(Gaussian(std))

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Viewer(nn.Module):
    def __init__(self, shape):
        super(Viewer, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Bi_Dis_base(nn.Module):
    def __init__(self, code_len, ngf=16):
        super(Bi_Dis_base, self).__init__()

        self.image_layer = nn.Sequential(
            Conv_Block(3, ngf, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.3),  # for the channels
            Conv_Block(ngf, ngf * 2, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 2, ngf * 4, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 4, ngf * 8, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 8, ngf * 16, 4, 2, 1, bn=False, activation='leaky', noise=False, std=0.5),
            Conv_Block(ngf * 16, 512, 4, 1, 0, bn=False, activation='leaky', noise=False, std=0.5),  # 1
            Viewer([-1, 512]))

        self.code_layer = nn.Sequential(Linear_Block(code_len, 512, bn=False, activation='leaky', noise=True, std=0.5),
                                        Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3),
                                        Linear_Block(512, 512, bn=False, activation='leaky', noise=True, std=0.3))

        self.joint = nn.Sequential(Linear_Block(1024, 1024, bn=False, activation='leaky', noise=False, std=0.5),
                                   Linear_Block(1024, 1, bn=False, activation='None'),
                                   Viewer([-1]))

    def forward(self, img, code):
        t1 = self.image_layer(img)

        t2 = self.code_layer(code)
        return self.joint(torch.cat([t1, t2], dim=1))


class Bi_Dis(nn.Module):
    def __init__(self):
        super(Bi_Dis, self).__init__()

        self.BD_z = Bi_Dis_base(cfg.GAN.Z_DIM)  # NOISE
        self.BD_b = Bi_Dis_base(cfg.FINE_GRAINED_CATEGORIES)  # BACKGROUND
        self.BD_p = Bi_Dis_base(cfg.SUPER_CATEGORIES)  # PARENTS
        self.BD_c = Bi_Dis_base(cfg.FINE_GRAINED_CATEGORIES)  # CHILEND

    def forward(self, img, z_code, b_code, p_code, c_code):
        which_pair_z = self.BD_z(img, z_code)
        which_pair_b = self.BD_b(img, b_code)
        which_pair_p = self.BD_p(img, p_code)
        which_pair_c = self.BD_c(img, c_code)

        return which_pair_z, which_pair_b, which_pair_p, which_pair_c


####################################### Feature ####################################


def Up_unet(in_c, out_c):
    return nn.Sequential(nn.ConvTranspose2d(in_c, out_c * 2, 4, 2, 1), nn.BatchNorm2d(out_c * 2), GLU())


def BottleNeck(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c * 2, 4, 4), nn.BatchNorm2d(out_c * 2), GLU())


def Down_unet(in_c, out_c):
    return nn.Sequential(nn.Conv2d(in_c, out_c * 2, 4, 2, 1), nn.BatchNorm2d(out_c * 2), GLU())


class FeatureExtractor(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.first = nn.Sequential(sameBlock(in_c, 32), sameBlock(32, 32))

        self.down1 = Down_unet(32, 32)
        # 32*64*64
        self.down2 = Down_unet(32, 64)
        # 64*32*32
        self.down3 = Down_unet(64, 128)
        # 128*16*16
        self.down4 = Down_unet(128, 256)
        # 256*8*8
        self.down5 = Down_unet(256, 512)
        # 512*4*4
        self.down6 = Down_unet(512, 512)
        # 512*2*2

        self.up1 = Up_unet(512, 256)
        # 256*4*4
        self.up2 = Up_unet(256 + 512, 512)
        # 256*8*8
        self.up3 = Up_unet(512 + 256, 256)
        # 256*16*16
        self.up4 = Up_unet(256 + 128, 128)
        # 128*32*32
        self.up5 = Up_unet(128 + 64, 64)
        # 64*64*64
        self.up6 = Up_unet(64 + 32, out_c)
        # out_c*128*128

        self.last = nn.Sequential(ResBlock(out_c), ResBlock(out_c))

    def forward(self, x):
        x = self.first(x)

        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        x = self.up1(self.down6(x5))

        x = self.up2(torch.cat([x, x5], dim=1))
        x = self.up3(torch.cat([x, x4], dim=1))
        x = self.up4(torch.cat([x, x3], dim=1))
        x = self.up5(torch.cat([x, x2], dim=1))
        x = self.up6(torch.cat([x, x1], dim=1))

        return self.last(x)


class Dis_Dis(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.encode_img = nn.Sequential(nn.Conv2d(in_c, 32, 4, 2, 0, bias=False),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(32, 64, 4, 2, 0, bias=False),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(64, 128, 4, 1, 0, bias=False),
                                        nn.LeakyReLU(0.2, inplace=True))
        self.rf_logits = nn.Sequential(nn.Conv2d(128, 1, kernel_size=4, stride=1), nn.Sigmoid())

    def forward(self, x):
        x = F.interpolate(x, [126, 126], mode='bilinear', align_corners=True)
        x = self.encode_img(x)
        return self.rf_logits(x)


class Feature_Encoder(nn.Module):
    #def __init__(self, m_c, b_c, f_c):
    def __init__(self):
        super(Feature_Encoder, self).__init__()
        self.m_l = cfg.SUPER_CATEGORIES #m_c.shape[1]
        self.b_l =  cfg.FINE_GRAINED_CATEGORIES #b_c.shape[1]
        self.f_l =  cfg.FINE_GRAINED_CATEGORIES #f_c.shape[1]
        ngf = 64
        self.in_dm = self.m_l + self.b_l + self.f_l

        self.fc = nn.Sequential(
            nn.Linear(self.in_dm, ngf * 8, bias=False), nn.BatchNorm1d(ngf * 8), GLU(),
            nn.Linear(ngf * 4, ngf * 8, bias=False), nn.BatchNorm1d(ngf * 8), GLU(),
            nn.Linear(ngf * 4, ngf * 8, bias=False), nn.BatchNorm1d(ngf * 8), GLU(),
        )
        self.model_f = nn.Sequential( nn.Linear(ngf * 4, cfg.FINE_GRAINED_CATEGORIES, bias=False))
        self.model_m = nn.Sequential(nn.Linear(ngf * 4, cfg.SUPER_CATEGORIES, bias=False))
        self.model_b = nn.Sequential(nn.Linear(ngf * 4, cfg.FINE_GRAINED_CATEGORIES, bias=False))

    def forward(self, m_c, b_c, f_c):
        input = torch.cat([m_c, b_c, f_c], dim = 1)
        input = self.fc(input)
        self.f = self.model_f(input)
        self.m = self.model_m(input)
        self.b = self.model_b(input)

        return self.f, self.m, self.b