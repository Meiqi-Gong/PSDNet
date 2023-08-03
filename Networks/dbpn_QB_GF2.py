import os
import sys

import torch.nn as nn
import torch.optim as optim
from Networks.base_networks import *
from torchvision.transforms import *
import numpy as np

class Net_stage1(nn.Module):
    def __init__(self, num_channels=4, base_filter=16, n_feat=16, num_stages=3, scale_factor=4,
                 scale_feat=16, kernel_size=3, reduction=4, bias=False):
        super(Net_stage1, self).__init__()

        self.up = nn.Upsample(scale_factor=4, mode='bicubic')
        act = nn.PReLU()
        self.body1_1 = nn.Sequential(DeconvBlock(num_channels, n_feat, 4, 4, 0, activation='prelu', norm=None),
                                     act,
                                     conv(n_feat, n_feat, kernel_size, bias=bias))

        modules_body2 = []
        modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body1_2 = nn.Sequential(*modules_body2)
        self.shallow_feat1 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))
        self.conv1_1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv1_2 = conv(n_feat, 4, kernel_size, bias=bias)
        self.DBPNnet = DBPN_block(n_feat, base_filter, n_feat, num_stages, scale_factor)

    def forward(self, ms, pan):
        _, _, H, W = ms.shape
        ms2top_img = ms[:, :, 0:int(H / 2), :]
        ms2bot_img = ms[:, :, int(H / 2):H, :]
        pan2top_img = pan[:, :, 0:int(H * 4 / 2), :]
        pan2bot_img = pan[:, :, int(H * 4 / 2):H * 4, :]

        # Four Patches for Stage 1
        ms1ltop_img = ms2top_img[:, :, :, 0:int(W / 2)]
        ms1rtop_img = ms2top_img[:, :, :, int(W / 2):W]
        ms1lbot_img = ms2bot_img[:, :, :, 0:int(W / 2)]
        ms1rbot_img = ms2bot_img[:, :, :, int(W / 2):W]

        pan1ltop_img = pan2top_img[:, :, :, 0:int(W * 4 / 2)]
        pan1rtop_img = pan2top_img[:, :, :, int(W * 4 / 2):W * 4]
        pan1lbot_img = pan2bot_img[:, :, :, 0:int(W * 4 / 2)]
        pan1rbot_img = pan2bot_img[:, :, :, int(W * 4 / 2):W * 4]

        ##stage1
        bicubic_xtop = self.up(ms2top_img)
        bicubic_xbot = self.up(ms2bot_img)
        xbodyltop = self.body1_1(ms1ltop_img)
        xbodyrtop = self.body1_1(ms1rtop_img)
        xbodylbot = self.body1_1(ms1lbot_img)
        xbodyrbot = self.body1_1(ms1rbot_img)

        ybodyltop = self.body1_2(pan1ltop_img)
        ybodyrtop = self.body1_2(pan1rtop_img)
        ybodylbot = self.body1_2(pan1lbot_img)
        ybodyrbot = self.body1_2(pan1rbot_img)

        x1ltop = self.shallow_feat1(torch.cat((xbodyltop, ybodyltop), 1))
        x1rtop = self.shallow_feat1(torch.cat((xbodyrtop, ybodyrtop), 1))
        x1lbot = self.shallow_feat1(torch.cat((xbodylbot, ybodylbot), 1))
        x1rbot = self.shallow_feat1(torch.cat((xbodyrbot, ybodyrbot), 1))

        xtop, xbot, concat1_l, concat1_h, concat3_l, concat3_h = self.DBPNnet(x1ltop, x1rtop, x1lbot, x1rbot)
        xtop = self.conv1_1(xtop)  ##[8,16,264,164]
        xbot = self.conv1_1(xbot)
        xtop = self.conv1_2(xtop)
        xbot = self.conv1_2(xbot)
        img1top = xtop + bicubic_xtop
        img1bot = xbot + bicubic_xbot
        ltop = [concat1_l[0], concat3_l[0]]
        lbot = [concat1_l[1], concat3_l[1]]
        htop = [concat1_h[0], concat3_h[0]]
        hbot = [concat1_h[1], concat3_h[1]]
        img1 = torch.cat((img1top, img1bot), 2)
        l = [ltop, lbot]
        h = [htop, hbot]

        return img1, img1top, img1bot, l, h

class Net_stage2(nn.Module):
    def __init__(self, in_c=4, n_feat=40,
                 scale_feat=16, kernel_size=3, reduction=4, bias=False):
        super(Net_stage2, self).__init__()
        act = nn.PReLU()
        modules_body1 = []
        modules_body1.append(conv(in_c, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body2_1 = nn.Sequential(*modules_body1)

        modules_body2 = []
        modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body2_2 = nn.Sequential(*modules_body2)

        self.shallow_feat2 = nn.Sequential(conv(n_feat*2, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.conv2_1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2_2 = conv(n_feat, 4, kernel_size, bias=bias)
    def forward(self, pan, img1, img1top, img1bot, l, h):
        _, _, H, W = pan.shape
        pan2top_img = pan[:, :, 0:int(H / 2), :]
        pan2bot_img = pan[:, :, int(H / 2):H, :]
        xtop = self.body2_1(img1top)
        xbot = self.body2_1(img1bot)
        ytop = self.body2_2(pan2top_img)
        ybot = self.body2_2(pan2bot_img)

        ltop, lbot = l
        htop, hbot = h
        xfeattop = self.shallow_feat2(torch.cat((xtop, ytop), 1))
        xfeatbot = self.shallow_feat2(torch.cat((xbot, ybot), 1))
        feat2_top = self.stage1_encoder(xfeattop, ltop, htop)
        feat2_bot = self.stage1_encoder(xfeatbot, lbot, hbot)
        feat2 = [torch.cat((k, v), 2) for k, v in zip(feat2_top, feat2_bot)]

        resx = self.stage1_decoder(feat2)
        x2 = self.conv2_1(resx[0])
        x3 = self.conv2_2(x2)
        img2 = x3 + img1
        return img2, feat2, resx

class ORB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias):
        super(ORB, self).__init__()
        modules = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                     SAB(n_feat, kernel_size, bias=bias),
                    conv(n_feat, n_feat, kernel_size, bias=bias), act]
        self.body1 = nn.Sequential(*modules)
        self.body2 = nn.Sequential(*modules)
        self.body3 = nn.Sequential(*modules)
        # self.body4 = nn.Sequential(*modules)
        # self.body5 = nn.Sequential(*modules)

    def forward(self, x):
        res = self.body1(x)
        res = self.body2(res)
        res = self.body3(res)
        # res = self.body4(res)
        # res = self.body5(res)
        res = res + x
        return res

class DBPN_block(nn.Module):
    def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
        super(DBPN_block, self).__init__()
        self.DBPNnet1 = DBPN_block1(n_feat, base_filter, n_feat, num_stages, scale_factor)
        self.DBPNnet2 = DBPN_block2(n_feat, base_filter, n_feat, num_stages, scale_factor)
        self.DBPNnet3 = DBPN_block3(n_feat, base_filter, n_feat, num_stages, scale_factor)

    def forward(self, xltop, xrtop, xlbot, xrbot):
        xfltop, xfrtop, xflbot, xfrbot, concat1_l, concat1_h = self.DBPNnet1(xltop, xrtop, xlbot, xrbot)
        xftop, xfbot = self.DBPNnet2(xfltop, xfrtop, xflbot, xfrbot)
        xtop, xbot, concat3_l, concat3_h = self.DBPNnet3(xftop, xfbot)
        return xtop, xbot, concat1_l, concat1_h, concat3_l, concat3_h

class Net_stage3(nn.Module):
    def __init__(self, in_c=4, n_feat=40,
                 scale_feat=16, kernel_size=3, reduction=4, bias=False):
        super(Net_stage3, self).__init__()
        act = nn.PReLU()
        modules_body1 = []
        modules_body1.append(conv(in_c, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body3_1 = nn.Sequential(*modules_body1)

        modules_body2 = []
        modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        self.body3_2 = nn.Sequential(*modules_body2)

        self.shallow_feat3 = nn.Sequential(conv(n_feat*2, n_feat, kernel_size, bias=bias),
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))
        self.block1 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block2 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block3 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.trans_enc1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.trans_dec1 = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        self.conv3_1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv3_2 = conv(n_feat, 4, kernel_size, bias=bias)
    def forward(self, pan, img2, en, de):
        x1 = self.body3_1(img2)
        y1 = self.body3_2(pan)
        x1 = self.shallow_feat3(torch.cat((x1, y1), 1))
        xfeat1 = self.block1(x1) + self.trans_enc1(en[0]) + self.trans_dec1(de[0])
        xfeat2 = self.block2(xfeat1)  # + self.trans_enc1(xfeat[0]) + self.trans_dec1(resx[0])
        xfeat3 = self.block3(xfeat2)  # + self.trans_enc1(xfeat[0]) + self.trans_dec1(resx[0])
        x2 = self.conv3_1(xfeat3)
        x3 = self.conv3_2(x2)
        img3 = x3 + img2
        return img3

# class Net(nn.Module):
#     def __init__(self, num_channels=4, base_filter=28, n_feat=28, num_stages=3, scale_factor=4,
#                  scale_feat=28, kernel_size=3, reduction=4, bias=False):
#         super(Net, self).__init__()
#         act = nn.PReLU()
#         modules_body1 = []
#         modules_body1.append(conv(num_channels, n_feat, kernel_size, bias=bias))
#         modules_body1.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body1.append(act)
#         modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body1.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body1.append(act)
#         self.body1_1 = nn.Sequential(DeconvBlock(num_channels, n_feat, 4, 4, 0, activation='prelu', norm='batch'),
#                                      act,
#                                      conv(n_feat, n_feat, kernel_size, bias=bias),
#                                      torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                           affine=True, track_running_stats=True),
#                                      act)
#         self.body2_1 = nn.Sequential(*modules_body1)
#         self.body3_1 = nn.Sequential(*modules_body1)
#
#         modules_body2 = []
#         modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
#         modules_body2.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body2.append(act)
#         modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body2.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body2.append(act)
#         self.body1_2 = nn.Sequential(*modules_body2)
#         self.body2_2 = nn.Sequential(*modules_body2)
#         self.body3_2 = nn.Sequential(*modules_body2)
#
#         self.shallow_feat1 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias),
#                                            torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True), act,
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
#                                            SAB(n_feat, kernel_size, bias=bias))
#         self.shallow_feat2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias),
#                                            torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True), act,
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
#                                            SAB(n_feat, kernel_size, bias=bias))
#         self.shallow_feat3 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias),
#                                            torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True), act,
#                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
#                                            SAB(n_feat, kernel_size, bias=bias))
#
#         self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
#         self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
#         self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.conv1_2 = nn.Sequential(conv(n_feat, 4, kernel_size, bias=bias), nn.Tanh())
#         self.conv2_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.conv2_2 = nn.Sequential(conv(n_feat, 4, kernel_size, bias=bias), nn.Tanh())
#         self.conv3_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.conv3_2 = nn.Sequential(conv(n_feat, 4, kernel_size, bias=bias), nn.Tanh())
#
#         self.block1 = ORB(n_feat, kernel_size, reduction, act, bias)
#         self.block2 = ORB(n_feat, kernel_size, reduction, act, bias)
#         self.block3 = ORB(n_feat, kernel_size, reduction, act, bias)
#         self.trans_enc1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.trans_enc2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
#                                         conv(n_feat*2, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.trans_enc3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
#                                         conv(n_feat*3, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.trans_dec1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.trans_dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
#                                         conv(n_feat*2, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         self.trans_dec3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
#                                         conv(n_feat*3, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#
#         self.DBPNnet = DBPN_block(n_feat, base_filter, n_feat, num_stages, scale_factor)
#         self.up = nn.Upsample(scale_factor=4, mode='bicubic')
#
#     def forward(self, ms, pan):
#         # Two Patches for Stage 2
#         _,_,H,W = ms.shape
#         bicubic_x = self.up(ms)
#         # print(ms.shape)
#         # print(pan.shape)
#         ms2top_img = ms[:,:,0:int(H/2)+4,:]
#         ms2bot_img = ms[:,:,int(H/2)-4:H,:]
#         pan2top_img = pan[:, :, 0:int(H*4 / 2)+16, :]
#         pan2bot_img = pan[:, :, int(H*4 / 2)-16:H*4, :]
#
#         # Four Patches for Stage 1
#         ms1ltop_img = ms2top_img[:,:,:,0:int(W/2)+4]
#         ms1rtop_img = ms2top_img[:,:,:,int(W/2)-4:W]
#         ms1lbot_img = ms2bot_img[:,:,:,0:int(W/2)+4]
#         ms1rbot_img = ms2bot_img[:,:,:,int(W/2)-4:W]
#
#         pan1ltop_img = pan2top_img[:, :, :, 0:int(W*4 / 2)+16]
#         pan1rtop_img = pan2top_img[:, :, :, int(W*4 / 2)-16:W*4]
#         pan1lbot_img = pan2bot_img[:, :, :, 0:int(W*4 / 2)+16]
#         pan1rbot_img = pan2bot_img[:, :, :, int(W*4 / 2)-16:W*4]
#         # print(ms1ltop_img.shape)
#         # print(pan1ltop_img.shape)
#         _,_,H,W = ms1ltop_img.shape
#
#         ##stage1
#         bicubic_xtop = self.up(ms2top_img)
#         bicubic_xbot = self.up(ms2bot_img)
#         xbodyltop = self.body1_1(ms1ltop_img)
#         xbodyrtop = self.body1_1(ms1rtop_img)
#         xbodylbot = self.body1_1(ms1lbot_img)
#         xbodyrbot = self.body1_1(ms1rbot_img)
#
#         ybodyltop = self.body1_2(pan1ltop_img)
#         ybodyrtop = self.body1_2(pan1rtop_img)
#         ybodylbot = self.body1_2(pan1lbot_img)
#         ybodyrbot = self.body1_2(pan1rbot_img)
#
#         # print(xbodyltop.shape)
#         # print(ybodyltop.shape)
#         x1ltop = self.shallow_feat1(torch.cat((xbodyltop, ybodyltop), 1))#[148,148]
#         x1rtop = self.shallow_feat1(torch.cat((xbodyrtop, ybodyrtop), 1))
#         x1lbot = self.shallow_feat1(torch.cat((xbodylbot, ybodylbot), 1))
#         x1rbot = self.shallow_feat1(torch.cat((xbodyrbot, ybodyrbot), 1))
#
#         xtop, xbot, concat1_l, concat1_h, concat3_l, concat3_h = self.DBPNnet(x1ltop, x1rtop, x1lbot, x1rbot)
#         xtoptemp = self.conv1_1(xtop)##[8,16,264,264]
#         xbottemp = self.conv1_1(xbot)
#         xtop = self.conv1_2(xtoptemp)
#         xbot = self.conv1_2(xbottemp)
#         # print(bicubic_xtop.shape)
#         # print(bicubic_xbot.shape)
#         img1top = xtop + bicubic_xtop
#         img1bot = xbot + bicubic_xbot
#         ltop = [concat1_l[0], concat3_l[0]]
#         lbot = [concat1_l[1], concat3_l[1]]
#         htop = [concat1_h[0], concat3_h[0]]
#         hbot = [concat1_h[1], concat3_h[1]]
#         # img1 = torch.cat((img1top[:, :, :W*4-32, :],
#         #                    (img1top[:, :, W*4-32:, :] + img1bot[:, :, :32, :]) / 2,
#         #                    img1bot[:, :, 32:, :]), 2)
#         for i in range(32):
#             y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
#             if i==0:
#                 y=img1top[:, :, W * 4 - 32:W * 4 - 31, :]*y1 + img1bot[:, :, :1, :]*(1-y1)
#             else:
#                 y=torch.cat((y, img1top[:, :, W * 4 - 32+i
#                   :W * 4 - 31+i, :]*y1 + img1bot[:, :, i:i+1, :]*(1-y1)), 2)
#         img1 = torch.cat((img1top[:, :, :W * 4 - 32, :], y, img1bot[:, :, 32:, :]), 2)
#
#         ##stage2
#         xtop = self.body2_1(img1top)
#         xbot = self.body2_1(img1bot)
#         ytop = self.body2_2(pan2top_img)
#         ybot = self.body2_2(pan2bot_img)
#
#         xfeattop = self.shallow_feat2(torch.cat((xtop, ytop), 1))
#         xfeatbot = self.shallow_feat2(torch.cat((xbot, ybot), 1))
#         feat2_top = self.stage1_encoder(xfeattop, ltop, htop)
#         feat2_bot = self.stage1_encoder(xfeatbot, lbot, hbot)
#         for i in range(32):
#             y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
#             y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 16))
#             y3 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
#             if i==0:
#                 yy0=feat2_top[0][:, :, W * 4 - 32:W * 4 - 31, :]*y1 + feat2_bot[0][:, :, :1, :]*(1-y1)
#                 yy1 = feat2_top[1][:, :, W * 2 - 16:W * 2 - 15, :] * y2 + feat2_bot[1][:, :, :1, :] * (1 - y2)
#                 yy2 = feat2_top[2][:, :, W * 1 - 8:W * 1 - 7, :] * y3 + feat2_bot[2][:, :, :1, :] * (1 - y3)
#             elif i>=16:
#                 yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32+i
#                       :W * 4 - 31+i, :]*y1 + feat2_bot[0][:, :, i:i+1, :]*(1-y1)), 2)
#             elif i>=8:
#                 yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
#                       :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
#                 yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
#                       :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
#             else:
#                 yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
#                       :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
#                 yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
#                       :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
#                 yy2 = torch.cat((yy2, feat2_top[2][:, :, W * 1 - 8 + i
#                       :W * 1 - 7 + i, :] * y3 + feat2_bot[2][:, :, i:i + 1, :] * (1 - y3)), 2)
#
#         feat_h = torch.cat((feat2_top[0][:, :, :W * 4 - 32, :], yy0,
#                             feat2_bot[0][:, :, 32:, :]), 2)
#         feat_m = torch.cat((feat2_top[1][:, :, :W * 2 - 16, :], yy1,
#                             feat2_bot[1][:, :, 16:, :]), 2)
#         feat_l = torch.cat((feat2_top[2][:, :, :W - 8, :], yy2,
#                             feat2_bot[2][:, :, 8:, :]), 2)
#
#         # print(feat_h.shape)
#         # print(feat_m.shape)
#         # print(feat_l.shape)
#         # sys.exit(0)
#         # feat_h = torch.cat((feat2_top[0][:,:,:W*4-32,:], (feat2_top[0][:,:,W*4-32:,:]
#         #          +feat2_bot[0][:,:,:32,:])/2, feat2_bot[0][:,:,32:,:]), 2)
#         # feat_m = torch.cat((feat2_top[1][:,:,:W*2-16,:], (feat2_top[1][:,:,W*2-16:,:]
#         #          +feat2_bot[1][:,:,:16,:])/2, feat2_bot[1][:,:,16:,:]), 2)
#         # feat_l = torch.cat((feat2_top[2][:,:,:W-8,:], (feat2_top[2][:,:,W-8:,:]
#         #          +feat2_bot[2][:,:,:8,:])/2, feat2_bot[2][:,:,8:,:]), 2)
#
#         resx = self.stage1_decoder(feat_h, feat_m, feat_l)
#         outfeat1 = resx[0]
#         x2 = self.conv2_1(resx[0])
#         x3 = self.conv2_2(x2)
#         img2 = x3 + bicubic_x
#
#         #stage3
#         x1 = self.body3_1(img2)
#         y1 = self.body3_2(pan)
#         x1 = self.shallow_feat3(torch.cat((x1, y1), 1))
#         x1 = x1 + self.trans_enc1(feat_h) + self.trans_dec1(resx[0])# + \
#              # self.trans_enc2(feat2[1]) + self.trans_dec2(resx[1]) + \
#              # self.trans_enc3(feat2[2]) + self.trans_dec3(resx[2])
#         xfeat1 = self.block1(x1)
#         # xfeat1 = self.block1(x1 + self.trans_enc1(feat2[0]) + self.trans_dec1(resx[0]))
#         xfeat2 = self.block2(xfeat1)# + self.trans_enc1(xfeat[0]) + self.trans_dec1(resx[0])
#         xfeat3 = self.block3(xfeat2)# + self.trans_enc1(xfeat[0]) + self.trans_dec1(resx[0])
#         outfeat2 = xfeat3
#         x2 = self.conv3_1(xfeat3)
#         x3 = self.conv3_2(x2)
#         img3 = x3 + bicubic_x
#         # sys.exit(0)
#         return img1, img2, img3, outfeat1, outfeat2

class Net_GT(nn.Module):
    def __init__(self, num_channels=8, base_filter=28, n_feat=28, num_stages=3, scale_factor=4,
                 scale_feat=28, kernel_size=3, reduction=4, bias=False):
        super(Net_GT, self).__init__()
        act = nn.PReLU()
        modules_body1 = []
        modules_body1.append(conv(num_channels, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        self.body1_1 = nn.Sequential(DeconvBlock(num_channels, n_feat, 4, 4, 0, activation='prelu', norm=None),
                                     act,
                                     conv(n_feat, n_feat, kernel_size, bias=bias),
                                     act)
        self.body2_1 = nn.Sequential(*modules_body1)
        self.body3_1 = nn.Sequential(*modules_body1)

        modules_body2 = []
        modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        self.body1_2 = nn.Sequential(*modules_body2)
        self.body2_2 = nn.Sequential(*modules_body2)
        self.body3_2 = nn.Sequential(*modules_body2)

        self.shallow_feat1 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))
        self.shallow_feat2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))
        self.shallow_feat3 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                           CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv1_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv2_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv2_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv3_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv3_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())

        self.block1 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block2 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block3 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.trans_enc1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_enc2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                        conv(n_feat*2, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_enc3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
                                        conv(n_feat*3, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                        conv(n_feat*2, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
                                        conv(n_feat*3, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)

        self.DBPNnet = DBPN_block(n_feat, base_filter, n_feat, num_stages, scale_factor)
        self.up = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, target, ms, pan):
        # Two Patches for Stage 2
        _,_,H,W = ms.shape
        target2top_img = target[:,:,0:int(H*4 / 2)+16,:]
        target2bot_img = target[:,:,int(H*4 / 2)-16:H*4,:]
        bicubic_x = self.up(ms)
        ms2top_img = ms[:,:,0:int(H/2)+4,:]
        ms2bot_img = ms[:,:,int(H/2)-4:H,:]
        pan2top_img = pan[:, :, 0:int(H*4 / 2)+16, :]
        pan2bot_img = pan[:, :, int(H*4 / 2)-16:H*4, :]

        # Four Patches for Stage 1
        ms1ltop_img = ms2top_img[:,:,:,0:int(W/2)+4]
        ms1rtop_img = ms2top_img[:,:,:,int(W/2)-4:W]
        ms1lbot_img = ms2bot_img[:,:,:,0:int(W/2)+4]
        ms1rbot_img = ms2bot_img[:,:,:,int(W/2)-4:W]

        pan1ltop_img = pan2top_img[:, :, :, 0:int(W*4 / 2)+16]
        pan1rtop_img = pan2top_img[:, :, :, int(W*4 / 2)-16:W*4]
        pan1lbot_img = pan2bot_img[:, :, :, 0:int(W*4 / 2)+16]
        pan1rbot_img = pan2bot_img[:, :, :, int(W*4 / 2)-16:W*4]
        _,_,H,W = ms1ltop_img.shape
        # print(ms1ltop_img.shape)

        ##stage1
        bicubic_xtop = self.up(ms2top_img)
        bicubic_xbot = self.up(ms2bot_img)
        xbodyltop = self.body1_1(ms1ltop_img)
        xbodyrtop = self.body1_1(ms1rtop_img)
        xbodylbot = self.body1_1(ms1lbot_img)
        xbodyrbot = self.body1_1(ms1rbot_img)

        ybodyltop = self.body1_2(pan1ltop_img)
        ybodyrtop = self.body1_2(pan1rtop_img)
        ybodylbot = self.body1_2(pan1lbot_img)
        ybodyrbot = self.body1_2(pan1rbot_img)

        x1ltop = self.shallow_feat1(torch.cat((xbodyltop, ybodyltop), 1))#[148,148]
        x1rtop = self.shallow_feat1(torch.cat((xbodyrtop, ybodyrtop), 1))
        x1lbot = self.shallow_feat1(torch.cat((xbodylbot, ybodylbot), 1))
        x1rbot = self.shallow_feat1(torch.cat((xbodyrbot, ybodyrbot), 1))

        xtop, xbot, concat1_l, concat1_h, concat3_l, concat3_h = self.DBPNnet(x1ltop, x1rtop, x1lbot, x1rbot)
        xtoptemp = self.conv1_1(xtop)##[8,16,264,264]
        xbottemp = self.conv1_1(xbot)
        xtop = self.conv1_2(xtoptemp)
        xbot = self.conv1_2(xbottemp)
        img1top = xtop + bicubic_xtop
        img1bot = xbot + bicubic_xbot
        ltop = [concat1_l[0], concat3_l[0]]
        lbot = [concat1_l[1], concat3_l[1]]
        htop = [concat1_h[0], concat3_h[0]]
        hbot = [concat1_h[1], concat3_h[1]]
        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            if i==0:
                y=img1top[:, :, W * 4 - 32:W * 4 - 31, :]*y1 + img1bot[:, :, :1, :]*(1-y1)
            else:
                y=torch.cat((y, img1top[:, :, W * 4 - 32+i
                  :W * 4 - 31+i, :]*y1 + img1bot[:, :, i:i+1, :]*(1-y1)), 2)
        img1 = torch.cat((img1top[:, :, :W * 4 - 32, :], y, img1bot[:, :, 32:, :]), 2)

        ##stage2
        xtop = self.body2_1(target2top_img)
        xbot = self.body2_1(target2bot_img)
        ytop = self.body2_2(pan2top_img)
        ybot = self.body2_2(pan2bot_img)

        xfeattop = self.shallow_feat2(torch.cat((xtop, ytop), 1))
        xfeatbot = self.shallow_feat2(torch.cat((xbot, ybot), 1))
        feat2_top = self.stage1_encoder(xfeattop, ltop, htop)
        feat2_bot = self.stage1_encoder(xfeatbot, lbot, hbot)
        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 16))
            y3 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
            if i==0:
                yy0=feat2_top[0][:, :, W * 4 - 32:W * 4 - 31, :]*y1 + feat2_bot[0][:, :, :1, :]*(1-y1)
                yy1 = feat2_top[1][:, :, W * 2 - 16:W * 2 - 15, :] * y2 + feat2_bot[1][:, :, :1, :] * (1 - y2)
                yy2 = feat2_top[2][:, :, W * 1 - 8:W * 1 - 7, :] * y3 + feat2_bot[2][:, :, :1, :] * (1 - y3)
            elif i>=16:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32+i
                      :W * 4 - 31+i, :]*y1 + feat2_bot[0][:, :, i:i+1, :]*(1-y1)), 2)
            elif i>=8:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
                      :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
                yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
                      :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
            else:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
                      :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
                yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
                      :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
                yy2 = torch.cat((yy2, feat2_top[2][:, :, W * 1 - 8 + i
                      :W * 1 - 7 + i, :] * y3 + feat2_bot[2][:, :, i:i + 1, :] * (1 - y3)), 2)

        feat_h = torch.cat((feat2_top[0][:, :, :W * 4 - 32, :], yy0,
                            feat2_bot[0][:, :, 32:, :]), 2)
        feat_m = torch.cat((feat2_top[1][:, :, :W * 2 - 16, :], yy1,
                            feat2_bot[1][:, :, 16:, :]), 2)
        feat_l = torch.cat((feat2_top[2][:, :, :W - 8, :], yy2,
                            feat2_bot[2][:, :, 8:, :]), 2)

        resx = self.stage1_decoder(feat_h, feat_m, feat_l)
        outfeat1 = resx[0]
        x2 = self.conv2_1(resx[0])
        x3 = self.conv2_2(x2)
        img2 = x3 + bicubic_x

        #stage3
        x1 = self.body3_1(img2)
        y1 = self.body3_2(pan)
        x1 = self.shallow_feat3(torch.cat((x1, y1), 1))
        x1 = x1 + self.trans_enc1(feat_h) + self.trans_dec1(resx[0])
        xfeat1 = self.block1(x1)
        xfeat2 = self.block2(xfeat1)
        xfeat3 = self.block3(xfeat2)
        outfeat2 = xfeat3
        x2 = self.conv3_1(xfeat3)
        x3 = self.conv3_2(x2)
        img3 = x3 + bicubic_x
        return img1, img2, img3, outfeat1, outfeat2

class Net(nn.Module):
    def __init__(self, num_channels=4, base_filter=28, n_feat=28, num_stages=3, scale_factor=4,
                 scale_feat=28, kernel_size=3, reduction=4, bias=False):
        super(Net, self).__init__()
        act = nn.PReLU()
        modules_body1 = []
        modules_body1.append(conv(num_channels, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        modules_body1.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body1.append(act)
        self.body1_1 = nn.Sequential(DeconvBlock(num_channels, n_feat, 4, 4, 0, activation='prelu', norm=None),
                                     act,
                                     conv(n_feat, n_feat, kernel_size, bias=bias),
                                     act)
        self.body2_1 = nn.Sequential(*modules_body1)
        self.body3_1 = nn.Sequential(*modules_body1)

        modules_body2 = []
        modules_body2.append(conv(1, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        modules_body2.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body2.append(act)
        self.body1_2 = nn.Sequential(*modules_body2)
        self.body2_2 = nn.Sequential(*modules_body2)
        self.body3_2 = nn.Sequential(*modules_body2)

        self.shallow_feat1 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                             CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                             SAB(n_feat, kernel_size, bias=bias))
        self.shallow_feat2 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))
        self.shallow_feat3 = nn.Sequential(conv(n_feat * 2, n_feat, kernel_size, bias=bias), act,
                                            CAB(n_feat, kernel_size, reduction, bias=bias, act=act),
                                           SAB(n_feat, kernel_size, bias=bias))

        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, bias, act, scale_feat)
        self.conv1_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv1_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv2_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv2_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())
        self.conv3_1 = nn.Sequential(conv(n_feat, n_feat, kernel_size, bias=bias), act)
        self.conv3_2 = nn.Sequential(conv(n_feat, num_channels, kernel_size, bias=bias), nn.Tanh())

        self.block1 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block2 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.block3 = ORB(n_feat, kernel_size, reduction, act, bias)
        self.trans_enc1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_enc2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                        conv(n_feat*2, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_enc3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
                                        conv(n_feat*3, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec1 = nn.Sequential(conv(n_feat, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bicubic'),
                                        conv(n_feat*2, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        self.trans_dec3 = nn.Sequential(nn.Upsample(scale_factor=4, mode='bicubic'),
                                        conv(n_feat*3, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)

        self.DBPNnet = DBPN_block(n_feat, base_filter, n_feat, num_stages, scale_factor)
        self.up = nn.Upsample(scale_factor=4, mode='bicubic')

    def forward(self, ms, pan):
        # Two Patches for Stage 2
        _,_,H,W = ms.shape
        bicubic_x = self.up(ms)
        ms2top_img = ms[:,:,0:int(H/2)+4,:]
        ms2bot_img = ms[:,:,int(H/2)-4:H,:]
        pan2top_img = pan[:, :, 0:int(H*4 / 2)+16, :]
        pan2bot_img = pan[:, :, int(H*4 / 2)-16:H*4, :]

        # Four Patches for Stage 1
        ms1ltop_img = ms2top_img[:,:,:,0:int(W/2)+4]
        ms1rtop_img = ms2top_img[:,:,:,int(W/2)-4:W]
        ms1lbot_img = ms2bot_img[:,:,:,0:int(W/2)+4]
        ms1rbot_img = ms2bot_img[:,:,:,int(W/2)-4:W]

        pan1ltop_img = pan2top_img[:, :, :, 0:int(W*4 / 2)+16]
        pan1rtop_img = pan2top_img[:, :, :, int(W*4 / 2)-16:W*4]
        pan1lbot_img = pan2bot_img[:, :, :, 0:int(W*4 / 2)+16]
        pan1rbot_img = pan2bot_img[:, :, :, int(W*4 / 2)-16:W*4]
        _,_,H,W = ms1ltop_img.shape

        ##stage1
        bicubic_xtop = self.up(ms2top_img)
        bicubic_xbot = self.up(ms2bot_img)
        xbodyltop = self.body1_1(ms1ltop_img)
        xbodyrtop = self.body1_1(ms1rtop_img)
        xbodylbot = self.body1_1(ms1lbot_img)
        xbodyrbot = self.body1_1(ms1rbot_img)

        ybodyltop = self.body1_2(pan1ltop_img)
        ybodyrtop = self.body1_2(pan1rtop_img)
        ybodylbot = self.body1_2(pan1lbot_img)
        ybodyrbot = self.body1_2(pan1rbot_img)

        x1ltop = self.shallow_feat1(torch.cat((xbodyltop, ybodyltop), 1))#[148,148]
        x1rtop = self.shallow_feat1(torch.cat((xbodyrtop, ybodyrtop), 1))
        x1lbot = self.shallow_feat1(torch.cat((xbodylbot, ybodylbot), 1))
        x1rbot = self.shallow_feat1(torch.cat((xbodyrbot, ybodyrbot), 1))

        xtop, xbot, concat1_l, concat1_h, concat3_l, concat3_h = self.DBPNnet(x1ltop, x1rtop, x1lbot, x1rbot)
        xtoptemp = self.conv1_1(xtop)##[8,16,264,264]
        xbottemp = self.conv1_1(xbot)
        xtop = self.conv1_2(xtoptemp)
        xbot = self.conv1_2(xbottemp)
        img1top = xtop + bicubic_xtop
        img1bot = xbot + bicubic_xbot
        ltop = [concat1_l[0], concat3_l[0]]
        lbot = [concat1_l[1], concat3_l[1]]
        htop = [concat1_h[0], concat3_h[0]]
        hbot = [concat1_h[1], concat3_h[1]]
        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            if i==0:
                y=img1top[:, :, W * 4 - 32:W * 4 - 31, :]*y1 + img1bot[:, :, :1, :]*(1-y1)
            else:
                y=torch.cat((y, img1top[:, :, W * 4 - 32+i
                  :W * 4 - 31+i, :]*y1 + img1bot[:, :, i:i+1, :]*(1-y1)), 2)
        img1 = torch.cat((img1top[:, :, :W * 4 - 32, :], y, img1bot[:, :, 32:, :]), 2)

        ##stage2
        xtop = self.body2_1(img1top)
        xbot = self.body2_1(img1bot)
        ytop = self.body2_2(pan2top_img)
        ybot = self.body2_2(pan2bot_img)

        xfeattop = self.shallow_feat2(torch.cat((xtop, ytop), 1))
        xfeatbot = self.shallow_feat2(torch.cat((xbot, ybot), 1))
        feat2_top = self.stage1_encoder(xfeattop, ltop, htop)
        feat2_bot = self.stage1_encoder(xfeatbot, lbot, hbot)
        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 16))
            y3 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
            if i==0:
                yy0=feat2_top[0][:, :, W * 4 - 32:W * 4 - 31, :]*y1 + feat2_bot[0][:, :, :1, :]*(1-y1)
                yy1 = feat2_top[1][:, :, W * 2 - 16:W * 2 - 15, :] * y2 + feat2_bot[1][:, :, :1, :] * (1 - y2)
                yy2 = feat2_top[2][:, :, W * 1 - 8:W * 1 - 7, :] * y3 + feat2_bot[2][:, :, :1, :] * (1 - y3)
            elif i>=16:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32+i
                      :W * 4 - 31+i, :]*y1 + feat2_bot[0][:, :, i:i+1, :]*(1-y1)), 2)
            elif i>=8:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
                      :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
                yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
                      :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
            else:
                yy0 = torch.cat((yy0, feat2_top[0][:, :, W * 4 - 32 + i
                      :W * 4 - 31 + i, :] * y1 + feat2_bot[0][:, :, i:i + 1, :] * (1 - y1)), 2)
                yy1 = torch.cat((yy1, feat2_top[1][:, :, W * 2 - 16 + i
                      :W * 2 - 15 + i, :] * y2 + feat2_bot[1][:, :, i:i + 1, :] * (1 - y2)), 2)
                yy2 = torch.cat((yy2, feat2_top[2][:, :, W * 1 - 8 + i
                      :W * 1 - 7 + i, :] * y3 + feat2_bot[2][:, :, i:i + 1, :] * (1 - y3)), 2)

        feat_h = torch.cat((feat2_top[0][:, :, :W * 4 - 32, :], yy0,
                            feat2_bot[0][:, :, 32:, :]), 2)
        feat_m = torch.cat((feat2_top[1][:, :, :W * 2 - 16, :], yy1,
                            feat2_bot[1][:, :, 16:, :]), 2)
        feat_l = torch.cat((feat2_top[2][:, :, :W - 8, :], yy2,
                            feat2_bot[2][:, :, 8:, :]), 2)

        resx = self.stage1_decoder(feat_h, feat_m, feat_l)
        outfeat1 = resx[0]
        x2 = self.conv2_1(resx[0])
        x3 = self.conv2_2(x2)
        img2 = x3 + bicubic_x

        #stage3
        x1 = self.body3_1(img2)
        y1 = self.body3_2(pan)
        x1 = self.shallow_feat3(torch.cat((x1, y1), 1))
        x1 = x1 + self.trans_enc1(feat_h) + self.trans_dec1(resx[0])
        xfeat1 = self.block1(x1)
        xfeat2 = self.block2(xfeat1)
        xfeat3 = self.block3(xfeat2)
        outfeat2 = xfeat3
        x2 = self.conv3_1(xfeat3)
        x3 = self.conv3_2(x2)
        img3 = x3 + bicubic_x
        return img1, img2, img3, outfeat1, outfeat2


class DBPN_block1(nn.Module):
    def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
        super(DBPN_block1, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        self.base_filter = base_filter
        # Initial Feature Extraction
        self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm=None),
                                  ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm=None))
        # Back-projection stages
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        # Reconstruction
        self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, xltop, xrtop, xlbot, xrbot):
        # x = self.feat0(x)
        # x = self.feat1(x)
        _,_,w,h = xltop.shape
        # print(xltop.shape)
        xfeatltop = self.feat(xltop)
        xfeatrtop = self.feat(xrtop)
        xfeatlbot = self.feat(xlbot)
        xfeatrbot = self.feat(xrbot)

        l1ltop = self.down1(xfeatltop)
        h1ltop = self.up1(l1ltop)
        lltop = self.down2(h1ltop)
        concat_lltop = torch.cat((lltop, l1ltop), 1)
        hltop = self.up2(concat_lltop)
        concat_hltop = torch.cat((hltop, h1ltop), 1)
        xltop = self.output_conv(concat_hltop)

        l1rtop = self.down1(xfeatrtop)
        h1rtop = self.up1(l1rtop)
        lrtop = self.down2(h1rtop)
        concat_lrtop = torch.cat((lrtop, l1rtop), 1)
        hrtop = self.up2(concat_lrtop)
        concat_hrtop = torch.cat((hrtop, h1rtop), 1)
        xrtop = self.output_conv(concat_hrtop)

        l1lbot = self.down1(xfeatlbot)
        h1lbot = self.up1(l1lbot)
        llbot = self.down2(h1lbot)
        concat_llbot = torch.cat((llbot, l1lbot), 1)
        hlbot = self.up2(concat_llbot)
        concat_hlbot = torch.cat((hlbot, h1lbot), 1)
        xlbot = self.output_conv(concat_hlbot)

        l1rbot = self.down1(xfeatrbot)
        h1rbot = self.up1(l1rbot)
        lrbot = self.down2(h1rbot)
        concat_lrbot = torch.cat((lrbot, l1rbot), 1)
        hrbot = self.up2(concat_lrbot)
        concat_hrbot = torch.cat((hrbot, h1rbot), 1)
        xrbot = self.output_conv(concat_hrbot)

        # print(concat_lltop[:,:,:,:32].shape)
        # print(concat_lltop[:,:,:,32:].shape)
        # print(concat_lrtop[:,:,:,:2].shape)
        # print(concat_lrtop[:,:,:,2:].shape)
        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
            if i==0:
                ylt = concat_lltop[:, :, :, w // 4 - 8:w // 4 - 7] * y2 + concat_lrtop[:, :, :, :1] * (1 - y2)
                ylb = concat_llbot[:, :, :, w // 4 - 8:w // 4 - 7] * y2 + concat_lrbot[:, :, :, :1] * (1 - y2)
                yht = concat_hltop[:, :, :, w - 32:w - 31] * y1 + concat_hrtop[:, :, :, :1] * (1 - y1)
                yhb = concat_hlbot[:, :, :, w - 32:w - 31] * y1 + concat_hrbot[:, :, :, :1] * (1 - y1)
            elif i>=8:
                yht = torch.cat((yht, concat_hltop[:, :, :, w - 32 + i
                    :w - 31 + i] * y1 + concat_hrtop[:, :, :, i:i + 1] * (1 - y1)), 3)
                yhb = torch.cat((yhb, concat_hlbot[:, :, :, w - 32 + i
                    :w - 31 + i] * y1 + concat_hrbot[:, :, :, i:i + 1] * (1 - y1)), 3)
            else:
                ylt=torch.cat((ylt, concat_lltop[:, :, :, w // 4 - 8+i
                  :w // 4 - 7+i]*y2 + concat_lrtop[:, :, :, i:i+1]*(1-y2)), 3)
                ylb = torch.cat((ylb, concat_llbot[:, :, :, w // 4 - 8 + i
                  :w // 4 - 7 + i] * y2 + concat_lrbot[:, :, :, i:i + 1] * (1 - y2)), 3)
                yht = torch.cat((yht, concat_hltop[:, :, :, w - 32 + i
                    :w - 31 + i] * y1 + concat_hrtop[:, :, :, i:i + 1] * (1 - y1)), 3)
                yhb = torch.cat((yhb, concat_hlbot[:, :, :, w - 32 + i
                    :w - 31 + i] * y1 + concat_hrbot[:, :, :, i:i + 1] * (1 - y1)), 3)

        concat_ltop = torch.cat((concat_lltop[:, :, :, :w // 4 - 8], ylt,
                                 concat_lrtop[:, :, :, 8:]), 3)
        concat_lbot = torch.cat((concat_llbot[:, :, :, :w // 4 - 8], ylb,
                                 concat_lrbot[:, :, :, 8:]), 3)
        concat_htop = torch.cat((concat_hltop[:, :, :, :w - 32], yht,
                                 concat_hrtop[:, :, :, 32:]), 3)
        concat_hbot = torch.cat((concat_hlbot[:, :, :, :w - 32], yhb,
                                 concat_hrbot[:, :, :, 32:]), 3)

        # concat_ltop = torch.cat((concat_lltop[:,:,:,:w//4-8], (concat_lltop[:,:,:,w//4-8:] +
        #                          concat_lrtop[:,:,:,:8])/2, concat_lrtop[:,:,:,8:]), 3)
        # concat_lbot = torch.cat((concat_llbot[:, :, :, :w//4-8], (concat_llbot[:, :, :, w//4-8:] +
        #                          concat_lrbot[:, :, :, :8]) / 2, concat_lrbot[:, :, :, 8:]), 3)
        # concat_htop = torch.cat((concat_hltop[:,:,:,:w-32], (concat_hltop[:,:,:,w-32:] +
        #                          concat_hrtop[:,:,:,:32])/2, concat_hrtop[:,:,:,32:]), 3)
        # concat_hbot = torch.cat((concat_hlbot[:, :, :, :w-32], (concat_hlbot[:, :, :, w-32:] +
        #                          concat_hrbot[:, :, :, :32]) / 2, concat_hrbot[:, :, :, 32:]), 3)
        concat_l = [concat_ltop, concat_lbot]
        concat_h = [concat_htop, concat_hbot]
        # print('need',concat_lbot.shape)
        # print('need2', concat_ltop.shape)
        # sys.exit(0)

        return xltop, xrtop, xlbot, xrbot, concat_l, concat_h


class DBPN_block2(nn.Module):
    def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
        super(DBPN_block2, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        self.base_filter = base_filter
        # Initial Feature Extraction
        self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm=None),
                                  ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm=None))
        # Back-projection stages
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        # Reconstruction
        self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, xltop, xrtop, xlbot, xrbot):
        xfeatltop = self.feat(xltop)
        xfeatrtop = self.feat(xrtop)
        xfeatlbot = self.feat(xlbot)
        xfeatrbot = self.feat(xrbot)

        l1ltop = self.down1(xfeatltop)
        h1ltop = self.up1(l1ltop)
        l1rtop = self.down1(xfeatrtop)
        h1rtop = self.up1(l1rtop)
        l1lbot = self.down1(xfeatlbot)
        h1lbot = self.up1(l1lbot)
        l1rbot = self.down1(xfeatrbot)
        h1rbot = self.up1(l1rbot)
        _,_,H,W = l1ltop.shape

        # print(l1ltop.shape)
        # sys.exit(0)

        for i in range(32):
            y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
            y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
            if i==0:
                ylt = l1ltop[:, :, :, W // 4 - 8:W // 4 - 7] * y2 + l1rtop[:, :, :, :1] * (1 - y2)
                ylb = l1lbot[:, :, :, W // 4 - 8:W // 4 - 7] * y2 + l1rbot[:, :, :, :1] * (1 - y2)
                yht = h1ltop[:, :, :, W - 32:W - 31] * y1 + h1rtop[:, :, :, :1] * (1 - y1)
                yhb = h1lbot[:, :, :, W - 32:W - 31] * y1 + h1rbot[:, :, :, :1] * (1 - y1)
            elif i>=8:
                yht = torch.cat((yht, h1ltop[:, :, :, W - 32 + i
                    :W - 31 + i] * y1 + h1rtop[:, :, :, i:i + 1] * (1 - y1)), 3)
                yhb = torch.cat((yhb, h1lbot[:, :, :, W - 32 + i
                    :W - 31 + i] * y1 + h1rbot[:, :, :, i:i + 1] * (1 - y1)), 3)
            else:
                ylt=torch.cat((ylt, l1ltop[:, :, :, W // 4 - 8+i
                  :W // 4 - 7+i]*y2 + l1rtop[:, :, :, i:i+1]*(1-y2)), 3)
                ylb = torch.cat((ylb, l1lbot[:, :, :, W // 4 - 8 + i
                  :W // 4 - 7 + i] * y2 + l1rbot[:, :, :, i:i + 1] * (1 - y2)), 3)
                yht = torch.cat((yht, h1ltop[:, :, :, W - 32 + i
                    :W - 31 + i] * y1 + h1rtop[:, :, :, i:i + 1] * (1 - y1)), 3)
                yhb = torch.cat((yhb, h1lbot[:, :, :, W - 32 + i
                    :W - 31 + i] * y1 + h1rbot[:, :, :, i:i + 1] * (1 - y1)), 3)
        # print(ylt.shape)
        l1top = torch.cat((l1ltop[:,:,:,:W-8], ylt,
                          l1rtop[:,:,:,8:]), 3)
        l1bot = torch.cat((l1lbot[:, :, :, :W-8], ylb,
                           l1rbot[:, :, :, 8:]), 3)
        h1top = torch.cat((h1ltop[:, :, :, :W*4-32], yht,
                           h1rtop[:, :, :, 32:]), 3)
        h1bot = torch.cat((h1lbot[:, :, :, :W*4-32], yhb,
                           h1rbot[:, :, :, 32:]), 3)

        # l1top = torch.cat((l1ltop[:,:,:,:W-8],
        #                   (l1ltop[:,:,:,W-8:] + l1rtop[:,:,:,:8])/2,
        #                   l1rtop[:,:,:,8:]), 3)
        # l1bot = torch.cat((l1lbot[:, :, :, :W-8],
        #                    (l1lbot[:, :, :, W-8:] + l1rbot[:, :, :, :8]) / 2,
        #                    l1rbot[:, :, :, 8:]), 3)
        # h1top = torch.cat((h1ltop[:, :, :, :W*4-32],
        #                    (h1ltop[:, :, :, W*4-32:] + h1rtop[:, :, :, :32]) / 2,
        #                    h1rtop[:, :, :, 32:]), 3)
        # h1bot = torch.cat((h1lbot[:, :, :, :W*4-32],
        #                    (h1lbot[:, :, :, W*4-32:] + h1rbot[:, :, :, :32]) / 2,
        #                    h1rbot[:, :, :, 32:]), 3)

        # print('l1ltop:',l1ltop.shape)
        # print('l1top: ', l1top.shape)
        # print('l1bot: ', l1bot.shape)
        # print('h1top: ', h1top.shape)
        # print('h1bot: ', h1bot.shape)

        ltop = self.down2(h1top)
        concat_ltop = torch.cat((ltop, l1top), 1)
        htop = self.up2(concat_ltop)
        concat_htop = torch.cat((htop, h1top), 1)
        xtop = self.output_conv(concat_htop)

        lbot = self.down2(h1bot)
        concat_lbot = torch.cat((lbot, l1bot), 1)
        hbot = self.up2(concat_lbot)
        concat_hbot = torch.cat((hbot, h1bot), 1)
        xbot = self.output_conv(concat_hbot)

        return xtop, xbot


class DBPN_block3(nn.Module):
    def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
        super(DBPN_block3, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2
        self.base_filter = base_filter
        # Initial Feature Extraction
        self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm=None),
                                  ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm=None))
        # Back-projection stages
        self.down1 = DownBlock(base_filter, kernel, stride, padding)
        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down2 = DownBlock(base_filter, kernel, stride, padding)
        self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
        # Reconstruction
        self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm=None)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, xtop, xbot):
        xfeattop = self.feat(xtop)
        xfeatbot = self.feat(xbot)

        l1top = self.down1(xfeattop)
        h1top = self.up1(l1top)
        ltop = self.down2(h1top)
        concat_ltop = torch.cat((ltop, l1top), 1)
        htop = self.up2(concat_ltop)
        concat_htop = torch.cat((htop, h1top), 1)
        xtop = self.output_conv(concat_htop)

        l1bot = self.down1(xfeatbot)
        h1bot = self.up1(l1bot)
        lbot = self.down2(h1bot)
        concat_lbot = torch.cat((lbot, l1bot), 1)
        hbot = self.up2(concat_lbot)
        concat_hbot = torch.cat((hbot, h1bot), 1)
        xbot = self.output_conv(concat_hbot)

        # print('lbot',concat_lbot.shape)
        # print('ltop', concat_ltop.shape)
        concat_l = [concat_ltop, concat_lbot]
        concat_h = [concat_htop, concat_hbot]

        return xtop, xbot, concat_l, concat_h

#
# class DBPN_block1(nn.Module):
#     def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
#         super(DBPN_block1, self).__init__()
#
#         if scale_factor == 2:
#             kernel = 6
#             stride = 2
#             padding = 2
#         elif scale_factor == 4:
#             kernel = 8
#             stride = 4
#             padding = 2
#         elif scale_factor == 8:
#             kernel = 12
#             stride = 8
#             padding = 2
#         self.base_filter = base_filter
#         # Initial Feature Extraction
#         self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm='batch'),
#                                   ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm='batch'))
#         # Back-projection stages
#         self.down1 = DownBlock(base_filter, kernel, stride, padding)
#         self.up1 = UpBlock(base_filter, kernel, stride, padding)
#         self.down2 = DownBlock(base_filter, kernel, stride, padding)
#         self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
#         # self.down3 = D_DownBlock(base_filter, kernel, stride, padding, 2)
#         # self.up3 = D_UpBlock(base_filter, kernel, stride, padding, 2)
#         # self.down4 = D_DownBlock(base_filter, kernel, stride, padding, 3)
#         # self.up4 = D_UpBlock(base_filter, kernel, stride, padding, 3)
#         # self.down5 = D_DownBlock(base_filter, kernel, stride, padding, 4)
#         # self.up5 = D_UpBlock(base_filter, kernel, stride, padding, 4)
#         # self.down6 = D_DownBlock(base_filter, kernel, stride, padding, 5)
#         # self.up6 = D_UpBlock(base_filter, kernel, stride, padding, 5)
#         # self.down7 = D_DownBlock(base_filter, kernel, stride, padding, 6)
#         # self.up7 = D_UpBlock(base_filter, kernel, stride, padding, 6)
#         # Reconstruction
#         self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm='batch')
#
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.find('Conv2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif classname.find('ConvTranspose2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, xltop, xrtop, xlbot, xrbot):
#         # x = self.feat0(x)
#         # x = self.feat1(x)
#         _,_,w,h = xltop.shape
#         # print(xltop.shape)
#         xfeatltop = self.feat(xltop)
#         xfeatrtop = self.feat(xrtop)
#         xfeatlbot = self.feat(xlbot)
#         xfeatrbot = self.feat(xrbot)
#
#         l1ltop = self.down1(xfeatltop)
#         h1ltop = self.up1(l1ltop)
#         lltop = self.down2(h1ltop)
#         concat_lltop = torch.cat((lltop, l1ltop), 1)
#         hltop = self.up2(concat_lltop)
#         concat_hltop = torch.cat((hltop, h1ltop), 1)
#         xltop = self.output_conv(concat_hltop)
#
#         l1rtop = self.down1(xfeatrtop)
#         h1rtop = self.up1(l1rtop)
#         lrtop = self.down2(h1rtop)
#         concat_lrtop = torch.cat((lrtop, l1rtop), 1)
#         hrtop = self.up2(concat_lrtop)
#         concat_hrtop = torch.cat((hrtop, h1rtop), 1)
#         xrtop = self.output_conv(concat_hrtop)
#
#         l1lbot = self.down1(xfeatlbot)
#         h1lbot = self.up1(l1lbot)
#         llbot = self.down2(h1lbot)
#         concat_llbot = torch.cat((llbot, l1lbot), 1)
#         hlbot = self.up2(concat_llbot)
#         concat_hlbot = torch.cat((hlbot, h1lbot), 1)
#         xlbot = self.output_conv(concat_hlbot)
#
#         l1rbot = self.down1(xfeatrbot)
#         h1rbot = self.up1(l1rbot)
#         lrbot = self.down2(h1rbot)
#         concat_lrbot = torch.cat((lrbot, l1rbot), 1)
#         hrbot = self.up2(concat_lrbot)
#         concat_hrbot = torch.cat((hrbot, h1rbot), 1)
#         xrbot = self.output_conv(concat_hrbot)
#
#         # print(concat_lltop[:,:,:,:32].shape)
#         # print(concat_lltop[:,:,:,32:].shape)
#         # print(concat_lrtop[:,:,:,:2].shape)
#         # print(concat_lrtop[:,:,:,2:].shape)
#         for i in range(32):
#             y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
#             y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
#             if i==0:
#                 ylt = concat_lltop[:, :, :, w // 4 - 8:w // 4 - 7] * y2 + concat_lrtop[:, :, :, :1] * (1 - y2)
#                 ylb = concat_llbot[:, :, :, w // 4 - 8:w // 4 - 7] * y2 + concat_lrbot[:, :, :, :1] * (1 - y2)
#                 yht = concat_hltop[:, :, :, w - 32:w - 31] * y1 + concat_hrtop[:, :, :, :1] * (1 - y1)
#                 yhb = concat_hlbot[:, :, :, w - 32:w - 31] * y1 + concat_hrbot[:, :, :, :1] * (1 - y1)
#             elif i>=8:
#                 yht = torch.cat((yht, concat_hltop[:, :, :, w - 32 + i
#                     :w - 31 + i] * y1 + concat_hrtop[:, :, :, i:i + 1] * (1 - y1)), 3)
#                 yhb = torch.cat((yhb, concat_hlbot[:, :, :, w - 32 + i
#                     :w - 31 + i] * y1 + concat_hrbot[:, :, :, i:i + 1] * (1 - y1)), 3)
#             else:
#                 ylt=torch.cat((ylt, concat_lltop[:, :, :, w // 4 - 8+i
#                   :w // 4 - 7+i]*y2 + concat_lrtop[:, :, :, i:i+1]*(1-y2)), 3)
#                 ylb = torch.cat((ylb, concat_llbot[:, :, :, w // 4 - 8 + i
#                   :w // 4 - 7 + i] * y2 + concat_lrbot[:, :, :, i:i + 1] * (1 - y2)), 3)
#                 yht = torch.cat((yht, concat_hltop[:, :, :, w - 32 + i
#                     :w - 31 + i] * y1 + concat_hrtop[:, :, :, i:i + 1] * (1 - y1)), 3)
#                 yhb = torch.cat((yhb, concat_hlbot[:, :, :, w - 32 + i
#                     :w - 31 + i] * y1 + concat_hrbot[:, :, :, i:i + 1] * (1 - y1)), 3)
#
#         concat_ltop = torch.cat((concat_lltop[:, :, :, :w // 4 - 8], ylt,
#                                  concat_lrtop[:, :, :, 8:]), 3)
#         concat_lbot = torch.cat((concat_llbot[:, :, :, :w // 4 - 8], ylb,
#                                  concat_lrbot[:, :, :, 8:]), 3)
#         concat_htop = torch.cat((concat_hltop[:, :, :, :w - 32], yht,
#                                  concat_hrtop[:, :, :, 32:]), 3)
#         concat_hbot = torch.cat((concat_hlbot[:, :, :, :w - 32], yhb,
#                                  concat_hrbot[:, :, :, 32:]), 3)
#
#         # concat_ltop = torch.cat((concat_lltop[:,:,:,:w//4-8], (concat_lltop[:,:,:,w//4-8:] +
#         #                          concat_lrtop[:,:,:,:8])/2, concat_lrtop[:,:,:,8:]), 3)
#         # concat_lbot = torch.cat((concat_llbot[:, :, :, :w//4-8], (concat_llbot[:, :, :, w//4-8:] +
#         #                          concat_lrbot[:, :, :, :8]) / 2, concat_lrbot[:, :, :, 8:]), 3)
#         # concat_htop = torch.cat((concat_hltop[:,:,:,:w-32], (concat_hltop[:,:,:,w-32:] +
#         #                          concat_hrtop[:,:,:,:32])/2, concat_hrtop[:,:,:,32:]), 3)
#         # concat_hbot = torch.cat((concat_hlbot[:, :, :, :w-32], (concat_hlbot[:, :, :, w-32:] +
#         #                          concat_hrbot[:, :, :, :32]) / 2, concat_hrbot[:, :, :, 32:]), 3)
#         concat_l = [concat_ltop, concat_lbot]
#         concat_h = [concat_htop, concat_hbot]
#         # print('need',concat_lbot.shape)
#         # print('need2', concat_ltop.shape)
#         # sys.exit(0)
#
#         return xltop, xrtop, xlbot, xrbot, concat_l, concat_h
#
#
# class DBPN_block2(nn.Module):
#     def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
#         super(DBPN_block2, self).__init__()
#
#         if scale_factor == 2:
#             kernel = 6
#             stride = 2
#             padding = 2
#         elif scale_factor == 4:
#             kernel = 8
#             stride = 4
#             padding = 2
#         elif scale_factor == 8:
#             kernel = 12
#             stride = 8
#             padding = 2
#         self.base_filter = base_filter
#         # Initial Feature Extraction
#         self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm='batch'),
#                                   ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm='batch'))
#         # Back-projection stages
#         self.down1 = DownBlock(base_filter, kernel, stride, padding)
#         self.up1 = UpBlock(base_filter, kernel, stride, padding)
#         self.down2 = DownBlock(base_filter, kernel, stride, padding)
#         self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
#         # Reconstruction
#         self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm='batch')
#
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.find('Conv2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif classname.find('ConvTranspose2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, xltop, xrtop, xlbot, xrbot):
#         xfeatltop = self.feat(xltop)
#         xfeatrtop = self.feat(xrtop)
#         xfeatlbot = self.feat(xlbot)
#         xfeatrbot = self.feat(xrbot)
#
#         l1ltop = self.down1(xfeatltop)
#         h1ltop = self.up1(l1ltop)
#         l1rtop = self.down1(xfeatrtop)
#         h1rtop = self.up1(l1rtop)
#         l1lbot = self.down1(xfeatlbot)
#         h1lbot = self.up1(l1lbot)
#         l1rbot = self.down1(xfeatrbot)
#         h1rbot = self.up1(l1rbot)
#         _,_,H,W = l1ltop.shape
#
#         # print(l1ltop.shape)
#         # sys.exit(0)
#
#         for i in range(32):
#             y1 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i/32))
#             y2 = 0.5 + 0.5 * torch.cos(torch.tensor(np.pi * i / 8))
#             if i==0:
#                 ylt = l1ltop[:, :, :, W // 4 - 8:W // 4 - 7] * y2 + l1rtop[:, :, :, :1] * (1 - y2)
#                 ylb = l1lbot[:, :, :, W // 4 - 8:W // 4 - 7] * y2 + l1rbot[:, :, :, :1] * (1 - y2)
#                 yht = h1ltop[:, :, :, W - 32:W - 31] * y1 + h1rtop[:, :, :, :1] * (1 - y1)
#                 yhb = h1lbot[:, :, :, W - 32:W - 31] * y1 + h1rbot[:, :, :, :1] * (1 - y1)
#             elif i>=8:
#                 yht = torch.cat((yht, h1ltop[:, :, :, W - 32 + i
#                     :W - 31 + i] * y1 + h1rtop[:, :, :, i:i + 1] * (1 - y1)), 3)
#                 yhb = torch.cat((yhb, h1lbot[:, :, :, W - 32 + i
#                     :W - 31 + i] * y1 + h1rbot[:, :, :, i:i + 1] * (1 - y1)), 3)
#             else:
#                 ylt=torch.cat((ylt, l1ltop[:, :, :, W // 4 - 8+i
#                   :W // 4 - 7+i]*y2 + l1rtop[:, :, :, i:i+1]*(1-y2)), 3)
#                 ylb = torch.cat((ylb, l1lbot[:, :, :, W // 4 - 8 + i
#                   :W // 4 - 7 + i] * y2 + l1rbot[:, :, :, i:i + 1] * (1 - y2)), 3)
#                 yht = torch.cat((yht, h1ltop[:, :, :, W - 32 + i
#                     :W - 31 + i] * y1 + h1rtop[:, :, :, i:i + 1] * (1 - y1)), 3)
#                 yhb = torch.cat((yhb, h1lbot[:, :, :, W - 32 + i
#                     :W - 31 + i] * y1 + h1rbot[:, :, :, i:i + 1] * (1 - y1)), 3)
#
#         l1top = torch.cat((l1ltop[:,:,:,:W-8], ylt,
#                           l1rtop[:,:,:,8:]), 3)
#         l1bot = torch.cat((l1lbot[:, :, :, :W-8], ylb,
#                            l1rbot[:, :, :, 8:]), 3)
#         h1top = torch.cat((h1ltop[:, :, :, :W*4-32], yht,
#                            h1rtop[:, :, :, 32:]), 3)
#         h1bot = torch.cat((h1lbot[:, :, :, :W*4-32], yhb,
#                            h1rbot[:, :, :, 32:]), 3)
#
#         # l1top = torch.cat((l1ltop[:,:,:,:W-8],
#         #                   (l1ltop[:,:,:,W-8:] + l1rtop[:,:,:,:8])/2,
#         #                   l1rtop[:,:,:,8:]), 3)
#         # l1bot = torch.cat((l1lbot[:, :, :, :W-8],
#         #                    (l1lbot[:, :, :, W-8:] + l1rbot[:, :, :, :8]) / 2,
#         #                    l1rbot[:, :, :, 8:]), 3)
#         # h1top = torch.cat((h1ltop[:, :, :, :W*4-32],
#         #                    (h1ltop[:, :, :, W*4-32:] + h1rtop[:, :, :, :32]) / 2,
#         #                    h1rtop[:, :, :, 32:]), 3)
#         # h1bot = torch.cat((h1lbot[:, :, :, :W*4-32],
#         #                    (h1lbot[:, :, :, W*4-32:] + h1rbot[:, :, :, :32]) / 2,
#         #                    h1rbot[:, :, :, 32:]), 3)
#
#         # print('l1top: ', l1top.shape)
#         # print('l1bot: ', l1bot.shape)
#         # print('h1top: ', h1top.shape)
#         # print('h1bot: ', h1bot.shape)
#
#         ltop = self.down2(h1top)
#         concat_ltop = torch.cat((ltop, l1top), 1)
#         htop = self.up2(concat_ltop)
#         concat_htop = torch.cat((htop, h1top), 1)
#         xtop = self.output_conv(concat_htop)
#
#         lbot = self.down2(h1bot)
#         concat_lbot = torch.cat((lbot, l1bot), 1)
#         hbot = self.up2(concat_lbot)
#         concat_hbot = torch.cat((hbot, h1bot), 1)
#         xbot = self.output_conv(concat_hbot)
#
#         return xtop, xbot
#
#
# class DBPN_block3(nn.Module):
#     def __init__(self, num_channels, base_filter, n_feat, num_stages, scale_factor):
#         super(DBPN_block3, self).__init__()
#
#         if scale_factor == 2:
#             kernel = 6
#             stride = 2
#             padding = 2
#         elif scale_factor == 4:
#             kernel = 8
#             stride = 4
#             padding = 2
#         elif scale_factor == 8:
#             kernel = 12
#             stride = 8
#             padding = 2
#         self.base_filter = base_filter
#         # Initial Feature Extraction
#         self.feat = nn.Sequential(ConvBlock(num_channels, n_feat, 3, 1, 1, activation='prelu', norm='batch'),
#                                   ConvBlock(n_feat, base_filter, 1, 1, 0, activation='prelu', norm='batch'))
#         # Back-projection stages
#         self.down1 = DownBlock(base_filter, kernel, stride, padding)
#         self.up1 = UpBlock(base_filter, kernel, stride, padding)
#         self.down2 = DownBlock(base_filter, kernel, stride, padding)
#         self.up2 = D_UpBlock(base_filter, kernel, stride, padding, 2)
#         # Reconstruction
#         self.output_conv = ConvBlock(2 * base_filter, num_channels, 3, 1, 1, activation='prelu', norm='batch')
#
#         for m in self.modules():
#             classname = m.__class__.__name__
#             if classname.find('Conv2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif classname.find('ConvTranspose2d') != -1:
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, xtop, xbot):
#         xfeattop = self.feat(xtop)
#         xfeatbot = self.feat(xbot)
#
#         l1top = self.down1(xfeattop)
#         h1top = self.up1(l1top)
#         ltop = self.down2(h1top)
#         concat_ltop = torch.cat((ltop, l1top), 1)
#         htop = self.up2(concat_ltop)
#         concat_htop = torch.cat((htop, h1top), 1)
#         xtop = self.output_conv(concat_htop)
#
#         l1bot = self.down1(xfeatbot)
#         h1bot = self.up1(l1bot)
#         lbot = self.down2(h1bot)
#         concat_lbot = torch.cat((lbot, l1bot), 1)
#         hbot = self.up2(concat_lbot)
#         concat_hbot = torch.cat((hbot, h1bot), 1)
#         xbot = self.output_conv(concat_hbot)
#
#         # print('lbot',concat_lbot.shape)
#         # print('ltop', concat_ltop.shape)
#         concat_l = [concat_ltop, concat_lbot]
#         concat_h = [concat_htop, concat_hbot]
#
#         return xtop, xbot, concat_l, concat_h
