# import sys
#
# import torch
# import math
# import torch.nn as nn
#
# class ConvBlock(torch.nn.Module):
#     def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=True, activation='prelu',
#                  norm='batch'):
#         super(ConvBlock, self).__init__()
#         self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
#
#         self.norm = norm
#         if self.norm == 'batch':
#             # self.bn = torch.nn.BatchNorm2d(output_size)
#             self.bn = torch.nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.001,
#                                                           affine=True, track_running_stats=True)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)
#
#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.conv(x))
#         else:
#             out = self.conv(x)
#
#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out
#
#
#
# class DeconvBlock(torch.nn.Module):
#     def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=False, activation='prelu',
#                  norm='batch'):
#         super(DeconvBlock, self).__init__()
#         self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
#
#         self.norm = norm
#         if self.norm == 'batch':
#             self.bn = torch.nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.001,
#                                            affine=True, track_running_stats=True)
#         elif self.norm == 'instance':
#             self.bn = torch.nn.InstanceNorm2d(output_size)
#
#         self.activation = activation
#         if self.activation == 'relu':
#             self.act = torch.nn.ReLU(True)
#         elif self.activation == 'prelu':
#             self.act = torch.nn.PReLU()
#         elif self.activation == 'lrelu':
#             self.act = torch.nn.LeakyReLU(0.2, True)
#         elif self.activation == 'tanh':
#             self.act = torch.nn.Tanh()
#         elif self.activation == 'sigmoid':
#             self.act = torch.nn.Sigmoid()
#
#     def forward(self, x):
#         if self.norm is not None:
#             out = self.bn(self.deconv(x))
#         else:
#             out = self.deconv(x)
#
#         if self.activation is not None:
#             return self.act(out)
#         else:
#             return out
#
#
# class UpBlock(torch.nn.Module):
#     def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
#         super(UpBlock, self).__init__()
#         self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#
#     def forward(self, x):
#         h0 = self.up_conv1(x)
#         l0 = self.up_conv2(h0)
#         h1 = self.up_conv3(l0 - x)
#         # print('x', x.shape)
#         # print('h0', h0.shape)
#         # print('l0', l0.shape)
#         # print('h1', h1.shape)
#         # sys.exit(0)
#         return h1 + h0
#
# def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
#     return nn.Conv2d(
#         in_channels, out_channels, kernel_size,
#         padding=(kernel_size//2), bias=bias, stride=stride)
#
# class DownSample(nn.Module):
#     def __init__(self, in_channels, s_factor, act):
#         super(DownSample, self).__init__()
#         self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
#                                   nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False),
#                                   torch.nn.BatchNorm2d(in_channels+s_factor, eps=1e-05, momentum=0.001,
#                                                        affine=True, track_running_stats=True),
#                                   act)
#
#     def forward(self, x):
#         x = self.down(x)
#         return x
#
# class SkipUpSample(nn.Module):
#     def __init__(self, in_channels, s_factor, act):
#         super(SkipUpSample, self).__init__()
#         self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
#                                 nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False),
#                                 torch.nn.BatchNorm2d(in_channels, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True),
#                                 act)
#
#     def forward(self, x, y):
#         x = self.up(x)
#         # print('x',x.shape)
#         # print('y',y.shape)
#         x = x + y
#         return x
#
# class CALayer(nn.Module):
#     def __init__(self, channel, reduction=16, bias=False):
#         super(CALayer, self).__init__()
#         # global average pooling: feature --> point
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.max_pool = nn.AdaptiveMaxPool2d(1)
#         # feature channel downscale and upscale --> channel weight
#         self.conv_du = nn.Sequential(
#                 nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
#                 torch.nn.BatchNorm2d(channel // reduction, eps=1e-05, momentum=0.001,
#                                                          affine=True, track_running_stats=True),
#                 nn.ReLU(inplace=True),
#                 nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
#                 torch.nn.BatchNorm2d(channel, eps=1e-05, momentum=0.001,
#                                                          affine=True, track_running_stats=True),
#                 nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         y1 = self.avg_pool(x)
#         y2 = self.max_pool(x)
#         y = self.conv_du(y1+y2)
#         return x * y
#
# class CAB(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act):
#         super(CAB, self).__init__()
#         modules_body = []
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body.append(act)
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                  affine=True, track_running_stats=True))
#         modules_body.append(act)
#
#         self.CA = CALayer(n_feat, reduction, bias=bias)
#         self.body = nn.Sequential(*modules_body)
#
#     def forward(self, x):
#         res = self.body(x)
#         res = self.CA(res)
#         res = res + x
#         return res
#
# class SAB(nn.Module):
#     def __init__(self, n_feat, kernel_size, bias):
#         super(SAB, self).__init__()
#         self.conv2d = nn.Sequential(conv(2, n_feat, kernel_size, bias=bias),
#                                     torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                          affine=True, track_running_stats=True))
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         avgout = torch.mean(x, dim=1, keepdim=True)
#         maxout, _ = torch.max(x, dim=1, keepdim=True)
#         out = torch.cat([avgout, maxout], dim=1)
#         out = self.sigmoid(self.conv2d(out))
#         out = out*x
#         # print('sab,',out.shape)
#         return out+x
#
#
# class enc(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act):
#         super(enc, self).__init__()
#         modules_body = []
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                   affine=True, track_running_stats=True))
#         modules_body.append(act)
#         modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
#         modules_body.append(torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                                  affine=True, track_running_stats=True))
#         modules_body.append(act)
#
#         self.body = nn.Sequential(*modules_body)
#
#         # self.conv_du = nn.Sequential(
#         #     nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
#         #     # nn.Sigmoid()
#         # )
#     def forward(self, x):
#         res = self.body(x)
#         # print(res.shape)
#         # res = self.conv_du(res)
#         res = res + x
#         return res
#
# class Encoder(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act, scale_feat):
#         super(Encoder, self).__init__()
#         self.encoder_level1 = [conv(n_feat, n_feat, kernel_size, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                affine=True, track_running_stats=True), act,
#                                enc(n_feat, kernel_size, reduction, bias, act)]
#         self.encoder_level2 = [enc(n_feat+scale_feat, kernel_size, reduction, bias, act)]
#         self.encoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, reduction, bias, act)]
#
#         self.encoder_level1 = nn.Sequential(*self.encoder_level1)
#         self.encoder_level2 = nn.Sequential(*self.encoder_level2)
#         self.encoder_level3 = nn.Sequential(*self.encoder_level3)
#
#         self.down12 = DownSample(n_feat, scale_feat, act)
#         self.down23 = DownSample(n_feat+scale_feat, scale_feat, act)
#
#         self.trans_enc1 = nn.Sequential(conv(n_feat*2, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                         nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         # self.trans_enc2 = nn.Conv2d(n_feat+scale_feat,     n_feat+scale_feat,     kernel_size=1, bias=bias)
#         self.trans_enc3 = nn.Sequential(conv(n_feat*2, n_feat+scale_feat*2, 3, bias=bias), torch.nn.BatchNorm2d(n_feat+scale_feat*2, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                        nn.Conv2d(n_feat+scale_feat*2, n_feat+scale_feat*2, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat+scale_feat*2, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#
#         self.trans_dec1 = nn.Sequential(conv(n_feat*2, n_feat, 3, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         # self.trans_dec2 = nn.Conv2d(n_feat+scale_feat,     n_feat+scale_feat,     kernel_size=1, bias=bias)
#         self.trans_dec3 = nn.Sequential(conv(n_feat*2, n_feat+scale_feat*2, 3, bias=bias), torch.nn.BatchNorm2d(n_feat+scale_feat*2, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act,
#                                        nn.Conv2d(n_feat+scale_feat*2, n_feat+scale_feat*2, kernel_size=1, bias=bias), torch.nn.BatchNorm2d(n_feat+scale_feat*2, eps=1e-05, momentum=0.001,
#                                         affine=True, track_running_stats=True), act)
#         # self.conv = conv(n_feat*2, n_feat, kernel_size, bias=bias)
#
#
#     def forward(self, x, l, h):
#         # x = self.conv(x)
#         enc1 = self.encoder_level1(x + self.trans_enc1(h[0]) + self.trans_dec1(h[1]))
#         # + self.trans_enc1(h[0]) + self.trans_dec1(h[1])
#         enc1 = enc1
#         x = self.down12(enc1)
#
#         enc2 = self.encoder_level2(x)
#         x = self.down23(enc2)
#
#         # print(self.trans_enc3(l[0]).shape)
#         # print(self.trans_dec3(l[1]).shape)
#         # print(x.shape)
#         # sys.exit(0)
#         enc3 = self.encoder_level3(x + self.trans_enc3(l[0]) + self.trans_dec3(l[1]))
#         # + self.trans_enc3(l[0]) + self.trans_dec3(l[1])
#         enc3 = enc3
#         return [enc1, enc2, enc3]
#
# class Decoder(nn.Module):
#     def __init__(self, n_feat, kernel_size, reduction, bias, act, scale_feat):
#         super(Decoder, self).__init__()
#         self.decoder_level1 = [enc(n_feat, kernel_size, reduction, bias, act)]
#         self.decoder_level2 = [enc(n_feat+scale_feat, kernel_size, reduction, bias, act)]
#         self.decoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, reduction, bias, act)]
#
#         self.decoder_level1 = nn.Sequential(*self.decoder_level1)
#         self.decoder_level2 = nn.Sequential(*self.decoder_level2)
#         self.decoder_level3 = nn.Sequential(*self.decoder_level3)
#
#         self.skip_attn1 = enc(n_feat, kernel_size, reduction, bias, act)
#         self.skip_attn2 = enc(n_feat+scale_feat, kernel_size, reduction, bias, act)
#
#         self.up21 = SkipUpSample(n_feat, scale_feat, act)
#         self.up32 = SkipUpSample(n_feat+scale_feat, scale_feat, act)
#
#     def forward(self, enc1, enc2, enc3):
#         # enc1, enc2, enc3 = outs
#         dec3 = self.decoder_level3(enc3)
#         # print('dec3:', dec3.shape)
#
#         x = self.up32(dec3, self.skip_attn2(enc2))
#         dec2 = self.decoder_level2(x)
#
#         x = self.up21(dec2, self.skip_attn1(enc1))
#         dec1 = self.decoder_level1(x)
#
#         return [dec1, dec2, dec3]
#
#
#
# class D_UpBlock(torch.nn.Module):
#     def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu'):
#         super(D_UpBlock, self).__init__()
#         self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm='batch')
#         self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#
#     def forward(self, x):
#         x = self.conv(x)
#         h0 = self.up_conv1(x)
#         l0 = self.up_conv2(h0)
#         h1 = self.up_conv3(l0 - x)
#         return h1 + h0
#
#
#
# class DownBlock(torch.nn.Module):
#     def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
#         super(DownBlock, self).__init__()
#         self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#         self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm='batch')
#
#     def forward(self, x):
#         l0 = self.down_conv1(x)
#         h0 = self.down_conv2(l0)
#         l1 = self.down_conv3(h0 - x)
#         return l1 + l0





import sys

import torch
import math
import torch.nn as nn

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=0, bias=False, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out



class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=False, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)

class DownSample(nn.Module):
    def __init__(self, in_channels, s_factor, act):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, in_channels+s_factor, 1, stride=1, padding=0, bias=False),
                                  act)

    def forward(self, x):
        x = self.down(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor, act):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels+s_factor, in_channels, 1, stride=1, padding=0, bias=False),
                                act)

    def forward(self, x, y):
        x = self.up(x)
        # print('x',x.shape)
        # print('y',y.shape)
        x = x + y
        return x

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y1 = self.avg_pool(x)
        y2 = self.max_pool(x)
        y = self.conv_du(y1+y2)
        return x * y

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res = res + x
        return res

class SAB(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAB, self).__init__()
        self.conv2d = nn.Sequential(conv(2, n_feat, kernel_size, bias=bias))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        out = out*x
        # print('sab,',out.shape)
        return out+x


class enc(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(enc, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)

        self.body = nn.Sequential(*modules_body)

        # self.conv_du = nn.Sequential(
        #     nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
        #     # nn.Sigmoid()
        # )
    def forward(self, x):
        res = self.body(x)
        # print(res.shape)
        # res = self.conv_du(res)
        res = res + x
        return res

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, scale_feat):
        super(Encoder, self).__init__()
        self.encoder_level1 = [conv(n_feat, n_feat, kernel_size, bias=bias), act,
                               enc(n_feat, kernel_size, reduction, bias, act)]
        self.encoder_level2 = [enc(n_feat+scale_feat, kernel_size, reduction, bias, act)]
        self.encoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, reduction, bias, act)]

        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)

        self.down12 = DownSample(n_feat, scale_feat, act)
        self.down23 = DownSample(n_feat+scale_feat, scale_feat, act)

        self.trans_enc1 = nn.Sequential(conv(n_feat*2, n_feat, 3, bias=bias), act,
                                        nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        # self.trans_enc2 = nn.Conv2d(n_feat+scale_feat,     n_feat+scale_feat,     kernel_size=1, bias=bias)
        self.trans_enc3 = nn.Sequential(conv(n_feat*2, n_feat+scale_feat*2, 3, bias=bias), act,
                                       nn.Conv2d(n_feat+scale_feat*2, n_feat+scale_feat*2, kernel_size=1, bias=bias), act)

        self.trans_dec1 = nn.Sequential(conv(n_feat*2, n_feat, 3, bias=bias), act,
                                       nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), act)
        # self.trans_dec2 = nn.Conv2d(n_feat+scale_feat,     n_feat+scale_feat,     kernel_size=1, bias=bias)
        self.trans_dec3 = nn.Sequential(conv(n_feat*2, n_feat+scale_feat*2, 3, bias=bias),  act,
                                       nn.Conv2d(n_feat+scale_feat*2, n_feat+scale_feat*2, kernel_size=1, bias=bias), act)
        # self.conv = conv(n_feat*2, n_feat, kernel_size, bias=bias)


    def forward(self, x, l, h):
        # x = self.conv(x)
        enc1 = self.encoder_level1(x + self.trans_enc1(h[0]) + self.trans_dec1(h[1]))
        # + self.trans_enc1(h[0]) + self.trans_dec1(h[1])
        enc1 = enc1
        x = self.down12(enc1)

        enc2 = self.encoder_level2(x)
        x = self.down23(enc2)

        # print(self.trans_enc3(l[0]).shape)
        # print(self.trans_dec3(l[1]).shape)
        # print(x.shape)
        # sys.exit(0)
        enc3 = self.encoder_level3(x + self.trans_enc3(l[0]) + self.trans_dec3(l[1]))
        # + self.trans_enc3(l[0]) + self.trans_dec3(l[1])
        enc3 = enc3
        return [enc1, enc2, enc3]

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act, scale_feat):
        super(Decoder, self).__init__()
        self.decoder_level1 = [enc(n_feat, kernel_size, reduction, bias, act)]
        self.decoder_level2 = [enc(n_feat+scale_feat, kernel_size, reduction, bias, act)]
        self.decoder_level3 = [enc(n_feat+scale_feat*2, kernel_size, reduction, bias, act)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)

        self.skip_attn1 = enc(n_feat, kernel_size, reduction, bias, act)
        self.skip_attn2 = enc(n_feat+scale_feat, kernel_size, reduction, bias, act)

        self.up21 = SkipUpSample(n_feat, scale_feat, act)
        self.up32 = SkipUpSample(n_feat+scale_feat, scale_feat, act)

    def forward(self, enc1, enc2, enc3):
        # enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        # print('dec3:', dec3.shape)

        x = self.up32(dec3, self.skip_attn2(enc2))
        dec2 = self.decoder_level2(x)

        x = self.up21(dec2, self.skip_attn1(enc1))
        dec1 = self.decoder_level1(x)

        return [dec1, dec2, dec3]



class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu'):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter * num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0



class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu'):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0




