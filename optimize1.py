 from __future__ import print_function
import argparse
import sys

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Networks.dbpn import Net_GT as NetGT
from Networks.dbpn import Net as Net
# from Networks.dbpn_QB_GF2 import Net as Net
from data import get_training_set, get_test_set, get_fulltraining_set
import socket
import time
import socket
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize1', type=int, default=1, help='training batch size')
parser.add_argument('--batchSize2', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=50, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--MSlr', type=float, default=0.0005, help='Learning Rate. Default=0.01')#pan:5e-3
parser.add_argument('--GTlr', type=float, default=0.0002, help='Learning Rate. Default=0.01')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./Dataset')
parser.add_argument('--patch_size', type=int, default=40, help='Size of cropped HR image')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='weights_MTF/', help='Location to save checkpoint models')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
writer = SummaryWriter("logs")

def erggg(hr, tar):
    hr = hr.permute(0,2,3,1)
    tar = tar.permute(0,2,3,1)

    channels = hr.shape[3]
    # num = hr.shape[0]
    inner_sum = 0
    # for n in range(num):
    for c in range(channels):
        band_img1 = hr[:, :, :, c]
        band_img2 = tar[:, :, :, c]
        rmse_value = torch.square(torch.sqrt(torch.mean(torch.square(band_img1 - band_img2))) / torch.mean(band_img2))
        inner_sum += rmse_value
    ergas = 100 / 4 * torch.sqrt(inner_sum / channels)
    return ergas

def trainMS(epoch):
    print('Training MS!')
    if epoch<=9:
        model_GT.train()
        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        for iteration, batch in enumerate(training_data_loader2, 1):
            GT, LRMS, PAN = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
            if cuda:
                pan = PAN.cuda(gpus_list[0])
                LR = LRMS.cuda(gpus_list[0])
                target = GT.cuda(gpus_list[0])

            optimizer_GT.zero_grad()
            prediction1, prediction2, prediction3, feat1, feat2 = model_GT(target, LR, pan)
            loss_MS1 = criterion(prediction1, target)
            loss_MS2 = criterion(prediction2, target)
            loss_MS3 = criterion(prediction3, target)
            loss = loss_MS1 + loss_MS2 + loss_MS3
            epoch_loss1 += loss_MS1.data
            epoch_loss2 += loss_MS2.data
            epoch_loss3 += loss_MS3.data
            loss.backward()
            optimizer_GT.step()

            if iteration % 200 == 0:
                niter = epoch * len(training_data_loader2) + iteration
                writer.add_scalars('Train_loss',
                                   {'stage1_loss': loss_MS1.data.item()}, niter)
                writer.add_scalars('Train_loss',
                                   {'stage2_loss': loss_MS2.data.item()}, niter)
                writer.add_scalars('Train_loss',
                                   {'stage3_loss': loss_MS3.data.item()}, niter)
                writer.add_image('image_GT', target[0, 0:3, :, :], niter)
                writer.add_image('image_stage1', prediction1[0, 0:3, :, :], niter)
                writer.add_image('image_stage2', prediction2[0, 0:3, :, :], niter)
                writer.add_image('image_stage3', prediction3[0, 0:3, :, :], niter)
                print("===> Epoch[{}]({}/{}):LossMS1: {:.4f} LossMS2: {:.4f} LossMS3: {:.4f}".format(epoch, iteration,
                                                                      len(training_data_loader2),
                                                                      loss_MS1.data, loss_MS2.data, loss_MS3.data))
        print("===> Epoch {} Complete: Avg. Loss1: {:.4f} Avg. Loss2: {:.4f} Avg. Loss3: {:.4f}"
              .format(epoch, epoch_loss1 / len(training_data_loader2), epoch_loss2 / len(training_data_loader2),
                      epoch_loss3 / len(training_data_loader2)))
        checkpoint('allGT', epoch)
    else:
        model.eval()
        e = 0
        ll=0
        for it, batch in enumerate(testing_data_loader, 1):
            # with torch.no_grad():
            ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            if cuda:
                LR = ms.cuda(gpus_list[0])
                pan = pan.cuda(gpus_list[0])
                target = ms_label.cuda(gpus_list[0])

            LR = LR.type(torch.cuda.FloatTensor)
            pan = pan.type(torch.cuda.FloatTensor)
            target = target.type(torch.cuda.FloatTensor)
            with torch.no_grad():
                prediction1, prediction2, prediction3, _, _ = model(LR, pan)
            prediction3 = prediction3.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            tar = target.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            if np.std(tar)>0.05:
                erg = ERGAS(prediction3, tar)
                ll = ll + 1
                e += erg
                if ll%20==0:
                    print('ergas: ', erg)
        print('Avg. ERGAS: {:.4f}'.format(e / ll))


        model_GT.eval()
        model.train()
        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        epoch_loss = 0
        epoch_distill = 0
        for iteration, batch in enumerate(training_data_loader2, 1):
            ms_label, pan_label, ms, pan = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(batch[3])
            if cuda:
                LR = ms.cuda(gpus_list[0])
                pan = pan.cuda(gpus_list[0])
                target = ms_label.cuda(gpus_list[0])
            optimizer2.zero_grad()
            with torch.no_grad():
                _, GT_2, GT_3, GT_feat1, GT_feat2 = model_GT(target, LR, pan)
            prediction1, prediction2, prediction3, outfeat1, outfeat2 = model(LR, pan)
            loss_MS1o = criterion(prediction1, target)
            loss_MS2o = criterion(prediction2, target)
            loss_MS3o = criterion(prediction3, target)
            loss_MS1 = erggg(prediction1, target)
            loss_MS2 = erggg(prediction2, target)
            loss_MS3 = erggg(prediction3, target)
            loss_distill = 1*(MSE(outfeat2, GT_feat2)+MSE(outfeat1, GT_feat1))
            loss = loss_MS1o + loss_MS2o + loss_MS3o + 1e-6*loss_distill
            epoch_loss1 += loss_MS1.data
            epoch_loss2 += loss_MS2.data
            epoch_loss3 += loss_MS3.data
            epoch_loss += loss_MS3o
            # epoch_distill += loss_distill
            loss.backward()
            optimizer2.step()
            if iteration % 100 == 0:
                niter = epoch * len(training_data_loader2) + iteration
                writer.add_image('image_GT', target[0, 0:3, :, :], niter)
                writer.add_image('image_stage1', prediction1[0, 0:3, :, :], niter)
                writer.add_image('image_stage2', prediction2[0, 0:3, :, :], niter)
                writer.add_image('image_stage3', prediction3[0, 0:3, :, :], niter)
                print("===> Epoch[{}]({}/{}):LossMS1: {:.4f} LossMS2: {:.4f} LossMS3: {:.4f}".format(epoch, iteration,
                                                                                                     len(training_data_loader2),
                                                                                                     loss_MS1o.data,
                                                                                                     loss_MS2o.data,
                                                                                                     loss_MS3o.data))
        print("===> Epoch {} Complete: Avg. Loss1: {:.4f} Avg. Loss2: {:.4f} Avg. Loss3: {:.4f} Avg. Loss: {:.4f} "
              .format(epoch, epoch_loss1 / len(training_data_loader2), epoch_loss2 / len(training_data_loader2),
                      epoch_loss3 / len(training_data_loader2), epoch_loss / len(training_data_loader2)))
        checkpoint('all', epoch)


def checkpoint(name, epoch):
    if name == 'all':
        model_out_path = opt.save_folder + "epoch_{}.pth".format(epoch)
        torch.save(model.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))
    elif name == 'allGT':
        model_out_path = opt.save_folder + "GTepoch_{}.pth".format(epoch)
        torch.save(model_GT.state_dict(), model_out_path)
        print("Checkpoint saved to {}".format(model_out_path))

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
train_set = get_training_set('/data2/gmq/pansharpening/SDAPS/data/WV2/ms_label',
                             '/data2/gmq/pansharpening/SDAPS/data/WV2/pan_label',
                             '/data2/gmq/pansharpening/SDAPS/data/WV2/ms',
                             '/data2/gmq/pansharpening/SDAPS/data/WV2/pan')
training_data_loader2 = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize2, shuffle=True)

test_set = get_test_set('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/ms_label', '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/pan_label',
                        '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/ms', '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/pan')
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)


model = Net()
model_GT = NetGT()

criterion = nn.L1Loss()
MSE = nn.MSELoss()

if cuda:
    model_GT = model_GT.cuda(gpus_list[0])
    model = model.cuda(gpus_list[0])

optimizer_GT = optim.Adam(model_GT.parameters(), lr=opt.MSlr, betas=(0.9, 0.999), eps=1e-8)
optimizer2 = optim.Adam(model.parameters(), lr=opt.MSlr, betas=(0.9, 0.999), eps=1e-8)

def ERGAS(hr_mul, label):
    """
    calc ergas.
    """
    h = 264
    l = 66
    channels = hr_mul.shape[2]
    inner_sum = 0
    for channel in range(channels):
        band_img1 = hr_mul[:, :, channel]
        band_img2 = label[:, :, channel]
        rmse_value = np.square(np.sqrt(np.mean(np.square(band_img1 - band_img2))) / np.mean(band_img2))
        inner_sum += rmse_value
    ergas = 100/(h/l)*np.sqrt(inner_sum/channels)
    return ergas

# if opt.pretrained:
#     epoch1=9
#     epoch2=10
#     model_name = "GTepoch_{}.pth".format(epoch1)
#     model2_name = "epoch_{}.pth".format(epoch2)
#     if os.path.exists(model_name):
#         # model_init.load_state_dict(torch.load(modelinit_name, map_location=lambda storage, loc: storage))
#         # model_GT.load_state_dict(torch.load(model2_name, map_location=lambda storage, loc: storage))
#         # for k in dd_GT.keys():
#         #     # print(k)
#         #     if k.startswith('DBPNnet') or k.startswith('conv1') or k.startswith('shallow_feat1') or k.startswith(
#         #             'body1'):
#         #     # if k.startswith('conv'):
#         #         # print(k)
#         #         dd[k] = dd_GT[k]
#         # model.load_state_dict(dd)
#
#         # model_GT.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
#         model.load_state_dict(torch.load(model2_name, map_location=lambda storage, loc: storage))
#         print('Pre-trained models are loaded.')
#         # sys.exit(0)

for epoch in range(opt.start_iter, opt.nEpochs + 1):
    if epoch>10 and epoch % 5 == 0:
        for param in optimizer2.param_groups:
            param['lr'] = param['lr'] * 0.95
    if epoch > 1 and epoch % 5 == 0:
        for param in optimizer_GT.param_groups:
            param['lr'] = param['lr'] * 0.95
    print('Learning rate decay now: lr={}'.format(optimizer2.param_groups[0]['lr']))

    trainMS(epoch)