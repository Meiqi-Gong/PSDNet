from __future__ import print_function
import argparse
import sys
from math import log10

import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Networks.dbpn import Net as Net
from data import get_test_set, get_fulltest_set
import socket
import time
import scipy.io as scio
import numpy as np
from scipy.misc import imread, imsave
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--save_dir', default='results/full/', help='Location to save results')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
hostname = str(socket.gethostname())
cudnn.benchmark = True
print(opt)
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
        rmse_value = np.mean(np.square(band_img1-band_img2))/np.mean(np.square(band_img2))
        inner_sum += rmse_value
    ergas = 100/(h/l)*np.sqrt(inner_sum/channels)

    return ergas

def save_img(img, name, ep):
    save_dir = opt.save_dir+'epoch'+str(ep)+'/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = save_dir + name[59:-4]
    scio.savemat(save_fn + '.mat', {'i': img})

def test(ep):
    model.eval()
    e=0
    t=0
    for it, batch in enumerate(testing_data_loader, 1):
        ms_label, pan_label, ms, pan, name = Variable(batch[0]), Variable(batch[1]), Variable(batch[2]), Variable(
            batch[3]), batch[4]
        if cuda:
            LR = ms.cuda(gpus_list[0])
            pan = pan.cuda(gpus_list[0])
            target = ms_label.cuda(gpus_list[0])
            panl = pan_label.cuda(gpus_list[0])

        t0 = time.time()
        if int(name[0][60:-4])>=70 and int(name[0][60:-4])<=80:
            with torch.no_grad():
                prediction1, prediction2, prediction3,_,_ = model(target, panl)

            prediction3 = prediction3.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            # LR = LR.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            # tar = target.permute(0, 2, 3, 1).cpu().detach().squeeze().numpy()
            # erg = ERGAS(prediction3, tar)
            # e += erg
            t1 = time.time()
            print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
            t=t+t1-t0
            save_img(prediction3, name[0], ep)
    print('Avg. Time: {:.4f}'.format(t / len(testing_data_loader)))
    print('Avg. ERGAS: {:.4f}'.format(e / len(testing_data_loader)))


cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading test datasets')
test_set = get_test_set('/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/ms_label', '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/pan_label',
                        '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/ms', '/data2/gmq/pansharpening/SDAPS/data/WV2/testdata/pan')
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=1, shuffle=False)

model = Net()


for epoch3 in range(105, 106):
    model_name = "weights_MSE/WV2/epoch_{}.pth".format(epoch3)
    if cuda:
        model = model.cuda(gpus_list[0])

    model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
    test(epoch3)