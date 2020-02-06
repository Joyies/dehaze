#!/usr/bin/env python
import argparse
import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
from PIL import Image
import scipy.misc
from skimage.measure import compare_ssim,compare_psnr

import numpy as np

from model.model import Net


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--checkpoint", type=str, default="",help="path to load model checkpoint")
parser.add_argument("--test", type=str, default="../ITS_data/test/data", help="path to load test images")

opt = parser.parse_args()

# for i in range(669,670):
net = Net()
print(opt)

print(opt.checkpoint)
net.load_state_dict(torch.load(opt.checkpoint)['state_dict'])
net.eval()
net = nn.DataParallel(net, device_ids=[0]).cuda()

images = utils.load_all_image(opt.test)

ssims = []
psnrs = []
psnrs2 = []
str_label = "../ITS_data/test/label/"

def compute_psnr(ref_im, res_im):
    ref_img = scipy.misc.fromimage(ref_im).astype(float) / 255.0

    res_img = scipy.misc.fromimage(res_im).astype(float) / 255.0
    squared_error = np.square(ref_img - res_img)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(1.0 / mse)
    return psnr
# print(images)
for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    im2_filename = filename.split('_')[0]+'.png'
    im = Image.open(im_path)
    im2_path = str_label + im2_filename
    im2 = Image.open(im2_path)
    h, w = im.size
    im = ToTensor()(im)
    im = im.unsqueeze(0).cuda()
    im = Variable(im, volatile=True)
    im = net(im)

    im = torch.clamp(im, 0., 1.)

    im = ToPILImage()(im.cpu().data[0])
    ssims.append(compare_ssim(np.array(im), np.array(im2), multichannel=True))
    psnrs2.append(compare_psnr(np.array(im), np.array(im2)))

print(np.mean(ssims), np.mean(psnrs2))
