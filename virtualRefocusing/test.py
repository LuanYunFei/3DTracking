# -*- coding: utf-8 -*-
import numpy as np
import argparse
import cv2

from model.unet_model import UNet
from model.model import MobileNet2
from data import my_loader

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

parser = argparse.ArgumentParser(description='Unet testing with PyTorch')

# ====================================for basic config=====================================
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers')

# ========================================for data=========================================
parser.add_argument('--test_dir',  required=True, type=str, help='Path to test folder')
parser.add_argument('--class_num',  default=41, type=int, help='The number of defocus levels')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64.')

# =======================================for model==========================================
parser.add_argument('--pre_trained', required=True, type=str, help='Path to pre-trained defocus estimation model')
parser.add_argument('--resume', required=True, type=str, help='Path to checkpoint of virtual redocusing model')
def main():
    args = parser.parse_args()

    # ================================= device===============================================
    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

    # ==============================data type=======================================================
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')

    # ===============================model===========================================================
    print("Load model.")
    net = UNet(n_channels=3, n_classes=3)
    net_pretrained = MobileNet2()

    net.to(device=device)
    net_pretrained.to(device=device)
    net_pretrained = torch.nn.DataParallel(net_pretrained, args.gpus)

    checkpoint = torch.load(args.pre_trained, map_location=device)
    net_pretrained.load_state_dict(checkpoint['state_dict'])
    net.load_state_dict(torch.load(args.resume, map_location=device))

    net.eval()
    net_pretrained.eval()
    print("Load model done.")

    # ===========================data loading========================================================
    test_data = my_loader(data_path=args.test_dir,augment=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # ============================evaluation=========================================================
    mse_loss_mean = 0
    ssim_loss_mean = 0
    psnr_loss_mean = 0

    with torch.no_grad():
        for test_counter, (test_name, test_image,test_label) in enumerate(test_loader):
            test_image = test_image.to(device=device, dtype=dtype)
            test_label = test_label.to(device=device, dtype=dtype)
            feature = net_pretrained(test_image)
            test_pred = net(test_image,feature)

            mse_loss_func = nn.MSELoss()
            mse_loss = mse_loss_func(test_pred,test_label)

            show_test_pred = np.transpose(np.squeeze(test_pred.cpu().detach().numpy(), axis=0),
                                              [1, 2, 0]) * 255
            show_test_label = np.transpose(np.squeeze(test_label.cpu().detach().numpy(), axis=0),
                                              [1, 2, 0]) * 255

            psnr_loss = compare_psnr(show_test_label,show_test_pred,data_range=255)
            ssim_loss = compare_ssim(show_test_label,show_test_pred,multichannel = True,data_range=255)

            mse_loss_mean = mse_loss_mean + mse_loss.item()
            ssim_loss_mean = ssim_loss_mean + ssim_loss
            psnr_loss_mean = psnr_loss_mean + psnr_loss

        mse_loss_mean = mse_loss_mean / len(test_data)
        ssim_loss_mean = ssim_loss_mean / len(test_data)
        psnr_loss_mean = psnr_loss_mean / len(test_data)
    print("Get predict result done.")
    print("MSE_loss_mean is {}".format(mse_loss_mean))
    print("ssim_loss_mean is {}".format(ssim_loss_mean))
    print ("psnr_loss_mean is {}".format(psnr_loss_mean))

if __name__ == '__main__':
    main()
