import argparse
import os
import random
import sys
from datetime import datetime
from tqdm import trange

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data

from model.model import MobileNet2
from model.unet_model import UNet
from data import get_loaders
from logger import CsvLogger
from run import train, test

parser = argparse.ArgumentParser(description='Unet training with PyTorch')

# ============================for basic config==============================================
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers (default: 4)')

# =============================for data=====================================================
parser.add_argument('--dataroot', required = True, help='Path to train and val folders')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64. Default: float32')
parser.add_argument('--class_num',  default=41, type = int, help='The number of defocus levels')

# =============================for training=================================================
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--learning-rate', '-lr', type=float, default=5e-5, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default= 5e-5, help='Weight decay (L2 penalty).')
parser.add_argument('--log-interval', type=int, default=100, help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, help='random seed (default: 1)')

# =============================for model====================================================
parser.add_argument('--results-dir', default='./results', help='Directory to store results')
parser.add_argument('--save', '-s', type=str, default='', help='Folder to save checkpoints.')
parser.add_argument('--pre-trained', required = True, help='Path to pre-trained defocus estimation model')

# =============================for loss weight==============================================
parser.add_argument('--w1', type=float, default=0.1, help='Ssim weight')

def main():
    args = parser.parse_args()

# ============================seed==========================================================
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print("Random Seed: ", args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpus:
        torch.cuda.manual_seed_all(args.seed)

# ==========================result save folder================================================
    time_stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if args.save == '':
        args.save = time_stamp
    save_path = os.path.join(args.results_dir, args.save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    csv_logger = CsvLogger(filepath=save_path, data=None)
    csv_logger.save_params(sys.argv, args)

# ==========================training device===================================================
    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

# ==========================data type==========================================================
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')

# ==============================model===========================================================
    e_model = MobileNet2(num_classes=args.class_num)
    if args.gpus is not None:
        e_model = torch.nn.DataParallel(e_model, args.gpus)
    d_model = UNet(n_channels=3, n_classes=3)

    e_model.to(device=device, dtype=dtype)
    d_model.to(device=device, dtype=dtype)

    checkpoint = torch.load(args.pre_trained, map_location=device)
    e_model.load_state_dict(checkpoint['state_dict'])

    d_optimizer = torch.optim.RMSprop(d_model.parameters(), lr=args.learning_rate, weight_decay=args.decay,
                                      momentum=args.momentum)

# ============================data loading======================================================
    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.workers)

# ========================hyper param============================================================
    w1 = args.w1

    train_network(args.epochs,  e_model, d_model, train_loader, val_loader, d_optimizer,
                 device, dtype, args.log_interval, csv_logger, save_path, w1)

def train_network(epochs, e_model, d_model, train_loader, val_loader, d_optimizer,
                  device, dtype, log_interval, csv_logger, save_path, w1):

    best_test_loss = float("inf")
    for epoch in trange(epochs + 1):
        train_loss = train(e_model, d_model, train_loader, epoch, d_optimizer, device,
                           dtype, log_interval, w1)
        test_loss, ssim_loss, mse_loss = test(e_model, d_model, val_loader, epoch, device, dtype, w1)

        csv_logger.write({'epoch': epoch + 1, 'val_loss': test_loss, 'train_loss': train_loss, 'ssim_loss': ssim_loss, 'mse_loss':mse_loss})

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(d_model.state_dict(), os.path.join(save_path, "best_d_model_epoch_{}.pth".format(epoch)))
            print("best d loss epoch is {}".format(epoch))

    csv_logger.write_text('Best loss is {}'.format(best_test_loss))

if __name__ == '__main__':
    main()