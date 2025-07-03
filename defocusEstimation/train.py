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
from torch.optim.lr_scheduler import MultiStepLR

from data import get_loaders
from logger import CsvLogger
from model import MobileNet2
from run import train, test, save_checkpoint

parser = argparse.ArgumentParser(description='MobileNetv2 training with PyTorch')
# ============================for basic config==============================================
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers')

# =============================for data=====================================================
parser.add_argument('--dataroot',  required=True, help='Path to train and val folders')
parser.add_argument('--class_num',  default=41, type=int, help='The number of defocus levels')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64.')

# ============================for training==================================================
parser.add_argument('--epochs', type=int, default=150, help='Number of epochs to train.')
parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch size')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.0005, help='The learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default= 5e-04, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma at scheduled epochs.')
parser.add_argument('--schedule', type=int, nargs='+', default=[75,110], help='Decrease learning rate at these epochs.')
parser.add_argument('--log-interval', type=int, default=100, help='Number of batches between log messages')
parser.add_argument('--seed', type=int, default=None, help='Random seed')

# =============================for model=====================================================
parser.add_argument('--results_dir', default='./results', help='Directory to store results')
parser.add_argument('--save', '-s', default='', type=str, help='Folder to save checkpoints.')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('--scaling', type=float, default=1, help='Scaling of MobileNet.')
parser.add_argument('--input-size', type=int, default=64, help='Input size, multiple of 64.')

# ============================for loss weight=================================================
parser.add_argument('--k1', type=float, default=1, help='BLO weight')
parser.add_argument('--k2', type=float, default=0.1, help='RnC weight')
parser.add_argument('--alpha', type=float, default=20, help='Decay rate in BLO')

def main():
    args = parser.parse_args()

# ============================seed============================================================
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

    csv_logger = CsvLogger(filepath=save_path)
    csv_logger.save_params(sys.argv, args)

# ===========================training device===================================================
    if args.gpus is not None:
        args.gpus = [int(i) for i in args.gpus.split(',')]
        device = 'cuda:' + str(args.gpus[0])
        cudnn.benchmark = True
    else:
        device = 'cpu'

# ==========================data type===========================================================
    if args.type == 'float64':
        dtype = torch.float64
    elif args.type == 'float32':
        dtype = torch.float32
    elif args.type == 'float16':
        dtype = torch.float16
    else:
        raise ValueError('Wrong type!')

# ========================model==================================================================
    class_num = args.class_num
    model = MobileNet2(input_size=args.input_size, scale=args.scaling, num_classes=class_num)

    # for fine-tune
    if args.resume:

        for param in model.parameters():
            param.requires_grad = False
        input_num = model.fc.in_features
        model.fc = torch.nn.Linear(input_num, class_num)

        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.gpus is not None:
        model = torch.nn.DataParallel(model, args.gpus)
    model.to(device=device, dtype=dtype)

    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum, weight_decay=args.decay,
                                nesterov=True)
    scheduler = MultiStepLR(optimizer, milestones=args.schedule, gamma=args.gamma)

# ========================data loading===========================================================
    train_loader, val_loader = get_loaders(args.dataroot, args.batch_size, args.workers, args.class_num)

# ========================hyper param============================================================
    k1 = args.k1
    k2 = args.k2
    alpha = args.alpha

    train_network(args.epochs, model, train_loader, val_loader, optimizer, scheduler,
                  device, dtype, args.log_interval, csv_logger, save_path, k1, k2, alpha)

def train_network(epochs, model, train_loader, val_loader, optimizer, scheduler, device, dtype,log_interval, csv_logger, save_path, k1, k2, alpha):

    best_test_loss = float("inf")
    for epoch in trange(epochs + 1):
        scheduler.step()
        train_loss = train(model, train_loader, epoch, optimizer, device, dtype, log_interval, k1, k2, alpha)
        test_loss,kl_loss,order_loss,rank_loss = test(model, val_loader, epoch, device, dtype, k1, k2, alpha)
        csv_logger.write({'epoch': epoch + 1, 'val_loss': test_loss,  'train_loss': train_loss,  'kl_loss': kl_loss, 'order_loss': order_loss, 'rank_loss': rank_loss})
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_loss': best_test_loss}, test_loss < best_test_loss, filepath=save_path)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            print("best test epoch is {}".format(epoch))

    csv_logger.write_text('Best loss is {}'.format(best_test_loss))

if __name__ == '__main__':
    main()
