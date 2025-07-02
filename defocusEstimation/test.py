from tqdm import tqdm
import numpy as np
import math
import argparse
from model import MobileNet2
from data import my_loader
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='MobileNetv2 testing with PyTorch')

# ====================================for basic config=====================================
parser.add_argument('--gpus', default='0', help='List of GPUs used for training - e.g 0,1,3')
parser.add_argument('-j', '--workers', default=4, type=int, help='Number of data loading workers')

# ========================================for data=========================================
parser.add_argument('--test_dir',  required=True, type=str, help='Path to test folder')
parser.add_argument('--class_num',  default=41, type=int, help='The number of defocus levels')
parser.add_argument('--type', default='float32', help='Type of tensor: float32, float16, float64.')

# =======================================for model==========================================
parser.add_argument('--resume', required=True, type=str, help='Path to checkpoint')
parser.add_argument('--scaling', type=float, default=1, help='Scaling of MobileNet.')
parser.add_argument('--input-size', type=int, default=32, help='Input size of MobileNet, multiple of 32.')
def main():
    args = parser.parse_args()

    # ===========================training device===================================================
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
    print ("load model")
    net = MobileNet2(input_size=args.input_size, scale=args.scaling, num_classes=args.class_num)
    if args.gpus is not None:
        net = torch.nn.DataParallel(net, args.gpus)
    net.to(device=device, dtype=dtype)
    checkpoint = torch.load(args.resume, map_location=device)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    print("load model done")

    # ===========================data loading========================================================
    test_data = my_loader(data_path= args.test_dir, class_num= args.class_num, augment=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=False)

    # ============================evaluation=========================================================
    error_tolerance = 1 / (args.class_num - 1)
    MAE_loss = 0
    Z_Acc = 0
    MAE_loss_list = []

    with torch.no_grad():
        for batch_idx, (image_name,image,target) in enumerate(tqdm(test_loader)):
            target = target.numpy()[0]
            image = image.to(device=device, dtype=dtype)
            feature, pred = net(image)

            output_prob = F.softmax(pred, dim=1)  ## 1 * class_num
            label = torch.div(torch.arange(0, args.class_num, 1), args.class_num - 1)  ## class_num
            label = label.to(device=device)
            output_value = torch.sum(output_prob[0] * label)
            pred = output_value.cpu().detach().numpy()

            MAE_loss += abs(pred - target)
            MAE_loss_list.append(abs(pred - target))
            Z_Acc += 1 if (abs(pred - target) < error_tolerance) else 0

    MAE_loss = MAE_loss / len(MAE_loss_list)
    Z_Acc = Z_Acc / len(MAE_loss_list)
    MAE_loss_list_square = list(map(lambda x:math.pow(x - MAE_loss,2),MAE_loss_list))
    std_error = np.sqrt(sum(MAE_loss_list_square) / len(MAE_loss_list_square))

    print ("MAE_loss",MAE_loss)
    print ("Z_Acc",Z_Acc)
    print ("std_error",std_error)

if __name__ == '__main__':
    main()
