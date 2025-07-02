import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

import os
from tqdm import tqdm
from loss import RnCLoss, BloLoss

def train(model, loader, epoch, optimizer, device, dtype, log_interval, k1, k2, alpha):
    model.train()
    train_loss = 0

    for batch_idx, (image_name, data, target) in enumerate(tqdm(loader)):
        batch_size = target.shape[0]
        target = torch.unsqueeze(target,1)  ## bs * 1
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)
        feature, output = model(data)   ## bs * class_num * representation_size  ## bs * class_num
        features = torch.cat([feature.unsqueeze(1), feature.unsqueeze(1)], dim=1)  ## bs * 2 * representation_size
        class_num = output.shape[1]
        label = torch.div(torch.arange(0, class_num, 1), class_num - 1)  ## class_num-1
        label = label.to(device=device)

        # one-hot label
        target_distribution = torch.zeros(size=[batch_size, class_num])
        for item_idx in range(batch_size):
            defocus_order = round(float(target[item_idx][0] * (class_num - 1)))
            target_distribution[item_idx][defocus_order] = 1
        target_distribution = target_distribution.to(device=device)

        # kl loss
        log_output_distribution = F.log_softmax(output, dim = 1)
        kl_func = nn.KLDivLoss(reduction="batchmean")
        kl_loss = kl_func(log_output_distribution, target_distribution)

        # order loss (BLO)
        blo_loss = BloLoss(alpha).to(device=device, dtype=dtype)
        order_loss = k1 * blo_loss(log_output_distribution, target, label)

        # rank loss (RNC)
        rnc_loss = RnCLoss().to(device=device, dtype=dtype)
        rank_loss = k2 * rnc_loss(features, target)

        optimizer.zero_grad()
        loss = kl_loss + order_loss + rank_loss
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        if batch_idx % log_interval == 0:
            tqdm.write(
                'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                'Train Loss: {:.2f}. '.format(epoch, batch_idx, len(loader),
                                                           100. * batch_idx / len(loader), loss.item(),
                                                           loss.item()
                                                           ))
    train_loss = train_loss / len(loader)
    return train_loss


def test(model, loader, epoch, device, dtype, k1, k2, alpha):
    model.eval()

    test_loss = 0
    mean_kl_loss = 0
    mean_order_loss = 0
    mean_rank_loss = 0

    for batch_idx, (image_name, data, target) in enumerate(tqdm(loader)):
        batch_size = target.shape[0]
        target = torch.unsqueeze(target, 1)  ## bs * 1
        data, target = data.to(device=device, dtype=dtype), target.to(device=device)

        with torch.no_grad():
            feature, output = model(data)
            features = torch.cat([feature.unsqueeze(1), feature.unsqueeze(1)], dim=1)  ## bs * 2 * representation_size
            class_num = output.shape[1]
            label = torch.div(torch.arange(0, class_num, 1), class_num - 1)  ## class_num-1
            label = label.to(device=device)

            # one-hot
            target_distribution = torch.zeros(size=[batch_size, class_num])
            for item_idx in range(batch_size):
                defocus_order = round(float(target[item_idx][0] * (class_num - 1)))
                print (defocus_order, target[item_idx][0])
                target_distribution[item_idx][defocus_order] = 1
            target_distribution = target_distribution.to(device=device)

            # kl loss
            log_output_distribution = F.log_softmax(output, dim=1)
            kl_func = nn.KLDivLoss(reduction="batchmean")
            kl_loss = kl_func(log_output_distribution, target_distribution)

            # order loss (BLO)
            blo_loss = BloLoss(alpha).to(device=device, dtype=dtype)
            order_loss = k1 * blo_loss(log_output_distribution, target, label)

            # rank loss (RNC)
            rnc_loss = RnCLoss().to(device=device, dtype=dtype)
            rank_loss = k2 * rnc_loss(features, target)

            loss = kl_loss + order_loss + rank_loss
            test_loss += loss.item()
            mean_kl_loss += kl_loss.item()
            mean_order_loss += order_loss.item()
            mean_rank_loss += rank_loss.item()

    test_loss /= len(loader)
    mean_kl_loss /= len(loader)
    mean_order_loss /= len(loader)
    mean_rank_loss /= len(loader)

    tqdm.write(
        '\nTest set: Epoch: {} Average loss: {:.4f} '.format(epoch, test_loss))
    return test_loss,mean_kl_loss,mean_order_loss, mean_rank_loss

def save_checkpoint(state, is_best, filepath='./'):
    best_path = os.path.join(filepath, 'model_best_epoch{}.pth.tar'.format(state['epoch']))
    if is_best:
        torch.save(state, best_path)



