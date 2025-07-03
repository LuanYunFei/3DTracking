import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
def train(e_model, d_model, train_loader, epoch, d_optimizer, device,
                           dtype, log_interval, w1):
        e_model.eval()
        d_model.train()
        train_loss = 0

        for batch_idx, (image_name, d_image, d_label) in enumerate(tqdm(train_loader)):
            d_image = d_image.to(device=device, dtype=dtype)
            d_label = d_label.to(device=device, dtype=dtype)

            with torch.no_grad():
                feature = e_model(d_image)

            d_pred = d_model(d_image,feature)

            # # # # Mse LOSS
            mse_loss_func = nn.MSELoss()
            mse_loss = mse_loss_func(d_pred,d_label)

            # # # # Ssim LOSS
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
            ssim_loss = 1 - ms_ssim(d_pred, d_label)
            ssim_loss *= w1

            d_optimizer.zero_grad()
            loss = ssim_loss + mse_loss
            loss.backward()
            d_optimizer.step()
            train_loss = train_loss + loss.item()
            if batch_idx % log_interval == 0:
                tqdm.write(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}. '
                    'Train Loss: {:.2f}. '.format(epoch, batch_idx, len(train_loader),
                                                  100. * batch_idx / len(train_loader), loss.item(),
                                                  loss.item()
                                                  ))

        train_loss = train_loss / len(train_loader)
        return train_loss

def test(e_model, d_model, val_loader, epoch, device, dtype, w1):
    e_model.eval()
    d_model.eval()
    test_loss = 0
    mean_ssim_loss = 0
    mean_mse_loss = 0

    for batch_idx, (image_name, d_image, d_label) in enumerate(tqdm(val_loader)):
        d_image = d_image.to(device=device, dtype=dtype)
        d_label = d_label.to(device=device, dtype=dtype)

        with torch.no_grad():

            feature= e_model(d_image)
            d_pred = d_model(d_image,feature)

            # # # # Mse LOSS
            mse_loss_func = nn.MSELoss()
            mse_loss = mse_loss_func(d_pred, d_label)

            # # # # Ssim LOSS
            ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
            ssim_loss = 1 - ms_ssim(d_pred, d_label)
            ssim_loss *= w1

            loss = mse_loss + ssim_loss
            test_loss += loss.item()  # sum up batch loss
            mean_ssim_loss += ssim_loss.item()
            mean_mse_loss += mse_loss.item()

    test_loss /= len(val_loader)
    mean_ssim_loss /= len(val_loader)
    mean_mse_loss /= len(val_loader)

    tqdm.write(
        '\nTest set: epoch: {} Average loss: {:.4f} '.format(epoch, test_loss))
    return test_loss, mean_ssim_loss , mean_mse_loss

