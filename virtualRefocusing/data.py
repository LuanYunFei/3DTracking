import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import glob
import random

class my_loader(Dataset):
    def __init__(self, data_path, augment=True):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path,'*','*.png'))
        self.augment = augment
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        image_name = image_path.split("/")[-1][:-4]
        label_path = os.path.join("/".join(image_path.split("/")[:-2]),'00', image_path.split("/")[-1])

        d_image = cv2.imread(image_path) / 255
        d_label = cv2.imread(label_path) / 255

        if self.augment == True :
            flipCode = random.choice([-1, 0, 1, 2])
            d_image = augment(d_image, flipCode)
            d_label = augment(d_label, flipCode)
            transform_list = transform()
            d_image = transform_list(d_image)
            d_label = transform_list(d_label)
        else :
            transform_list = transform()
            d_image = transform_list(d_image)
            d_label = transform_list(d_label)

        return image_name,d_image,d_label

    def __len__(self):
        return len(self.imgs_path)
def augment(image, flipCode):
    flip = cv2.flip(image, flipCode)
    return flip

def transform():
    return transforms.Compose([transforms.ToTensor()])

def get_loaders(dataroot, batch_size, workers):
    val_data = my_loader(data_path=os.path.join(dataroot, 'val'),augment=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)

    train_data = my_loader(data_path=os.path.join(dataroot, 'train'),augment=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)

    return train_loader, val_loader
