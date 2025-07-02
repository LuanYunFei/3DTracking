import os
import glob
from PIL import Image
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torchvision import transforms
from torch.utils.data import Dataset

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

class my_loader(Dataset):
    def __init__(self, data_path, class_num, augment=True):
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path,'*', '*.png'))
        self.augment = augment
        self.class_num = class_num
    def __getitem__(self, index):
        image_path = self.imgs_path[index]
        # to gray, since color is irrelevant to defocus estimation
        image = Image.open(image_path).convert("L").convert("RGB")

        if self.augment == True :
            transform_list = inception_preproccess()
            image = transform_list(image)

        else :
            transform_list = scale_crop()
            image = transform_list(image)

        # normalize the label to [0,1]
        target = float(image_path.split("/")[-2]) / (self.class_num - 1)
        target = torch.tensor(target)
        return image_path, image, target

    def __len__(self):
        return len(self.imgs_path)


def inception_preproccess(normalize=__imagenet_stats):

    return transforms.Compose([
        transforms.RandomRotation(90, expand=False),
        transforms.CenterCrop(40),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ])

def scale_crop(normalize=__imagenet_stats):
    t_list = [
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(**normalize),
    ]
    return transforms.Compose(t_list)

def get_loaders(dataroot, batch_size, workers, class_num):
    #============================================validation set=============================================
    val_data = my_loader(data_path=os.path.join(dataroot, 'val'), class_num=class_num, augment=False)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=workers,
                                             pin_memory=True)
    #============================================training set================================================
    train_data = my_loader(data_path=os.path.join(dataroot, 'train'), class_num=class_num, augment=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                               num_workers=workers, pin_memory=True)
    return train_loader, val_loader
