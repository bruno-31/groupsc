from torchvision import transforms
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import random
from typing import Sequence

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class DenoisingDataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False):
        """
        Args:
            root_dirs (string): A list of directories with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('tif','png','jpg','jpeg','bmp'))]
        self.verbose = verbose

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        image = Image.open(img_name).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image


class ColorDenoisingDataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False, ycbcr=False):
        """
        Args:
            root_dirs (string): A list of directories with all the images' folders.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('tif','png','jpg','jpeg','bmp'))]
        self.verbose = verbose
        self.ycbcr = ycbcr

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]
        if self.ycbcr:
            image = Image.open(img_name).convert('Ycbcr')
        else:
            image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image


def get_dataloaders(train_path_list, test_path_list, crop_size=128, batch_size=1,downscale=0,drop_last=True,concat=True):
    batch_sizes = {'train': batch_size, 'test':1}
    tfs = []
    if downscale==0:
        tfs = [transforms.RandomCrop(crop_size)]
    elif downscale==1:
        tfs += [transforms.RandomResizedCrop(crop_size,(2.0,2.01))]
    elif downscale==2:
        tfs += [transforms.RandomResizedCrop(crop_size,(1.0,2.01))]

    tfs += [transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()]

    train_transforms = transforms.Compose(tfs)
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}

    if concat:
        image_datasets = {'train': torch.utils.data.ConcatDataset([DenoisingDataset(train_path_list, data_transforms['train']) for _ in range(batch_sizes['train'])]), # constant num of iter per epoch not depending on batch size
                          'test': DenoisingDataset(test_path_list, data_transforms['test'])}
    else:
        image_datasets = {'train': DenoisingDataset(train_path_list, data_transforms['train']),
                          'test': DenoisingDataset(test_path_list, data_transforms['test'])}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], drop_last=drop_last, shuffle=(x == 'train')) for x in ['train', 'test']}
    return dataloaders


def get_color_dataloaders(train_path_list, test_path_list, crop_size=128, batch_size=1,downscale=0,drop_last=True,concat=True):

    batch_sizes = {'train': batch_size, 'test':1}
    tfs = []
    if downscale==0:
        tfs = [transforms.RandomCrop(crop_size)]
    elif downscale==1:
        tfs += [transforms.RandomResizedCrop(crop_size,(2.0,2.01))]
    elif downscale==2:
        tfs += [transforms.RandomResizedCrop(crop_size,(1.0,2.01))]

    tfs += [transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()]

    train_transforms = transforms.Compose(tfs)
    test_transforms = transforms.Compose([
        transforms.ToTensor()])

    data_transforms = {'train': train_transforms,
                       'test': test_transforms}

    if concat:
        image_datasets = {'train': torch.utils.data.ConcatDataset([ColorDenoisingDataset(train_path_list, data_transforms['train']) for _ in range(batch_sizes['train'])]), # constant num of iter per epoch not depending on batch size
                          'test': ColorDenoisingDataset(test_path_list, data_transforms['test'])}
    else:
        image_datasets = {'train': ColorDenoisingDataset(train_path_list, data_transforms['train']),
                          'test': ColorDenoisingDataset(test_path_list, data_transforms['test'])}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x], drop_last=drop_last, shuffle=(x == 'train')) for x in ['train', 'test']}
    return dataloaders

