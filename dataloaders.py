from torchvision import transforms
from torch.utils.data import Dataset
from os import listdir, path
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import random
from typing import Sequence
from itertools import repeat

def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Dataset(Dataset):
    def __init__(self, root_dirs, transform=None, verbose=False, grey=False):
        self.root_dirs = root_dirs
        self.transform = transform
        self.images_path = []
        for cur_path in root_dirs:
            self.images_path += [path.join(cur_path, file) for file in listdir(cur_path) if file.endswith(('tif','png','jpg','jpeg','bmp'))]
        self.verbose = verbose
        self.grey = grey

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_name = self.images_path[idx]

        if self.grey:
            image = Image.open(img_name).convert('L')
        else:
            image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.verbose:
            return image, img_name.split('/')[-1]

        return image


def get_dataloaders(train_path_list, test_path_list, val_path_list, crop_size=128, batch_size=1, downscale=0,
                    drop_last=True, concat=True, n_worker=0, scale_min=0.001, scale_max=0.1, verbose=False, grey=False):

    batch_sizes = {'train': batch_size, 'test':1, 'val': 1}
    tfs = []
    if downscale==0:
        tfs = [transforms.RandomCrop(crop_size)]
    elif downscale==1:
        tfs += [transforms.RandomResizedCrop(crop_size, scale=(scale_min,scale_max), ratio=(1.0,1.0))]
    elif downscale==2:
        print('mode 2')
        tfs += [transforms.Resize(300)]
        tfs += [transforms.RandomCrop(crop_size)]

    tfs += [transforms.RandomCrop(crop_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()]

    train_transforms = transforms.Compose(tfs)
    test_transforms = transforms.Compose([transforms.ToTensor()])

    data_transforms = {'train': train_transforms, 'test': test_transforms, 'val': test_transforms}

    if concat:
        train = torch.utils.data.ConcatDataset(
            [Dataset(train_path_list, data_transforms['train'], verbose=verbose, grey=grey) for _ in range(batch_sizes['train'])])
    else:
        train = Dataset(train_path_list, data_transforms['train'], verbose=verbose, grey=grey)

    image_datasets = {'train': train,
                      'test': Dataset(test_path_list, data_transforms['test'], verbose=verbose, grey=grey),
                      'val': Dataset(val_path_list, data_transforms['test'], verbose=verbose, grey=grey)}

    if len(val_path_list) == 0 or len(train_path_list) == 0:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                      num_workers=n_worker, drop_last=drop_last, shuffle=(x == 'train'))
                       for x in ['test']}
    else:
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_sizes[x],
                                                  num_workers=n_worker,drop_last=drop_last, shuffle=(x == 'train')) for x in ['train', 'test', 'val']}
    return dataloaders
