import torch
import torch.functional as F
from random import randint
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def gen_bayer_mask(h,w):
    x = torch.zeros(1, 3, h, w)

    x[:, 0, 1::2, 1::2] = 1  # r
    x[:, 1, ::2, 1::2] = 1
    x[:, 1, 1::2, ::2] = 1  # g
    x[:, 2, ::2, ::2] = 1  # b

    return x

def togray(tensor):
    b, c, h, w = tensor.shape
    tensor = tensor.view(b, 3, -1, h, w)
    tensor = tensor.sum(1)
    return tensor

def torch_to_np(img_var):
    return img_var.detach().cpu().numpy()

def plot_tensor(img, **kwargs):
    inp_shape = tuple(img.shape)
    print(inp_shape)
    img_np = torch_to_np(img)
    if inp_shape[1]==3:
        img_np_ = img_np.transpose([1,2,0])
        plt.imshow(img_np_)

    elif inp_shape[1]==1:
        img_np_ = np.squeeze(img_np)
        plt.imshow(img_np_, **kwargs)

    else:
        # raise NotImplementedError
        plt.imshow(img_np, **kwargs)
    plt.axis('off')


def get_mask(A):
    mask = A.clone().detach()
    mask[A != 0] = 1
    return mask.byte()

def sparsity(A):
    return get_mask(A).sum().item()/A.numel()

def soft_threshold(x, lambd):
    return nn.functional.relu(x - lambd) - nn.functional.relu(-x - lambd)

def fastSoftThrs(x, lmbda):
    return x + 0.5 * (torch.abs(x-torch.abs(lmbda))-torch.abs(x+torch.abs(lmbda)))

def save_checkpoint(state,ckpt_path):
    torch.save(state, ckpt_path)

def generate_key():
    return '{}'.format(randint(0, 100000))

def show_mem():
    mem = torch.cuda.memory_allocated() * 1e-6
    max_mem = torch.cuda.max_memory_allocated() * 1e-6
    return mem, max_mem

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def step_lr(optimizer, lr_decay):
    lr = optimizer.param_groups[0]['lr']
    optimizer.param_groups[0]['lr'] = lr * lr_decay

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def gen_mask_windows(h, w):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = 1

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]


def gen_linear_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k)
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def gen_quadra_mask_windows(h, w, h_,w_):
    '''
    return mask for block window
    :param h:
    :param w:
    :return: (h,w,h,w)
    '''

    x = torch.ones(1, 1, h - h_ + 1, w - w_ + 1)
    k = torch.ones(1, 1, h_, w_)
    kernel = F.conv_transpose2d(x, k) **2
    kernel /= kernel.max()
    mask = torch.zeros(2 * h, 2 * w, h, w)
    for i in range(h):
        for j in range(w):
            mask[i:i + h, j:j + w, i, j] = kernel

    return mask[h // 2:-h // 2, w // 2:-w // 2, :, :]

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

