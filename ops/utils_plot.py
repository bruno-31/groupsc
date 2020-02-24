import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
from ops.im2col import *
from ops.utils import get_mask

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


def hist_tensor(img,**kwargs):
    inp_shape = tuple(img.shape)
    print(inp_shape)
    img_np = torch_to_np(img)
    return plt.hist(img_np.flatten(),**kwargs)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

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



def show_dict(m,a=None, norm_grid=False, sort_freq=True, norm=True):
    n_elem,_,s = m.shape
    s_ = int(math.sqrt(s))
    m=m.view(n_elem,1,s_,s_)
    if norm:
        m = normalize_patches(m)
    if sort_freq:
        if a is None:
            raise ValueError("provide code array to sort dicts by usage frequency")
        idx = sort_patches(a)
        m = m[idx]

    grid = make_grid(m, normalize=norm_grid, padding=2,nrow=int(math.sqrt(n_elem)))
    return grid

def whiten_col(tx,eps=1e-4):
    shape = tx.shape
    tx = tx.squeeze()
    D = torch.mm(tx, tx.t()) / len(tx)
    diag, v = torch.symeig(D, eigenvectors=True)
    diag[diag < eps] = 1
    diag = diag ** 0.5
    diag = 1 / diag
    S = torch.diag(diag)
    out = v @ S @ v.t() @ tx
    out = out.view(shape)
    return out

def normalize_patches(D):
    p=3.5
    M=D.max()
    m=D.min()
    if m>=0:
        me = 0
    else:
        me = D.mean()
    sig = torch.sqrt(((D-me)**2).mean())
    D=torch.min(torch.max(D, -p*sig),p*sig)
    M=D.max()
    m=D.min()
    D = (D-m)/(M-m)
    return D

def sort_patches(a):
    code = get_mask(a).float()
    code_freq = code.mean([0, 2, 3]).flatten()
    _, idx = code_freq.sort(descending=True)
    return idx
