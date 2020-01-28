import torch
import torch.nn as nn


class SoftMatchingLayer(nn.Module):
    ''' Generate gaussian similarity matrix from feature map '''

    def __init__(self,temperature=None):
        super(SoftMatchingLayer, self).__init__()
        self.mask_ = False

    def set_mask(self, mask):
        self.register_buffer('mask', mask)
        self.mask_ = True

    def forward(self, x,y=None):
        assert len(x.shape) == 4
        if y is not None:
            assert x.shape == y.shape

        x_flat = x.flatten(2,3)
        x_norm = x_flat.pow(2).sum(1,keepdim=True)

        if y is not None:
            y_flat = y.flatten(2, 3)
            y_norm = y_flat.pow(2).sum(1)
        else:
            y_flat = x_flat
            y_norm = x_norm

        dint = torch.matmul(x_flat.transpose(1,2),y_flat)

        dist = (x_norm + y_norm.transpose(1,2) - 2.0 * dint).clamp(0.)

        if self.mask_:
            dist = dist + self.mask

        f = torch.exp(-dist)

        return f.view((x.shape[0],)+(x.shape[2]*x.shape[2],)+x.shape[2:])



