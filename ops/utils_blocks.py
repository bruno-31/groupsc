import torch
import torch.nn.functional as F
from ops.im2col import Im2Col, Col2Im


def shape_pad_even(tensor_shape, patch,stride):
    assert len(tensor_shape) == 4
    b,c,h,w = tensor_shape
    required_pad_h = stride - (h-patch) % stride
    required_pad_w = stride - (w-patch) % stride
    return required_pad_h,required_pad_w


class block_module():

    def __init__(self,block_size,block_stride, kernel_size, params):
        super(block_module).__init__()
        self.params = params
        self.kernel_size = kernel_size
        self.block_size = block_size
        self.block_stride = block_stride
        # self.channel_size = channel_size

    def _make_blocks(self, image, return_padded=False):
        '''
        :param image: (1,C,H,W)
        :return: raw block (batch,C,block_size,block_size), tulple shape augmented image
        '''
        params = self.params

        self.channel_size = image.shape[1]

        if params['pad_block']:
            pad = (self.block_size - 1,) * 4
        elif params['pad_patch']:
            pad = (self.kernel_size,)*4
        elif params['no_pad']:
            pad = (0,) * 4
        elif params['custom_pad'] is not None:
            pad = (params['custom_pad'],) * 4

        else:
            raise NotImplementedError

        image_mirror_padded = F.pad(image, pad, mode='reflect')
        pad_even = shape_pad_even(image_mirror_padded.shape,  self.block_size, self.block_stride)
        pad_h, pad_w =  pad_even
        if params['centered_pad']:
            pad_ = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
        else:
            pad_ =(0, pad_w, 0, pad_h)
        pad = tuple([x+y for x,y in zip(pad,pad_)])
        self.pad = pad

        image_mirror_padded_even = F.pad(image, pad, mode='reflect')  # add half kernel cause block edges are dump

        self.augmented_shape = image_mirror_padded_even.shape

        if return_padded:
            return  image_mirror_padded

        batch_blocks = Im2Col(image_mirror_padded_even,
                                kernel_size=self.block_size,
                                stride= self.block_stride,
                                padding=0)

        batch_blocks = batch_blocks.permute(2, 0, 1)
        batch_blocks = batch_blocks.view(-1, self.channel_size, self.block_size, self.block_size)
        return batch_blocks

    def _agregate_blocks(self,batch_out_blocks):
        '''
        :param blocks: processed blocks
        :return: image of averaged estimates
        '''
        h_pad, w_pad = self.augmented_shape[2:]
        params = self.params
        l = self.kernel_size // 2
        device = batch_out_blocks.device

        # batch_out_blocks_flatten = batch_out_blocks.flatten(2, 3).permute(1, 2, 0)
        batch_out_blocks_flatten = batch_out_blocks.view(-1,self.channel_size * self.block_size**2).transpose(0,1).unsqueeze(0)

        if params['ponderate_out_blocks']:
            mask = F.conv_transpose2d(torch.ones((1,1)+(self.block_size - 2 * l,)*2),
                                      torch.ones((1,1)+(self.kernel_size,)*2))
            mask = mask.to(device=device)
            batch_out_blocks *= mask

            # batch_out_blocks_flatten = batch_out_blocks.flatten(2, 3).permute(1, 2, 0)

            output_padded = Col2Im(batch_out_blocks_flatten,
                                   output_size=(h_pad, w_pad),
                                   kernel_size=self.block_size,
                                   stride=self.block_stride,
                                   padding=0,
                                   avg=False)

            batch_out_blocks_ones = torch.ones_like(batch_out_blocks) * mask
            # batch_out_blocks_flatten_ones = batch_out_blocks_ones.flatten(2, 3).permute(1, 2, 0)
            batch_out_blocks_flatten_ones = batch_out_blocks_ones.view(-1, self.channel_size * self.block_size ** 2).transpose(0,1).unsqueeze(0)

            if params['avg']:
                mask_ = Col2Im(batch_out_blocks_flatten_ones,
                               output_size=(h_pad, w_pad),
                               kernel_size=self.block_size,
                               stride=self.block_stride,
                               padding=0,
                               avg=False)
                output_padded /= mask_

        elif params['crop_out_blocks']:
            kernel_ = self.block_size - 2 * l
            # batch_out_blocks_flatten = batch_out_blocks.flatten(2, 3).permute(1, 2, 0)
            output_padded = Col2Im(batch_out_blocks_flatten,
                                   output_size=(h_pad - 2 * l, w_pad - 2 * l),
                                   kernel_size=kernel_,
                                   stride=self.block_size,
                                   padding=0,
                                   avg=params['avg'])

        elif params['sum_blocks']:
            # batch_out_blocks_flatten = batch_out_blocks.flatten(2, 3).permute(1, 2, 0)
            output_padded = Col2Im(batch_out_blocks_flatten,
                                   output_size=(h_pad, w_pad),
                                   kernel_size=self.block_size,
                                   stride=self.block_stride,
                                   padding=0,
                                   avg=params['avg'])
        else:
            raise NotImplementedError

        pad = self.pad
        output = output_padded[:, :, pad[2]:-pad[3], pad[0]:-pad[1]]

        return output

