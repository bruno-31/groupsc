import torch
import torch.nn as nn
from ops.im2col import Im2Col, Col2Im
from collections import namedtuple
from ops.utils import soft_threshold


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings','threshold',
                                         'multi_lmbda','sigma_min','sigma_max','n_sigma'])

class Lista(nn.Module):
    '''
    Tied lista with coupling
    '''

    def __init__(self, params: ListaParams):
        super(Lista, self).__init__()

        print('random init of weights ')
        D = torch.randn(params.kernel_size ** 2*3, params.num_filters)

        dtd = D.t() @ D
        _, s, _ = dtd.svd()
        l = torch.max(s)
        D /= torch.sqrt(l)
        A = D.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        B = torch.clone(A.transpose(0, 1))
        W = torch.clone(A.transpose(0, 1))

        self.apply_A = torch.nn.Conv2d(params.kernel_size ** 2*3, params.num_filters, kernel_size=1, bias=False)
        self.apply_D = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2*3, kernel_size=1, bias=False)
        self.apply_W = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2*3, kernel_size=1, bias=False)

        self.apply_A.weight.data = A
        self.apply_D.weight.data = B
        self.apply_W.weight.data = W

        self.params = params

        print("centered patches")

        if params.multi_lmbda:
            self.lmbda = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(params.n_sigma*params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.lmbda]
        else:
            self.lmbda = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(params.n_sigma)])
            [nn.init.constant_(x, params.threshold) for x in self.lmbda]
        
        print([n for n,_ in self.named_parameters()])
        self.soft_threshold = soft_threshold

        d = (params.sigma_max-params.sigma_min)/params.n_sigma
        self.arr = [params.sigma_min + d*i for i in range(params.n_sigma+1)]

    def lookup_arr(self,value):
        lookup = [value <= x for x in self.arr]
        for i,val in enumerate(lookup):
            if val:
                return max(i-1,0)
        return i-1
                

    def forward(self, I, sigma_hat):
        
        params = self.params
        thresh_fn = self.soft_threshold
        noise_set = self.lookup_arr(sigma_hat)
        I_size = I.shape
        I_col = Im2Col(I, kernel_size=params.kernel_size, stride=params.stride, padding=0, tensorized=True)

        mean_patch = I_col.mean(dim=1, keepdim=True)
        I_col = I_col - mean_patch
        lin_input = self.apply_A(I_col)
        # lmbda_ = self.lmbda[noise_set][0] if params.multi_lmbda else self.lmbda[noise_set]
        lmbda_ = self.lmbda[noise_set*params.unfoldings] if params.multi_lmbda else self.lmbda[noise_set]
        gamma_k = thresh_fn(lin_input, lmbda_)

        for k in range(params.unfoldings - 1):
            x_k = self.apply_D(gamma_k)
            res = x_k - I_col
            r_k = self.apply_A(res)
            # lmbda_ = self.lmbda[noise_set][k] if params.multi_lmbda else self.lmbda[noise_set]
            lmbda_ = self.lmbda[noise_set*params.unfoldings + k+1] if params.multi_lmbda else self.lmbda[noise_set]
            gamma_k = thresh_fn(gamma_k - r_k, lmbda_)

        output_all = self.apply_W(gamma_k)
        output_all = output_all + mean_patch
        output = Col2Im(output_all, I_size[2:], kernel_size=params.kernel_size, stride=params.stride, padding=0,
                        avg=True, input_tensorized=True)

        return output



