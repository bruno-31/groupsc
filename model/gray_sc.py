import torch
import torch.nn as nn
from ops.im2col import Im2Col,Col2Im
from collections import namedtuple
import torch.nn.functional as F
from ops.utils import soft_threshold,sparsity
from tqdm import tqdm


ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings','threshold', 'multi_lmbda','verbose','spams'])


class Lista(nn.Module):

    '''
    Tied lista with coupling
    '''

    def __init__(self, params: ListaParams):
        super(Lista, self).__init__()

        if params.spams:
            str_spams = f'./datasets/dictionnaries/{params.num_filters}_{params.kernel_size}x{params.kernel_size}.pt'
            print(f'loading spams dict @ {str_spams}')
            try:
                D = torch.load(str_spams).t()
            except:
                print('no spams dict found for this set of parameters')
        else:
            print('random init of weights ')
            D = torch.randn(params.kernel_size **2, params.num_filters)

        dtd = D.t() @ D
        _, s, _ = dtd.svd()
        l = torch.max(s)
        D /= torch.sqrt(l)
        A = D.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        B = torch.clone(A.transpose(0, 1))
        W = torch.clone(A.transpose(0, 1))

        self.apply_A = torch.nn.Conv2d(params.kernel_size ** 2, params.num_filters, kernel_size=1, bias=False)
        self.apply_D = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2, kernel_size=1, bias=False)
        self.apply_W = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2, kernel_size=1, bias=False)

        self.apply_A.weight.data = A
        self.apply_D.weight.data = B
        self.apply_W.weight.data = W

        self.params = params

        if params.multi_lmbda:
            self.lmbda = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.lmbda]
        else:
            self.lmbda = nn.Parameter(torch.zeros(1, params.num_filters, 1, 1))
            nn.init.constant_(self.lmbda, params.threshold)

        self.soft_threshold = soft_threshold


    def forward(self, I,writer=None,epoch=None,return_patches=False):

        params = self.params
        thresh_fn = self.soft_threshold

        I_size = I.shape
        I_col = Im2Col(I,kernel_size=params.kernel_size, stride=params.stride, padding=0,tensorized=True)

        mean_patch = I_col.mean(dim=1,keepdim=True)
        I_col = I_col - mean_patch


        lin_input = self.apply_A(I_col)

        # gamma_k = thresh_fn(lin_input, self.lmbda)

        lmbda_ = self.lmbda[0] if params.multi_lmbda else self.lmbda
        gamma_k = thresh_fn(lin_input, lmbda_)

        num_unfoldings = params.unfoldings
        N = I_col.shape[2] * I_col.shape[3]  * I_col.shape[0]


        for k in range(num_unfoldings - 1):
            x_k = self.apply_D(gamma_k)
            res = x_k - I_col
            r_k = self.apply_A(res)

            lmbda_ = self.lmbda[k+1] if params.multi_lmbda else self.lmbda
            gamma_k = thresh_fn(gamma_k - r_k, lmbda_)
            
            if params.verbose:
                residual = 0.5 * (x_k - I_col).pow(2).sum() / N
                loss =residual + (lmbda_ * gamma_k).abs().sum()/N
                tqdm.write(
                    f'patch res {residual.item():.2e} | sparsity {sparsity(gamma_k):.2f} | loss {loss.item():0.4e} ')

        output_all = self.apply_W(gamma_k)
        output_all = output_all + mean_patch
        output = Col2Im(output_all,I_size[2:],kernel_size=params.kernel_size, stride=params.stride, padding=0,avg=True,input_tensorized=True)
        if params.verbose:
            tqdm.write('')
        return output



