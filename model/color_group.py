import torch
import torch.nn as nn
from collections import namedtuple
from ops_nonlocal.matching import SoftMatchingLayer
from ops.im2col import Im2Col, Col2Im
import torch.nn.functional as F
from ops.utils import gen_mask_windows, gen_quadra_mask_windows

ListaParams = namedtuple('ListaParams', ['kernel_size', 'num_filters', 'stride', 'unfoldings',
                                         'freq', 'corr_update', 'lmbda_init', 'h', 'spams',
                                         'multi_lmbda','center_windows', 'std_gamma', 'std_y',
                                         'block_size','nu_init','mask', 'multi_std'])

NUM_CHANNEL = 3

def group_prox(A, theta, mask_sim):
    '''
    :param theta: (1,p,1,1)
    :param A: (b,p,h,w)
    :param mask_sim:(b,1,N_patch,N_patch)
    :return: prox(A)
    '''
    A_shape = A.shape
    A = A.flatten(2,3)
    support_sim = (mask_sim.abs().sum(2, keepdim=True) + 1e-12)**0.5
    norm_2_group = ((torch.matmul(A.pow(2),mask_sim.squeeze(1)) +1e-8)**0.5).unsqueeze(2)
    A = A.unsqueeze(2)
    out = F.relu(1-theta * support_sim / (norm_2_group))*A
    return out.view(A_shape)


def norm_group(A,mask_sim, theta):
    '''
    :param A: (b,p,N_group,N_patch)
    :param mask_sim:(b,1,N_group,N_patch)
    :return: norm(A)
    '''
    A = A.flatten(2, 3).unsqueeze(2)
    support_sim = (mask_sim.abs().sum(2, keepdim=True) + 1e-12)**0.5
    norm_2 = ((torch.matmul(A.pow(2), mask_sim) + 1e-10) ** 0.5) / support_sim
    norm_1_2 = (norm_2*theta).sum(1, keepdim=True)
    return  norm_1_2



class groupLista(nn.Module):

    def __init__(self, params: ListaParams):
        super(groupLista, self).__init__()

        if params.spams:
            str_spams = f'./datasets/dictionnaries/c{params.num_filters}_{params.kernel_size}x{params.kernel_size}.pt'
            print(f'loading spams dict @ {str_spams}')
            try:
                D = torch.load(str_spams).t()
            except:
                print('no spams dict found for this set of parameters')
        else:
            print('random init of weights ')
            D = torch.randn(params.kernel_size ** 2*NUM_CHANNEL, params.num_filters)

        dtd = D.t() @ D
        _, s, _ = dtd.svd()
        l = torch.max(s)
        D /= torch.sqrt(l)
        A = D.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
        W = torch.clone(A.transpose(0, 1))
        B = torch.clone(A.transpose(0, 1))

        self.apply_A = torch.nn.Conv2d(params.kernel_size ** 2 *NUM_CHANNEL, params.num_filters, kernel_size=1, bias=False)
        self.apply_D = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2*NUM_CHANNEL, kernel_size=1, bias=False)
        self.apply_W = torch.nn.Conv2d(params.num_filters, params.kernel_size ** 2*NUM_CHANNEL, kernel_size=1, bias=False)

        self.apply_A.weight.data = A
        self.apply_D.weight.data = B
        self.apply_W.weight.data = W

        self.params = params
        num_lista = (params.unfoldings // (params.freq+1) +1) if params.multi_std else 1
        self.num_lista = num_lista

        self.simLayer = nn.ModuleList([SoftMatchingLayer() for _ in range(num_lista)])
        # =====

        if params.std_gamma:
            self.std_g = nn.ParameterList([nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(num_lista-1)])
            [nn.init.constant_(elem, 1.) for elem in self.std_g]

        if params.std_y:
            self.std = nn.ParameterList([nn.Parameter(torch.zeros(1, params.kernel_size ** 2 *NUM_CHANNEL, 1, 1)) for _ in range(num_lista)])
            [nn.init.constant_(elem, 1 / params.h)  for elem in self.std]
        else:
            self.std = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, 1, 1)) for _ in range(num_lista)])
            [nn.init.constant_(elem, 1 / params.h)  for elem in self.std]

        [nn.init.constant_(elem, 1 / (0.5 * params.h))  for elem in self.std[1:]] #smaller constant for next

        self.nu = nn.ParameterList([nn.Parameter(torch.Tensor(1, )) for _ in range(num_lista)])  # update correlation map
        [nn.init.constant_(elem,params.nu_init) for elem in self.nu]

        #  =====

        if params.multi_lmbda:
            self.lmbda = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.lmbda_init) for x in self.lmbda]
        else:
            self.lmbda = nn.Parameter(torch.zeros(1, params.num_filters, 1, 1))
            nn.init.constant_(self.lmbda, params.lmbda_init)

        self.threshold = group_prox

        if params.center_windows:
            n_mask = params.block_size - (params.kernel_size//2)*2

            if params.mask == 1:
                mask_window_blocks = gen_mask_windows(n_mask,n_mask).contiguous().view(
                    1,n_mask**2,n_mask**2)
                mask_window_blocks[mask_window_blocks==0]= 1e10
                mask_window_blocks[mask_window_blocks==1]=0
            elif params.mask ==2:
                cut_dist = n_mask-5
                mask_window_blocks = gen_quadra_mask_windows(n_mask,n_mask,cut_dist,cut_dist).contiguous().view(
                    1,n_mask**2,n_mask**2)
                mask_window_blocks = (1-mask_window_blocks)*1e3
            else:
                raise NotImplementedError

            [module.set_mask(mask_window_blocks) for module in self.simLayer]


    def forward(self, I, I_clean=None, writer=None, epoch=None):
        params = self.params
        thresh_fn = self.threshold
        I_size = I.shape
        I_col = Im2Col(I,kernel_size=params.kernel_size,stride=params.stride,padding=0,tensorized=True)
        N = I_col.shape[2] * I_col.shape[3]
        b,n,h,w = I_col.shape

        mean_patch = I_col.mean(dim=1, keepdim=True)
        I_col -= mean_patch
        counter_lista = 0

        col = I_col * self.std[counter_lista]

        similarity_map = self.simLayer[counter_lista](col).flatten(2,3).view(b,1,N,N)

        lin = self.apply_A(I_col)

        lmbda_ = self.lmbda[0] if params.multi_lmbda else self.lmbda
        gamma_k = thresh_fn(lin, lmbda_, similarity_map)

        num_unfoldings = params.unfoldings

        for k in range(num_unfoldings - 1):
            if (k) % params.freq == 0 and k != 0:  # freq update correlation map
                if params.multi_std:
                    counter_lista += 1

                # gamma_k_corr_update = (gamma_k * self.std_g) if (params.std_gamma) else gamma_k
                # gamma_k_corr_update =  gamma_k
                gamma_k_corr_update = (gamma_k * self.std_g[counter_lista-1]) if (params.std_gamma) else gamma_k

                if params.corr_update == 2:
                    patch_estimates = self.apply_W(gamma_k_corr_update)

                elif params.corr_update == 3:
                    patch_estimates = self.apply_W(gamma_k_corr_update) + mean_patch
                    int_output = Col2Im(patch_estimates, I_size[2:], kernel_size=params.kernel_size, stride=params.stride,padding=0, avg=True, input_tensorized=True)
                    int_I_col = Im2Col(int_output,kernel_size=params.kernel_size,stride=params.stride,padding=0, tensorized=True)
                    int_I_col -= mean_patch
                    patch_estimates = int_I_col

                col = patch_estimates * self.std[counter_lista]

                similarity_map_new = self.simLayer[counter_lista](col).flatten(2, 3).view(b, 1, N, N)
                nu_sigmoid = torch.sigmoid(self.nu[counter_lista-1])
                similarity_map = (1-nu_sigmoid) * similarity_map + (nu_sigmoid) * similarity_map_new

            x_k = self.apply_D(gamma_k)
            res = I_col - x_k
            r_k = self.apply_A(res)

            lmbda_ = self.lmbda[k+1] if params.multi_lmbda else self.lmbda
            gamma_k = thresh_fn(gamma_k + r_k, lmbda_,similarity_map)

        output_all = self.apply_W(gamma_k)
        output_all += mean_patch
        output = Col2Im(output_all,I_size[2:],kernel_size=params.kernel_size,stride=params.stride,padding=0, avg=True,input_tensorized=True)

        return output

