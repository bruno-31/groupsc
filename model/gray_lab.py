import torch
import torch.nn as nn
from ops.im2col import Im2Col, Col2Im
from collections import namedtuple
import torch.nn.functional as F
from ops.utils import soft_threshold, sparsity
from tqdm import tqdm
from ops.utils_plot import plot_tensor,plt

ListaParams = namedtuple('ListaParams',
                         ['kernel_size', 'num_filters', 'stride', 'unfoldings', 'threshold', 'multi_lmbda', 'verbose',
                          'spams'])


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
            D = torch.randn(params.kernel_size ** 2, params.num_filters)

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

        print("centered patches")

        if params.multi_lmbda:
            self.lmbda = nn.ParameterList(
                [nn.Parameter(torch.zeros(1, params.num_filters, 1, 1)) for _ in range(params.unfoldings)])
            [nn.init.constant_(x, params.threshold) for x in self.lmbda]
        else:
            self.lmbda = nn.Parameter(torch.zeros(1, params.num_filters, 1, 1))
            nn.init.constant_(self.lmbda, params.threshold)

        self.soft_threshold = soft_threshold
        
        
    ### W computation
    # W = C * pseudoinv(D)


    def forward(self, I, writer=None, epoch=None, return_patches=False):

        params = self.params
        
        ## W computation
        D = self.apply_D.weight.squeeze()
        A = self.apply_A.weight.squeeze().t()
        
        _,s1,_ = (A.t()@D).svd()
        cond1 = s1[0] / s1[-1]
        print(f'conditionement AtD {cond1:.1e}')
        _,s2,_ = (D.t()@D).svd()

        cond2 = s2[0] / s2[-1]
        print(f'conditionement DtD {cond2:.1e}')

        # eig,_ = torch.symeig(D.t()@D)
        # eig,_ = torch.symeig(A.t()@D)
        # _,s,_ = (A.t()@D).svd()
        #
        # L = torch.max(eig)
        L = 0.1
        print(L)

        A = L**0.5 * A
        D = L**0.5  * D
        # pinv = torch.pinverse(D)
        u,s,v = D.svd(some=True)
        pinv = v @ torch.diag(1/(s+1e-3)) @ u.t()
        W = A @ pinv

        if False:
            import matplotlib.pyplot as plt
            h = D.t() @ D
            _,s,_ = h.svd()
            cond = s[0]/s[-1]
            plt.subplot(121)
            plot_tensor(h.unsqueeze(0))
            plt.title(f'cond {cond:.2e}')
            plt.colorbar()
            h = D.t() @ W @ D
            _,s,_ = h.svd()
            cond = s[0]/s[-1]
            plt.subplot(122)
            plot_tensor(h.unsqueeze(0))
            plt.title(f'cond {cond:.2e}')
            plt.colorbar()
            plt.show()

            # plot Z
            from ops.utils_plot import show_dict
            W.shape
            from torchvision.utils import make_grid
            Wt = W
            m = Wt.view(-1,1,9,9)
            # plot_tensor(make_grid(m),cmap='gray')
            plot_tensor(show_dict(m.view(-1,1,9**2),sort_freq=False),cmap='gray')
            plt.show()

        thresh_fn = self.soft_threshold
        
        I_size = I.shape
        I_col = Im2Col(I, kernel_size=params.kernel_size, stride=params.stride, padding=0, tensorized=True)

        mean_patch = I_col.mean(dim=1, keepdim=True)
        I_col = I_col - mean_patch

        lin_input = self.apply_A(I_col)

        # gamma_k = thresh_fn(lin_input, self.lmbda)

        lmbda_ = self.lmbda[0] if params.multi_lmbda else self.lmbda
        gamma_k = thresh_fn(lin_input, lmbda_)

        num_unfoldings = params.unfoldings
        N = I_col.shape[2] * I_col.shape[3] * I_col.shape[0]
        
        if params.verbose:
            _res = []
            _weighted_res = []
            _loss = []
            _penalty = []
            _weighted_loss = []
            
        for k in range(num_unfoldings - 1):
            x_k = self.apply_D(gamma_k)
            res = x_k - I_col
            r_k = self.apply_A(res)

            lmbda_ = self.lmbda[k + 1] if params.multi_lmbda else self.lmbda
            gamma_k = thresh_fn(gamma_k - r_k, lmbda_)

            if params.verbose:
                lmbda = lmbda_ * L

                # euclidian
                residual = 0.5 * (x_k - I_col).pow(2).sum() / N
                penalty =   (lmbda* gamma_k.abs()).sum(1).sum() / N
                loss = residual + penalty
                
                # weighted loss
                v = (x_k - I_col)
                Wv = torch.einsum('bvhw,nm->bmhw',(v,W))
                vWv = torch.einsum('bvhw,bvhw->bhw',(v,Wv))
                weighted_res = 0.5 * (vWv).sum()/N
                weigthed_loss = weighted_res + penalty

                unique_loss =0.5 * (x_k - I_col).pow(2).sum(1)[0,0,0] + (lmbda* gamma_k.abs()).sum(1)[0,0,0]
                unique_weighted = 0.5 * (vWv)[0,0,0] + (lmbda * gamma_k).abs().sum(1)[0,0,0]
                
                _res.append(residual.item())
                _weighted_res.append(weighted_res.item())
                _weighted_loss.append(weigthed_loss.item())
                _loss.append(loss.item())
                _penalty.append(penalty.item())
                
                tqdm.write(
                    f' res {residual.item():.2e} | w-res {weighted_res.item():.2e} | sparsity {sparsity(gamma_k):.2e}'
                    f' | loss {loss.item():0.4e} | weighted {weigthed_loss:0.4e}')

        output_all = self.apply_W(gamma_k)
        output_all = output_all + mean_patch
        output = Col2Im(output_all, I_size[2:], kernel_size=params.kernel_size, stride=params.stride, padding=0,
                        avg=True, input_tensorized=True)
        if params.verbose:
            import matplotlib.pyplot as plt
            tqdm.write('')
            plt.subplot(411)
            plt.plot(_loss,label='loss')
            plt.legend()
            plt.subplot(412)

            plt.plot(_weighted_loss,label='weighted-loss')
            plt.legend()

            plt.subplot(413)
            plt.plot(_res,label='residual')
            plt.legend()

            plt.subplot(414)

            plt.plot(_weighted_res,label='weighted-residual')
            plt.legend()
            
            # plt.subplot(313)

            # plt.plot(_penalty,label='penalty')
            plt.legend()

            plt.show()
            
        return output




