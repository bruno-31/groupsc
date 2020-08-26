import dataloaders
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
import torch.nn.functional as F
import time
from ops.utils_blocks import block_module
from ops.utils import show_mem, generate_key, save_checkpoint, str2bool, step_lr, get_lr

parser = argparse.ArgumentParser()
#model
parser.add_argument("--mode", type=str, default='group',help='[group, sc]')
parser.add_argument("--stride", type=int, dest="stride", help="stride size", default=1)
parser.add_argument("--num_filters", type=int, dest="num_filters", help="Number of filters", default=256)
parser.add_argument("--kernel_size", type=int, dest="kernel_size", help="The size of the kernel", default=9)
parser.add_argument("--noise_level", type=int, dest="noise_level", help="Should be an int in the range [0,255]", default=25)
parser.add_argument("--unfoldings", type=int, dest="unfoldings", help="Number of LISTA step unfolded", default=24)
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Size of image blocks to process", default=56)
parser.add_argument("--rescaling_init_val", type=float, default=1.0)
parser.add_argument("--lmbda_prox", type=float, default=0.02, help='intial threshold value of lista')
parser.add_argument("--spams_init", type=str2bool, default=1, help='init dict with spams dict')
parser.add_argument("--nu_init", type=float, default=1, help='convex combination of correlation map init value')
parser.add_argument("--corr_update", type=int, default=3, help='choose update method in [2,3] without or with patch averaging')
parser.add_argument("--multi_theta", type=str2bool, default=1, help='wether to use a sequence of lambda [1] or a single vector during lista [0]')
parser.add_argument("--diag_rescale_gamma", type=str2bool, default=0,help='diag rescaling code correlation map')
parser.add_argument("--diag_rescale_patch", type=str2bool, default=1,help='diag rescaling patch correlation map')
parser.add_argument("--freq_corr_update", type=int, default=6, help='freq update correlation_map')
parser.add_argument("--mask_windows", type=int, default=1,help='binarym, quadratic mask [1,2]')
parser.add_argument("--center_windows", type=str2bool, default=1, help='compute correlation with neighboors only within a block')
parser.add_argument("--multi_std", type=str2bool, default=0)

#training
parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=6e-4)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="ADAM Learning rate step for decay", default=80)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.35)
parser.add_argument("--backtrack_decay", type=float, help='decay when backtracking',default=0.8)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--validation_every", type=int, default=10, help='validation frequency on training set (if using backtracking)')
parser.add_argument("--backtrack", type=str2bool, default=1, help='use backtrack to prevent model divergence')
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=300)
parser.add_argument("--train_batch", type=int, default=32, help='batch size during training')
parser.add_argument("--test_batch", type=int, default=10, help='batch size during eval')
parser.add_argument("--aug_scale", type=int, default=0)

#save
parser.add_argument("--out_dir", type=str, dest="out_dir", help="Results' dir path", default='./trained_model')
parser.add_argument("--model_name", type=str, dest="model_name", help="The name of the model to be saved.", default=None)
parser.add_argument("--data_path", type=str, dest="data_path", help="Path to the dir containing the training and testing datasets.", default="./datasets/")
parser.add_argument("--resume", type=str2bool, dest="resume", help='Resume training of the model',default=True)
parser.add_argument("--dummy", type=str2bool, dest="dummy", default=False)
parser.add_argument("--tqdm", type=str2bool, default=False)

#inference
parser.add_argument("--stride_test", type=int, default=12, help='stride of overlapping image blocks [4,8,16,24,48] kernel_//stride')
parser.add_argument("--stride_val", type=int, default=48, help='stride of overlapping image blocks for validation [4,8,16,24,48] kernel_//stride')
parser.add_argument("--test_every", type=int, default=100, help='report performance on test set every X epochs')
parser.add_argument("--block_inference", type=str2bool, default=True,help='if true process blocks of large image in paralel')
parser.add_argument("--pad_image", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--pad_block", type=str2bool, default=1,help='padding strategy for inference')
parser.add_argument("--pad_patch", type=str2bool, default=0,help='padding strategy for inference')
parser.add_argument("--no_pad", type=str2bool, default=False, help='padding strategy for inference')
parser.add_argument("--custom_pad", type=int, default=None,help='padding strategy for inference')

parser.add_argument("--verbose", type=str2bool, default=0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else os.cpu_count()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

test_path = [f'{args.data_path}/BSD68/']
train_path = [f'{args.data_path}/BSD400/']
val_path = train_path

noise_std = args.noise_level / 255

loaders = dataloaders.get_dataloaders(train_path, test_path, val_path, crop_size=args.patch_size,
                                      batch_size=args.train_batch, downscale=args.aug_scale, concat=1,grey=True)


if args.mode == 'sc':
    print('sc mode')
    from model.gray_sc import ListaParams
    from model.gray_sc import Lista

    params = ListaParams(kernel_size=args.kernel_size, num_filters=args.num_filters, stride=args.stride, spams=args.spams_init,
                         unfoldings=args.unfoldings,threshold=args.lmbda_prox, multi_lmbda=args.multi_theta, verbose=args.verbose)

elif args.mode == 'group':
    print('group mode')
    from model.gray_group import ListaParams
    from model.gray_group import  groupLista as Lista

    params = ListaParams(kernel_size=args.kernel_size, num_filters=args.num_filters, stride=args.stride,
                         unfoldings=args.unfoldings, freq=args.freq_corr_update, corr_update=args.corr_update,
                         lmbda_init=args.lmbda_prox, h=args.rescaling_init_val, spams=args.spams_init,
                         multi_lmbda=args.multi_theta,
                         center_windows=args.center_windows, std_gamma=args.diag_rescale_gamma,
                         std_y=args.diag_rescale_patch, block_size=args.patch_size, nu_init=args.nu_init,
                         mask=args.mask_windows, multi_std=args.multi_std)

else:
    raise NotImplementedError

model = Lista(params).to(device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.eps)

if args.backtrack:
    reload_counter = 0

pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Arguments: {vars(args)}')
print('Nb tensors: ',len(list(model.named_parameters())), "; Trainable Params: ", pytorch_total_params, "; device: ", device,
      "; name : ", device_name)

psnr = {x: np.zeros(args.num_epochs) for x in ['train', 'test', 'val']}

model_name = args.model_name if args.model_name is not None else generate_key()

out_dir = os.path.join(args.out_dir, model_name)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

ckpt_path = os.path.join(out_dir+'/ckpt')
config_dict = vars(args)

if args.resume:
    if os.path.isfile(ckpt_path):
        try:
            print('\n existing ckpt detected')
            checkpoint = torch.load(ckpt_path)
            start_epoch = checkpoint['epoch']
            psnr_validation = checkpoint['psnr_validation']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{ckpt_path}' (epoch {start_epoch})")
        except Exception as e:
            print(e)
            print(f'ckpt loading failed @{ckpt_path}, exit ...')
            exit()

    else:
        print(f'\nno ckpt found @{ckpt_path}')
        start_epoch = 0
        psnr_validation = 22.0
        if args.backtrack:
            state = {'psnr_validation': psnr_validation,
                     'epoch': 0,
                     'config': config_dict,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(), }
            torch.save(state, ckpt_path + '_lasteval')

print(f'... starting training ...\n')

l = args.kernel_size // 2
mask = F.conv_transpose2d(torch.ones(1, 1, args.patch_size - 2 * l, args.patch_size - 2 * l),
                          torch.ones(1, 1, args.kernel_size, args.kernel_size))
mask /= mask.max()
mask = mask.to(device=device)

epoch = start_epoch

while epoch < args.num_epochs:

    tic = time.time()

    phases = ['train',  'val', 'test',]

    for phase in phases:
        if phase == 'train':
            if (epoch % args.lr_step) == 0 and (epoch != 0) :
                step_lr(optimizer, args.lr_decay)
            model.train()

        elif phase == 'val':
            if not (args.backtrack and ((epoch+1) % args.validation_every == 0)):
                continue
            model.eval()   # Set model to evaluate mode
            print(f'\nstarting validation on train set with stride {args.stride_val}...')


        elif phase == 'test':
            if (epoch+1) % args.test_every != 0:
                continue # test every k epoch
            print(f'\nstarting eval on test set with stride {args.stride_test}...')
            model.eval()  # Set model to evaluate mode


        # Iterate over data.
        num_iters = 0
        psnr_set = 0

        loader = loaders[phase]

        for batch in tqdm(loader,disable=not args.tqdm):
            batch = batch.to(device=device)
            noise = torch.randn_like(batch) * noise_std
            noisy_batch = batch + noise
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):

                # Block inference during test phase
                if (phase == 'test' or phase == 'val'):

                    if phase == 'val':
                        stride_test = args.stride_val
                    else:
                        stride_test = args.stride_test

                    if args.block_inference:
                        params = {
                            'crop_out_blocks': 0,
                            'ponderate_out_blocks': 1,
                            'sum_blocks': 0,
                            'pad_even': 1,  # otherwise pad with 0 for las
                            'centered_pad': 0,  # corner pixel have only one estimate
                            'pad_block': args.pad_block,  # pad so each pixel has S**2 estimate
                            'pad_patch': args.pad_patch,  # pad so each pixel from the image has at least S**2 estimate from 1 block
                            'no_pad': args.no_pad,
                            'custom_pad': args.custom_pad,
                            'avg': 1}
                        block = block_module(args.patch_size, stride_test, args.kernel_size, params)
                        batch_noisy_blocks = block._make_blocks(noisy_batch)
                        patch_loader = torch.utils.data.DataLoader(batch_noisy_blocks, batch_size=args.test_batch, drop_last=False)
                        batch_out_blocks = torch.zeros_like(batch_noisy_blocks)

                        for i, inp in enumerate(patch_loader):  # if it doesnt fit in memory
                            id_from, id_to = i * patch_loader.batch_size, (i + 1) * patch_loader.batch_size
                            batch_out_blocks[id_from:id_to] = model(inp)

                        output = block._agregate_blocks(batch_out_blocks)
                    else:
                        output = model(noisy_batch)

                    loss_psnr = -10 * torch.log10((output.clamp(0.,1.) - batch).pow(2).flatten(2, 3).mean(2)).mean()

                if phase == 'train':

                    output = model(noisy_batch)
                    loss = (mask * (output - batch)).pow(2).sum() / batch.shape[0]

                    loss_psnr = -10 * torch.log10((output - batch).pow(2).flatten(2, 3).mean(2)).mean()

                    loss.backward()
                    optimizer.step()

            psnr_set += loss_psnr.item()
            num_iters += 1

            if args.dummy:
                break

        tac = time.time()
        psnr_set /= num_iters

        psnr[phase][epoch] = psnr_set

        if phase == 'val':
            r_err = -(psnr_set - psnr_validation)
            print(
                f'validation psnr {psnr_set:0.4f}, {psnr_validation:0.4f}, absolute_delta {-r_err:0.2e}, reload counter {reload_counter}')
            path = ckpt_path + '_lasteval'

            if r_err > 0.2:  # test divergence
                if os.path.isfile(path):
                    try:
                        print('backtracking: previous ckpt detected')
                        checkpoint = torch.load(path)
                        epoch = checkpoint['epoch']
                        model.load_state_dict(checkpoint['state_dict'])
                        optimizer.load_state_dict(checkpoint['optimizer'])
                        [step_lr(optimizer, args.backtrack_decay) for _ in range(reload_counter + 1)]
                        print(f"loaded checkpoint '{path}' (epoch {epoch}), decreasing lr ==> {get_lr(optimizer):0.2e}")
                        reload_counter += 1
                    except Exception as e:
                        print('catched exception :')
                        print(e)
                        print(f'ckpt loading failed @{path}')
                else:
                    print('no ckpt found for backtrack')
            else:
                reload_counter = 0
                state = {'psnr_validation': psnr_validation,
                         'epoch': epoch,
                         'config': config_dict,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(), }
                torch.save(state, ckpt_path + '_lasteval')
                psnr_validation = psnr_set

        if torch.cuda.is_available():
            mem_used, max_mem = show_mem()
            tqdm.write(f'epoch {epoch} - {phase} psnr: {psnr[phase][epoch]:0.4f} ({tac-tic:0.1f} s,  {(tac - tic) / num_iters:0.3f} s/iter, max gpu mem allocated {max_mem:0.1f} Mb, lr {get_lr(optimizer):0.1e})')
        else:
            tqdm.write(f'epoch {epoch} - {phase} psnr: {psnr[phase][epoch]:0.4f} ({(tac-tic)/num_iters:0.3f} s/iter,  lr {get_lr(optimizer):0.2e})')

        with open(f'{out_dir}/{phase}.psnr','a') as psnr_file:
            psnr_file.write(f'{psnr[phase][epoch]:0.4f}\n')

    epoch += 1
    ##################### saving #################
    save_checkpoint({'epoch': epoch,
                     'config': config_dict,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict(),
                     'psnr_validation':psnr_validation}, ckpt_path)

