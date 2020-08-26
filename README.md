# GroupSC

This is the code for our ECCV 2020 paper : 
[Revisiting Non Local Sparse Models for Image Restoration](https://arxiv.org/abs/1912.02456).
In this work, we propose a differentiable unrolled algorithm to solve the group lasso optimization problem 
in order to enforce joint sparsity among sets of similar patches.
The code reproduces the results presented in the paper.

Please cite our work if you find it useful for your research and work:
```
@article{lecouat2020fully,
  title={Fully Trainable and Interpretable Non-Local Sparse Models for Image Restoration},
  author={Lecouat, Bruno and Ponce, Jean and Mairal, Julien},
  journal={Proc. European Conference on Computer Vision (ECCV)},
  year={2020}
}
```
## Requirements

The repo supports python 3.6 + pytorch 1.3.0

## Run the Code

To train a new model for color denoising:
```
python train_color_denoising.py
```
To train a new model for gray denoising:
```
python train_gray_denoising.py
```

To train a new model for gray denoising:
```
python train_mosaic.py
```

To evaluate some pretrained models (choose the model and the noise level accordingly):

```
python test_gray.py --model_name trained_model/gray/****/ckpt --noise_level ***
```

```
python test_color.py --model_name trained_model/color/****/ckpt --noise_level ***

```

```
 python test_mosaic.py --model_name trained_model/mosaic/****/ckpt
```

## Acknowledgement

This code is built on CSCnet https://github.com/drorsimon/CSCNet (PyTorch). We thank the authors for sharing their codes. 

