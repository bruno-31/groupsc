# group-sc

This is the code for our paper : Revisiting Non Local Sparse Models for Image Restoration. https://arxiv.org/abs/1912.02456.
The code reproduces the results presented in the paper.

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

To evaluate some pretrained models (choose the model and the noise level accordingly):

```
python test_gray.py --model_name trained_model/gray/****/ckpt --noise_level ***
```

```
python test_color.py --model_name trained_model/color/****/ckpt --noise_level ***

```

```
 python test_mosaic.py --ssim 0 --model_name trained_model/mosaic/****/ckpt
```

## Acknowledgement

This code is built on CSCnet https://github.com/drorsimon/CSCNet (PyTorch). We thank the authors for sharing their codes. 

