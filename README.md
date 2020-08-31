# GroupSC

This is the code for our **ECCV 2020** paper : 
[Fully Trainable and Interpretable Non-Local
Sparse Models for Image Restoration](https://arxiv.org/abs/1912.02456).
In this work, we propose an unrolled **relaxation of the group lasso** to enforce **joint sparsity** among sets of similar patches.
The code reproduces the results presented in the paper for several image restoration problems.


#### New :  Variance Reduction Penalty

We also integrated the variance reduction mechanism which increases performance of our models
 (pretrained models will be available soon). The idea being
to encourage concistency of pixel's predictions among overlapping patches.
More details regarding the variance reduction penalty can be found in this paper [Designing and Learning Trainable Priors with Non-Cooperative Games
](https://arxiv.org/abs/2006.14859).

Please cite our work if you find it useful for your research and work:
```
@article{lecouat2020fully,
  title={Fully Trainable and Interpretable Non-Local Sparse Models for Image Restoration},
  author={Lecouat, Bruno and Ponce, Jean and Mairal, Julien},
  journal={Proc. European Conference on Computer Vision (ECCV)},
  year={2020}
}

@article{lecouat2020designing,
  title={Designing and Learning Trainable Priors with Non-Cooperative Games},
  author={Lecouat, Bruno and Ponce, Jean and Mairal, Julien},
  journal={arXiv preprint arXiv:2006.14859},
  year={2020}
}
```
## Requirements

The repo supports python 3.6 + pytorch 1.3.0

## Datasets
All the models are trained on BSD400 and tested on BSD68. Simply modify 
the arguments ``--test_path`` and ``--train_path`` for training/testing on other datatets.

## Results

###Demosaicking
(Training on CBSD400)

| Model                | Params | Kodak24 | CBSD68 | 
|----------------------|:--------:|:-------:|:------:|
| IRCNN                |   -    |  40.54  |  39.90 |  
| RNAN                 | 8.96M  |  42.86  |  42.61 |  
|**GroupSc**       | 119k   |  42.72  |  42.96 |   

### Color denoising
(Training on CBSD400)

| Model   | Params | sigma=5 | sigma=15 | sigma=25 | sigma=50 |
|---------|:------:|:-------:|:--------:|:--------:|:--------:|
| CDnCNN   |   668k     |  40.50       |    33.99      |  31.31        |   28.01       |
| **GroupSC** |  119k      | 40.59        |   34.12       |  31.41        |  28.08        |

### Gray denoising
(Training on BSD400)

| Model   | Params | sigma=5 | sigma=15 | sigma=25 | sigma=50 |
|---------|:------:|:-------:|:--------:|:--------:|:--------:|
| DnCNN   |   556k     |    37.68     |   31.73       |    29.22      |    26.23      |
| **GroupSC** |  68k      |   37.96      |   31.70       |  29.19       |   26.18       |


## Run the Code


To train a new model for color denoising:
```
python train_color_denoising.py
```
To train a new model for gray denoising:
```
python train_gray_denoising.py
```

To train a new model for demosaicking:
```
python train_mosaic.py
```

## Pretrained models

Pretrained models are also available on google drive [here](https://drive.google.com/drive/folders/1jHupBV1n7NaOAvnoNTA71o3NjuyVAtWc?usp=sharing).
To evaluate a model, please choose the model and the noise level accordingly.
See below some examples of commands to evaluate pretrained models.

```
python test_gray.py --model_name PATH_TO_CKPT/gray/****/ckpt --noise_level ***
```

```
python test_color.py --model_name PATH_TO_CKPT/color/****/ckpt --noise_level ***

```

```
 python test_mosaic.py --model_name PATH_TO_CKPT/demosaicking/ckpt
```

## Acknowledgements

This code structure is inspired from CSCnet https://github.com/drorsimon/CSCNet (PyTorch). We thank the authors for sharing their codes. 

