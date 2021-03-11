# PlueckerNet: Learn to Register 3D Line Reconstructions


This contains the datasets and codes for training the 3D line registration method described in : [PlueckerNet: Learn to Register 3D Line Reconstructions](https://arxiv.org/pdf/2012.01096.pdf), CVPR2021.


# Datasets
Please download [Structured3D and Semantic3D](https://drive.google.com/file/d/1bVI0Ny4Ly1M4cBxbgRIjgHr8DtIXZLbb/view?usp=sharing) and [Apollo](https://drive.google.com/file/d/16gVHdqsrvI1nsdFCo_NAayJbnfEAY_Si/view?usp=sharing) datasets.

Please put the downloaded files under the folder ./dataset

# Codes and Models

## Prerequisites
Pytorch=1.1.0 : (This is the version on my PC, but I think it also works on yours)

numpy

opencv

tensorboardX

easydict

logging

json

If you find missing Prerequisites, please Google and install them using conda or pip

## Overview
Our model is implemented in Pytorch. 
All our models are trained from scratch, please run the training codes to obtain models.

For pre-trained models, please [download](https://drive.google.com/file/d/1XhjGkwlSK9jV4ZGLd4Ucg327mHpr_0r9/view?usp=sharing). Under the folder of each dataset, there is a folder named preTrained and you can find it there.

Please put the downloaded pre-trained models under the folder ./output


## Training

Run: 
```diff
python main_train.py
```

## Testing

Run:
```diff
python main_test.py
```

If you have questions, please first refer to comments in scripts.

# Publications

If you like, you can cite the following publication:

*Liu, Liu, Hongdong Li, Haodong Yao, and Ruyi Zha. "PlueckerNet: Learn to Register 3D Line Reconstructions." arXiv preprint arXiv:2012.01096 (2020).*

<pre>
@article{liu2020plueckernet,
  title={PlueckerNet: Learn to Register 3D Line Reconstructions},
  author={Liu, Liu and Li, Hongdong and Yao, Haodong and Zha, Ruyi},
  journal={arXiv preprint arXiv:2012.01096},
  year={2020}
}
</pre>


# Contact

If you have any questions (NOT those you can find answers via Google), drop me an email (Liu.Liu@anu.edu.au)









