# Semantic Segmentation Algorithms
This repository contains a suite of semantic segmentation algorithms implemented from scratch with Jax and Flax. The main aim is to implement the algorithms in the simplest possible way while maintaining stability during training and reducing training time. 

## Implementation Details
The segmentation models are trained on RGB images from the scene parse 150 dataset [1], which contains 150 classes. The output of the models will have a four dimensional shape (B, H, W, C), where B is the batch size, H is the image height, W is the image width and C is the number of classes. Some alterations were added to the original models to speed up training and make the models more robust, such as adding Group Norm and Dropout layers. The DICE loss is used as the loss function for training and evaluation. The models are saved using the Orbax checkpointer and will be provided on Huggingface once the training has completed. 

## Algorithms

### U-Net
U-Net uses the same ideas from the Fully Convolutional Network (FCN) and improves upon them. The main idea is to use an encoder-decoder architecture with skip connections from the encoder layers to the decoder layers. This provides global and local information to the final segmentation layers, which improves the classification and localization in the predicted segmentation. U-Net has a symetric architecture, giving it the U shape it was named after. It's simpler to implement than FCN and is also very fast. This has made U-Net one of the most popular segmentation models today. 



## Installation Requirements
If you have a GPU you can install Jax by running the following first:
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
All the requirements are provided below:
```
pip install datasets
pip install flax
pip install augmax
pip install -qq nest_asyncio
pip install matplotlib
pip install pandas
pip install jupyter
pip install scikit-learn
```

## References
- [1] [Scene Parse 150 Dataset](https://huggingface.co/datasets/scene_parse_150)
- [2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)