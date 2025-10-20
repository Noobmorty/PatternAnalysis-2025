# Segmentation of the HipMRI Study on Prostate Cancer with 2D UNet

# Overview
This task implements a 2D UNet convolutional neural network for segmentation of MRI images
from the HipMRI study on Prostate Cancer.

# UNet architecture

The model that I implemented for this task is a lightweight 2D UNet designed for medical
image segmentation on HipMRI slices. The model follows the standard encoder-decoder architecture
with skip connections.

The encoder consists of 4 convolutional blocks, with 2 3x3 convolutions with padding=1,
followed by a BatchNorm2d and ReLU activations. In between these blocks is a 2x2 max pooling
layer with stride 2 downsamples the feature maps, doubling the number of channels. 

The decoder mirrors the encoder but implements upsampling by a 2x2 transposed convolution(stride 2).
The upsampled feature map is concatenated with the corresponding encoder block output to keep the 
spatial details. They are then refined by a conv block similar to the encoder blocks.

In the output layer, a 1x1 convolution maps the decoder output to six segmentation classes.
During interference, a softmax activation is applied to get per pixel probabilities.

Design adjustments:
    Input size: 256x256 grayscale images.
    Output size: 256x256 segment maps
    Base feature width: 32 channels
    Depth: 3 encoder decoder levels instead of 4 to prevent latent space from being too small
            given the dataset dimensions provided
    Normalisation: BatchNorm2D applied after each convolution
    Activation: ReLU

The architecture retains the core advantages of what a UNet provides while still being lightweight
to be able to train on consumer level GPUs.

The architecture of U-Net can be seen in /recognition/2dUnet-s47151388/U-Net architecture.png

# Dependencies

The dependencies needed for this project are as listed below: 
    -torch 2.4.0
    -torchvision 0.19.0
    -numpy 1.26.4
    -nibabel 5.2.1
    -matplotlib 3.9.1
    -tqdm 4.66.5