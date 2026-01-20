# Style-Transfer
Neural style transfer using a VGG19-based CNN to blend content and artistic style through feature extraction and optimization.

# Dataset Link for Image Classification:
https://www.kaggle.com/competitions/cifar100-image-classification/data

Images Used for Style Transfer :

Content Image used: 
Tesla's Image from random google page

Style Image used:
https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer?select=style_image.jpg

# Style Transfer using VGG19

Neural style transfer implementation using a VGG19-based convolutional neural network to blend the content of one image with the artistic style of another through feature extraction and optimization in PyTorch.

# Project Overview

This project implements Neural Style Transfer (NST) by leveraging the deep convolutional layers of a VGG19 model. The core idea is to generate a new image that preserves the content of a given content image while adopting the style of a separate style image. This is achieved by extracting meaningful feature representations and optimizing a noise image using custom loss functions.

Model Architecture
VGG19 Network

The project defines a custom VGG19 class inspired by the original VGG19 architecture:

Feature Extractor
Consists of convolutional layers followed by ReLU activations to extract low-level and high-level visual features such as edges, textures, and shapes.

#Classifier
Fully connected layers used for image classification during the initial training phase. These layers are not directly used during style transfer.

#Image Processing

The image_processing() function:

Reads image files from disk

Resizes images to 224×224

Normalizes images using standard ImageNet mean and standard deviation

Converts images into PyTorch tensors

Adds a batch dimension to make them compatible with the network

#Dataset Handling
ImageDataset Class

A custom dataset class that:

Loads image paths and labels from a CSV file

Applies preprocessing using image_processing()

Returns image–label pairs

Images are loaded efficiently using PyTorch’s DataLoader with a batch size of 25.

Model Training (Phase 1: Classification)

Dataset: CIFAR-100

Loss Function: CrossEntropyLoss

Optimizer: Adam

Training Duration: 20 epochs

During training:

Predictions are compared with ground-truth labels

Gradients are computed via backpropagation

Model weights are updated to minimize classification loss

Accuracy and loss are tracked per batch

Although classification is not required for style transfer, this phase trains the VGG19 network to learn meaningful visual representations.

Feature Extraction for Style Transfer

After training, a new class Vgg19 reuses the trained convolutional layers for feature extraction.

extract_features()

Extracts feature maps from selected intermediate layers (required_layers)

These layers capture important content and style representations

Outputs are later used to compute content and style losses

Loss Functions
Content Loss

Measures the difference between feature maps of the content image and the generated image

Encourages preservation of structural details

Style Loss

Uses Gram matrices to represent style

Measures correlation differences between feature maps of the style image and generated image

Computed using Mean Squared Error (MSE)

Total Loss
Total Loss = content_w × Content Loss + style_w × Style Loss

Where content_w and style_w control the emphasis on content versus style.

Optimization Process

A noise image (initialized from the content image) is optimized

Optimizer: Adam

Iterations: 100 steps

At each step:

Extract features from content, style, and noise images

Compute content and style losses

Calculate total loss

Update the noise image via backpropagation

Post-processing

After optimization:

The final tensor is converted back to an image using deprocess_image()

The stylized image is saved as the output

Flow Summary

Train VGG19 for image classification

Reuse convolutional layers for feature extraction

Preprocess content and style images

Compute content and style losses using feature maps and Gram matrices

Optimize a noise image to minimize total loss

Generate and save the stylized output image

Conclusion

This project demonstrates an end-to-end implementation of neural style transfer using deep CNN features. It combines transfer learning, custom loss design, and gradient-based optimization to generate visually appealing images that blend content and artistic style. The work provides strong practical insight into CNN internals, feature representations, and image optimization.
