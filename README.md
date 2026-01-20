# Style Transfer

Neural style transfer using a VGG19 based CNN to blend content and artistic style through feature extraction and optimization.

# Dataset Link for Image Classification
[https://www.kaggle.com/competitions/cifar100-image-classification/data](https://www.kaggle.com/competitions/cifar100-image-classification/data)

# Images Used for Style Transfer

Content Image Used
Tesla image taken from a random Google source

Style Image Used
[https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer?select=style_image.jpg](https://www.kaggle.com/datasets/vbookshelf/art-by-ai-neural-style-transfer?select=style_image.jpg)

# Style Transfer using VGG19

This project implements neural style transfer using a VGG19 based convolutional neural network to combine the content of one image with the artistic style of another. The approach relies on feature extraction from deep convolutional layers and optimization of a generated image using custom loss functions in PyTorch.

# Project Overview

The goal of this project is to generate a new image that preserves the structural content of a given content image while adopting the visual style of a separate style image. This is achieved by extracting meaningful feature representations from a VGG19 network and optimizing a noise image using content and style losses.

# Model Architecture VGG19 Network

A custom VGG19 class is defined based on the original VGG19 architecture.

# Feature Extractor
The feature extraction part consists of convolutional layers followed by ReLU activations. These layers capture low level and high level visual features such as edges, textures, and shapes.

# Classifier
The classifier consists of fully connected layers used during the image classification training phase. These layers are not used during the style transfer process, as the focus is only on feature extraction.

# Image Processing

The image_processing function performs the following steps
Reads image files from disk
Resizes images to 224 by 224
Normalizes images using standard ImageNet mean and standard deviation
Converts images into PyTorch tensors
Adds a batch dimension to make the input compatible with the network

# Dataset Handling ImageDataset Class

A custom ImageDataset class is implemented to handle data loading. It
Reads image paths and labels from a CSV file
Processes each image using the image_processing function
Returns image and label pairs

Images are loaded efficiently using PyTorch DataLoader with a batch size of 25.

# Model Training Phase 1 Classification

The VGG19 model is first trained for image classification using the CIFAR 100 dataset.

Dataset used is CIFAR 100
Loss function used is CrossEntropyLoss
Optimizer used is Adam
Training duration is 20 epochs

During training
Predicted labels are compared with ground truth labels
Gradients are computed using backpropagation
Model weights are updated to minimize classification loss
Loss and accuracy are tracked for each batch

Although classification is not directly required for style transfer, this phase helps the network learn meaningful visual representations.

# Feature Extraction for Style Transfer

After training, a new Vgg19 class reuses the trained convolutional layers for feature extraction.

The extract_features function
Extracts feature maps from selected intermediate layers defined in required_layers
These layers capture important content and style representations
The extracted features are later used to compute content and style losses

# Loss Functions

Content Loss
Measures the difference between feature maps of the content image and the generated image. This loss helps preserve the structural details of the content image.

Style Loss
Represents style using Gram matrices computed from feature maps. It measures the difference in feature correlations between the style image and the generated image using mean squared error.

Total Loss
Total loss is computed as the weighted sum of content loss and style loss. The content weight and style weight control the emphasis on content versus style during optimization.

# Optimization Process

A noise image initialized from the content image is optimized.

Optimizer used is Adam
Number of iterations is 100

At each iteration
Features are extracted from the content image, style image, and noise image
Content loss and style loss are computed
Total loss is calculated
The noise image is updated using backpropagation to minimize total loss

# Post Processing

After optimization
The final tensor is converted back into an image using the deprocess_image function
The stylized image is saved as the final output

# Flow Summary

The VGG19 model is trained for image classification
The trained convolutional layers are reused for feature extraction
Content and style images are preprocessed into tensors
Content and style losses are computed using feature maps and Gram matrices
A noise image is optimized to minimize total loss
The final stylized image is generated and saved

# Conclusion

This project demonstrates an end to end implementation of neural style transfer using deep convolutional neural network features. It combines transfer learning, custom loss function design, and gradient based optimization to generate visually appealing images that blend content and artistic style. The project provides strong practical understanding of CNN architectures, feature representations, and image optimization.
