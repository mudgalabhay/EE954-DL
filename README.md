# Fashion-MNIST Classification with CNN and MLP

This project involves building and training a Convolutional Neural Network (CNN) combined with a Multi-Layer Perceptron (MLP) to classify images from the Fashion-MNIST dataset. The CNN extracts features from the images, which are then passed to the MLP for classification.

## Model Architectures

### CNN Model
The CNN model consists of 5 convolutional layers, each followed by ReLU activation and a max-pooling layer. The architecture is designed to extract hierarchical features from the input images.

- **Layer 1:** Conv2d (1 input channel, 16 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 2:** Conv2d (16 input channels, 32 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 3:** Conv2d (32 input channels, 64 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 4:** Conv2d (64 input channels, 128 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=2)
- **Layer 5:** Conv2d (128 input channels, 256 output channels, kernel size=2, padding=1), ReLU, MaxPool2d (kernel size=2, stride=1)

The final output is flattened to create feature vectors for the MLP.

### MLP Model
The MLP model takes the flattened features from the CNN and performs the classification task.

- **Layer 1:** Linear (256 input features, 256 output features), ReLU
- **Layer 2:** Linear (256 input features, 128 output features), ReLU
- **Layer 3:** Linear (128 input features, 10 output features)

## Training Setup

### Hyperparameters
- **Kernel Size:** 2
- **Number of Kernels:** [16, 32, 64, 128, 256]
- **Learning Rate:** 0.001
- **Number of Epochs:** 10

### Weight Initialization Methods
- Xavier Initialization

## Training and Evaluation

### Instructions for Running the Models

1. **Define the CNN and MLP Models:**
   - Use the provided `CNN_Model` and `MLP` class definitions.

2. **Initialize the Models:**
   - Initialize the models with the desired hyperparameters and weight initialization methods.

3. **Train and Evaluate the Models:**
   - Use the `train_and_evaluate` function to train and evaluate the models.

4. **Experiment with Different Hyperparameters:**
   - Run the experiments by adjusting the hyperparameters and weight initialization methods.
