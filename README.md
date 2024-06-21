# Well Fish Machine Learning 
![WhatsApp Image 2024-06-17 at 23 44 55_78286244](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/68eeefbd-c4e1-48ba-9e2e-401c729e4cb4)

# Main Features
1. **Classification of Fish Species**: The application can accurately identify different species of fish from images.
2. **Detection of Fish Diseases**: The application can detect various diseases in fish based on image analysis.

## Classification of Fish Species using CNN Model
This project focuses on classifying different species of fish using a Convolutional Neural Network (CNN) model. The model is trained and evaluated on a dataset of fish images and demonstrates high accuracy and generalization capabilities.

### Project Overview
This project utilizes a CNN model to classify images of different fish species. The model is trained on a diverse dataset of fish images and evaluated to ensure high accuracy and robust performance on unseen data.

### Dataset
The dataset consists of images of various fish species divided into training, validation, and test sets. The images are preprocessed and augmented to improve model generalization.
Link Dataset: https://drive.google.com/drive/folders/1w5R5eo_bn3_fdId_UJ0LGVIK3PRM2MP6?usp=sharing

### Model Architecture
The CNN model is built using the following layers:
- Convolutional layers with Leaky ReLU activation
- MaxPooling layers
- Flatten layer
- Fully connected Dense layers with Leaky ReLU activation
- Output layer with softmax activation for multi-class classification

### Training and Evaluation
The model is trained using the Adam optimizer and categorical cross-entropy loss. Early stopping is used to prevent overfitting. The training process is visualized using learning and loss curves.
