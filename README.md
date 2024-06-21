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

### Results and Graph Visualization
#### Plot Learning Curve
The learning curve plot shows our CNN model's performance in fish species classification. Training accuracy (blue) reaches nearly 100%, and validation accuracy (orange) stabilizes around 95%, indicating strong generalization and high accuracy.
![image](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/c643ce10-0bb6-4911-af28-81831e559ac8)

#### Plot Loss Curve 
The loss curve graph illustrates our CNN model's performance. Training loss (blue) consistently decreases, while validation loss (orange) also decreases and stabilizes, showing effective learning and good generalization.
![image](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/ddcb1b8d-cd7c-4aa1-9df0-7ac713bd6bf5)

#### Evaluate the Model on New Testing Data
On new test data, the model achieved a test loss of 0.5224 and a test accuracy of 90.57%, indicating good performance and successful generalization to unseen data.
![Screenshot 2024-06-20 135241](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/b5fc5922-c9d1-46d1-88a8-b52f0ae59af0)

#### Confusion Matrix on New Testing Data
The confusion matrix shows that the CNN model accurately categorizes most fish species, though some misclassifications highlight areas for improvement.
![image](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/7a0701dc-b24e-4aab-96c1-fbd78025a0d9)

#### Machine Learning Model Simulation for Fish Type Classification
The simulation demonstrates the model correctly identifying a submitted fish image as "Bangus" (milkfish), showcasing its ability to accurately identify fish species from photographs.
![Screenshot 2024-06-21 012953](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/72650963/a85ae81d-39d3-4c6b-b9da-93c8f4382469)

The model shows strong generalization capabilities and accurately classifies most fish species.

### Requirements
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- OpenCV
- TensorFlow.js

### Environment Setup
To set up the environment and run the project in Google Colab, follow these steps:

1. **Mount Google Drive**:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install Dependencies**:
    ```python
    !pip install tensorflowjs
    ```


