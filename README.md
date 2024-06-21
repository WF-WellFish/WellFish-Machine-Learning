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

### Usage
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/WF-WellFish/WellFish-Machine-Learning/tree/main/Types_of_fish/V2
    ```

2. **Open the Notebook**:
    - Go to [Google Colab](https://colab.research.google.com/)
    - Upload the `.ipynb` file from the cloned repository

3. **Run the Code**:
    - Follow the steps in the notebook to run the code, which includes:
        - Loading and preprocessing data
        - Building and training the model
        - Saving the model as an H5 file
        - Converting the model to TensorFlow.js format

4. **Save and Convert the Model**:
    ```python
    model_path = '/content/drive/My Drive/classification_of_fish_species.h5'
    model.save(model_path)
    output_dir = '/content/drive/MyDrive/tfjs_model'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    !tensorflowjs_converter --input_format keras '/content/drive/MyDrive/classification_of_fish_species.h5' '/content/drive/MyDrive/tfjs_model'
    ```


## Detection of Fish Diseases
This project aims to develop a fish disease detection system using the SSD MobileNet V2 model. Six types of diseases will be detected, namely:
1. **EUS (Epizootic Ulcerative Syndrome)**: A disease in fish caused by the parasitic fungus Aphanomyces invadans.
2. **Eye Disease**: Diseases or disorders affecting fish eyes such as cataracts, bacterial or parasitic infections, and others.
3. **Fin Lesions**: Lesions or damage to fish fins that can be caused by bacterial, fungal, parasitic infections, or poor environmental conditions.
4. **Rotten Gills**: Poor or damaged fish gills.
5. **Head Worms**: Parasitic worm infestations attacking the fish's head area.
6. **Body Worms**: Parasitic worm infestations attacking parts of the fish's body other than the head.


### Project Overview
This project involves several key steps, namely data collection, dataset labeling, model training, and model performance evaluation. The resulting system is expected to detect various types of fish diseases with high accuracy. 

### Dataset
The dataset used in this project includes images of fish infected with various types of diseases. This data is divided into training, testing, and validation datasets. Each image in the dataset has labeled to determine the type of disease present in the fish.
Link Dataset: [Dataset](https://drive.google.com/drive/folders/1_PHweflueiCAbL1Ycikoa8gRPsKowa7z?usp=sharing)

### Model Architecture
The model used in this project is SSD MobileNet V2. SSD (Single Shot Multibox Detector) is an object detection algorithm that allows detection in a single stage, making it faster compared to other object detection models that require two stages. MobileNet V2 is a convolutional neural network architecture optimized for devices with low computational power such as smartphones and IoT devices.

### Model Training
The model was trained through a series of steps totaling 36.2k steps. Below are the results of the training process:
Link: [Training](https://drive.google.com/drive/folders/1_PHweflueiCAbL1Ycikoa8gRPsKowa7z?usp=sharing]](https://drive.google.com/drive/folders/1A_wM1iKmDidlq_lmaRVnLf5kiiHDs5VU?usp=sharing)

### Results and Graph Visualization
#### Loss/Classification Loss
The graph shows a significant decrease from the beginning to the end of training. After training for 36.2k steps, the classification loss is around 0.066. This result indicates that the model is improving in classification accuracy.
![Screenshot 2024-06-20 210424](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/408a3d77-65da-4263-a555-21e8e5ef7d59)

#### Loss/Localization Loss
The graph shows a decrease from around 0.20 to approximately 0.026 at 36.2k steps. This result indicates that the model is becoming more accurate in determining object locations within the data.
![Screenshot 2024-06-20 210509](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/bb3ea8cc-35dc-4db3-bf18-957176516093)

#### Loss/Regularization Loss
The graph shows a stable decrease from around 0.34 to 0.146 at 36.2k steps. Regularization helps prevent overfitting.
![Screenshot 2024-06-20 210532](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/92885b91-f247-49e6-a395-ca54c74bc44b)

#### Loss/Total Loss
The graph shows a stable decrease from around 0.80 to 0.238 at 36.2k steps. The total loss is a combination of all the measured loss components, and the decrease in total loss indicates an improvement in the model's performance.
![Screenshot 2024-06-20 210603](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/f5b064bc-22f8-4cc2-b245-32a18fdaaf48)

#### Machine Learning Model Simulation for Fish Type Classification
The simulation demonstrates the model correctly identifying a submitted fish disease as EUS, Fin_lesions, Eye_disease, Rotten_gills showcasing its ability to accurately identify fish disease from photographs.
![download](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/5416fc72-eb7c-4dc9-83b8-46a80072f68d)
![download](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/b6491afb-1c2b-460c-929a-cd555a8f7021)
![download](https://github.com/WF-WellFish/WellFish-Machine-Learning/assets/130066367/7d952ca0-6ea5-47f0-8c9d-f80df7cf0b2b)

The model shows strong generalization capabilities and accurately detect fish disease. 

#### mAP Results for Fish Disease Detection Using SSD MobileNet V2 Model
The following are the results of calculating the mean Average Precision (mAP) at various Intersection over Union (IoU) thresholds for detecting different types of fish diseases using the SSD MobileNet V2 model:


| Class        | Average mAP @ 0.5:0.95 |
|--------------|-------------------------|
| EUS          | 56.41%                 |
| Eye_disease  | 64.13%                 |
| Fin_lesions  | 54.63%                 |
| Rotten_gills | 59.60%                 |
| Head_Worms   | 29.45%                 |
| Body_Worms   | 49.88%                 |
| **Overall**  | **52.35%**             |

### Requirements
- TensorFlow
- CUDA
- Cython
- Protobuf
- Keras
- Seaborn
- OpenCV
- TensorFlow.js

## Final Note
Congratulations! The machine learning model for fish species classification is now ready for use.


---
**Copyright © WellFish 2024**
