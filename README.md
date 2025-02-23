# Multiclass-Fish-Image-Classification

1. Project Overview:

Project Description:
```
This project aims to classify fish images into multiple species using Deep Learning. The dataset comprises images of various fish categories, and the goal is to build a model that can accurately predict the fish species from a single image.
```

Key Objectives:
```
1. Enhanced Accuracy: Test different models (CNN from scratch and pre-trained networks) to achieve high classification accuracy.
2. Deployment: Implement a user-friendly Streamlit web application for real-time inference.
3. Model Comparison: Collect and compare performance metrics (accuracy, precision, recall, F1-score, etc.) across models to identify the best architecture.
```

Business Use Case / Value:
```
* Fisheries & Aquaculture: Automate identification of fish species to enhance supply chain and inventory systems.
* Wildlife Conservation: Assist in monitoring fish populations by quickly cataloging species from images.
* Consumer Apps: Hobbyists and educators can learn fish species in real-time using a simple web interface.
```
2. Methodology:

Data Collection and Preprocessing:
```
* Dataset: Comprises images of various fish species, placed in subfolders named after each fish category
* Structure: Images are organized into subfolders by species for easy loading.
* Approach:
- Each subfolder represents a class.
- A train-validation split is handled automatically by TensorFlow’s ImageDataGenerator (e.g., 80/20).
* Preprocessing:
- Rescaling: Pixel values normalized to the [0,1] range.
- Augmentation: Random rotations, shifts, zooms, and flips to increase model robustness.
- Train/Validation Split: A typical 80/20 split is used.
```

Model Architectures:
```
1. CNN From Scratch:
* Three convolutional+pooling blocks, followed by fully connected layers.
2. Transfer Learning:
* VGG16
* ResNet50
* MobileNet
* InceptionV3
* EfficientNetB0
* These models are initialized with ImageNet weights (pre-trained) and extended with custom layers for the fish classification task.
```

Model Training:
```
* Hyperparameters:
- Batch Size: 32
- Learning Rate: 0.001 (for scratch CNN), 0.0001 (for transfer learning)
- Epochs: 10 (stopped early if validation loss does not improve)
* Loss Function: categorical_crossentropy
* Optimization: Adam
* Callbacks: Early Stopping (patience=5) to avoid overfitting
```

Model Evaluation:
```
* Accuracy: Quick measure of correct predictions / total predictions.
* Precision, Recall, F1-Score: Evaluate model performance per class.
* Confusion Matrix: Visualize misclassifications across classes.
```

3. Implementation Details:

Code Organization:
```
* Data Preprocessing: Handled by ImageDataGenerator in TensorFlow/Keras.
* Model Building: Created separate scripts/notebooks for:
- CNN from scratch
- Transfer learning with each pre-trained architecture
* Training: Each model is trained and validated; results (accuracy, loss) are plotted.
```
Transfer Learning Models:
```
We load five pre-trained networks (trained on ImageNet) and adapt them to our fish classification problem. The general approach is:
* Load a pre-trained base (e.g., ResNet50).
* Freeze its layers initially to leverage learned features.
* Add GlobalAveragePooling2D and Dense layers for our classification task.
```

Streamlit Deployment:
```
streamlit_app.py:
* Loads the best-performing best_model.h5 model.
* Provides a file uploader for images.
* Makes a prediction and displays the results with confidence scores.
```

4. Results and Analysis

Evaluation Results:
Accuracy Comparison:
```
After training each model, collect their final validation accuracies:

Model Validation    Accuracy
CNN (Scratch)     ~ 95.85% 
VGG16             ~ 75.5%
ResNet50          ~ 18.58%
MobileNet         ~ 98.14%
InceptionV3       ~ 96.13%
EfficientNetB0    ~ 17.15%

Highest performer in this example is MobileNet with ~98.14% validation accuracy.
```
Key Insights:
```
* Training Time: Pre-trained models often converge faster and yield higher accuracy compared to the CNN from scratch.
* Data Augmentation: Helped reduce overfitting and improved generalization.
* Fine-tuning: If you partially unfreeze certain layers of the pre-trained networks, you can boost performance but also risk overfitting if the dataset is small.
```

5. Conclusion and Next Steps:

```
Conclusion:

The Multiclass Fish Image Classification project successfully demonstrates how Deep Learning can be applied to image-based fish species identification. By combining data augmentation, transfer learning, and a Streamlit deployment, we can build a robust solution that can be easily integrated into real-world applications such as fisheries management, wildlife monitoring, or consumer educational tools.

* The MobileNet architecture achieved the highest accuracy outpacing both a custom CNN from scratch and other pre-trained models in this project.
* A Streamlit application was successfully developed, enabling real-time fish species classification from user-uploaded images.
```

Future Enhancements:
```
1. Larger / More Diverse Dataset: Improving robustness by including additional fish species or more variations (lighting, background).
2. Hyperparameter Tuning: Experimenting with different learning rates, batch sizes, optimizers, and fine-tuning strategies.
3. Advanced Architectures: Exploring newer models (e.g., Vision Transformers) for potential gains.
4. Deployment Options: Containerizing the application using Docker or deploy to cloud platforms (AWS, Azure, or Streamlit Cloud) for broader accessibility.
```








