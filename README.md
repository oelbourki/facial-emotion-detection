## Facial Emotion Recognition: A Deep Learning Approach 

This repository hosts a Python project dedicated to building and training a facial emotion recognition system using deep learning techniques. The application utilizes Convolutional Neural Networks (CNNs) to classify facial expressions into seven distinct categories:

* **Angry**
* **Disgusted**
* **Fearful**
* **Happy**
* **Neutral**
* **Sad**
* **Surprised**

**Project Features:**

* **Real-time Emotion Detection:** Analyze live video streams from webcams to detect and display emotions in real-time.
* **Image-Based Emotion Recognition:** Upload static images and receive predictions of the depicted emotional states.
* **GUI Integration:** Interact with the application seamlessly through a user-friendly graphical interface built with Tkinter.
* **Pre-trained Model Options:** Utilize pre-trained CNN architectures like VGG16 and ResNet50, fine-tuned for facial emotion recognition, or train custom models.
* **Performance Evaluation:** Assess the accuracy and performance of different models through comprehensive testing and evaluation.

**Project Structure:**

```
facial-expression-recognition/
├── src/
│   ├── gui.py            # Main script for the graphical user interface
│   └── functions.py     # Contains functions for model training, emotion detection, image processing, etc.
├── examples/             # Sample images for testing
│   ├── happy-human.jpg
│   ├── sad-human.jpg
│   └── fearful-human.jpg
├── facial-emotion-detection.ipynb  # Jupyter Notebook detailing model training and experimentation
├── .gitignore
├── README.md
├── environment.yml        # Conda environment specifications
├── fer2013.tar.gz         # Compressed dataset (FER-2013)
└── haarcascade_frontalface_default.xml # Haar Cascade classifier for face detection
```

**Key Components:**

* **fer2013.tar.gz:** The FER-2013 dataset, a publicly available dataset of facial expressions, is used for training and evaluating the models. It contains thousands of grayscale images of faces, each labeled with one of the seven emotion categories.
* **haarcascade_frontalface_default.xml:** This file contains the pre-trained Haar Cascade classifier, specifically designed for frontal face detection. It enables the system to locate and extract faces from images or video frames.
* **models.tar.gz:** (Optional) This archive contains pre-trained models (custom CNN, VGG16, ResNet50) ready for deployment.
* **functions.py:** This module houses a collection of Python functions that form the core of the application's functionality. It includes functions for:
    * Loading and pre-processing images and video frames
    * Implementing face detection using the Haar Cascade classifier
    * Defining and training various CNN architectures for emotion classification
    * Loading pre-trained models
    * Displaying emotions on images and video streams
* **gui.py:** This script builds the interactive graphical user interface using the Tkinter library. It allows users to:
    * Upload images for emotion detection
    * Activate the webcam for real-time emotion recognition
    * Quit the application

**Installation & Usage:**

1. **Clone the Repository:** `git clone https://github.com/your-username/facial-emotion-recognition.git`
2. **Navigate to the Directory:** `cd facial-expression-recognition`
3. **Create a Conda Environment (Recommended):** 
    * `conda env create -f environment.yml`
    * `conda activate facial-emotion-recognition`
4. **Extract the Dataset:** `tar -xvzf fer2013.tar.gz`
5. **(Optional) Extract Pre-trained Models:** `tar -xvzf models.tar.gz`
6. **Run the Application:** `python src/gui.py`

**Customization & Exploration:**

* **Model Training:** The `facial-emotion-detection.ipynb` Jupyter Notebook provides a detailed walkthrough of the model training process. Experiment with different CNN architectures, hyperparameters, and data augmentation techniques to optimize performance.
* **GUI Enhancements:** Tailor the graphical interface by adding features like:
    * Displaying emotion probabilities alongside predictions
    * Saving captured images or video recordings
    * Integrating additional emotion recognition models or datasets

**Potential Applications:**

* **Human-Computer Interaction:** Enhance user interfaces by adapting to emotional states.
* **Market Research:** Analyze customer reactions to products or advertisements.
* **Healthcare:** Assist in diagnosing and monitoring mental health conditions.
* **Robotics:** Develop robots with enhanced social intelligence. 

**Disclaimer:** This project is for educational and experimental purposes. Facial emotion recognition technology should be used responsibly and ethically, acknowledging its limitations and potential biases. 
