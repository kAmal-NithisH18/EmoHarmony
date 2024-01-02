# EmoHarmony
EmoHarmony is an application that uses real-time emotion detection to recommend music based on the user's current emotional state. The project integrates a Flask web server with a Convolutional Neural Network (CNN) model for emotion detection and a music is classified and stored in folder for recommendation.

## Features:

- Real-time Emotion Detection using a CNN model.
- Music Recommendation based on detected emotions.
- Web-based interface using Flask.

## Prerequisites:

- Python 3.x
- Required Python packages: Flask, OpenCV, NumPy, Keras, Librosa, Pillow

## Installation:

1. Clone the repository:

    ```bash
    git clone https://github.com/subash-s-07/EmoHarmony.git
    ```

2. Navigate to the project directory:

    ```bash
    cd EmoHarmony
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage:

1. Run the application:

    ```bash
    python app.py
    ```

2. Open your web browser and go to [http://localhost:5000](http://localhost:5000).

3. The camera will be accessed, and real-time emotion detection will begin. Recommended music will be displayed on the web interface.

## Screenshots:

<h2>Home Page</h2>
<img src="/Screenshots/1.png" alt="Home Page" width="700" />

<h2>About Model</h2>
<img src="/Screenshots/2.png" alt="About Model" width="700" />

<h2>How model Works</h2>
<img src="/Screenshots/3.png" alt="How model Works" width="700" />

<h2>Music Player</h2>
<img src="/Screenshots/4.png" alt="Music Player" width="700" />

<h2>Music Space</h2>
<img src="/Screenshots/6.png" alt="Music Space" width="700" />

<h2>Music Classifier</h2>
<img src="/Screenshots/5.png" alt="Music Classifier" width="700" />




## Folder Structure:

- `/static`: Contains static files such as CSS, images, and music.
- `/templates`: Contains HTML templates for the web interface.
- `/uploads`: Default folder for uploaded music files.
## EmoHarmony Model Description

### Overview

The EmoHarmony model is designed for real-time emotion detection using facial expressions and classifying emotions into seven categories: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise. The model is implemented using a Convolutional Neural Network (CNN) architecture and is trained on the FER2013 dataset.

### Model Architecture

The CNN architecture comprises several convolutional layers followed by max-pooling, batch normalization, and dropout layers to enhance its performance and prevent overfitting. The final layers consist of densely connected layers for emotion classification. Here is a summary of the model architecture:

1. Convolutional Layer (32 filters, kernel size: 3x3, activation: ReLU, input shape: 48x48x1)
2. Convolutional Layer (64 filters, kernel size: 3x3, activation: ReLU)
3. Batch Normalization
4. Max Pooling Layer (2x2)
5. Dropout Layer (25% dropout rate)

6. Convolutional Layer (128 filters, kernel size: 5x5, activation: ReLU)
7. Batch Normalization
8. Max Pooling Layer (2x2)
9. Dropout Layer (25% dropout rate)

10. Convolutional Layer (512 filters, kernel size: 3x3, activation: ReLU, regularization: L2)
11. Batch Normalization
12. Max Pooling Layer (2x2)
13. Dropout Layer (25% dropout rate)

14. Convolutional Layer (512 filters, kernel size: 3x3, activation: ReLU, regularization: L2)
15. Batch Normalization
16. Max Pooling Layer (2x2)
17. Dropout Layer (25% dropout rate)

18. Flatten Layer
19. Dense Layer (256 units, activation: ReLU)
20. Batch Normalization
21. Dropout Layer (25% dropout rate)

22. Dense Layer (512 units, activation: ReLU)
23. Batch Normalization
24. Dropout Layer (25% dropout rate)

25. Output Layer (7 units, activation: Softmax)

### Training

The model is trained using the Adam optimizer with a learning rate of 0.0001 and categorical cross-entropy as the loss function. Data augmentation is applied to the training set, including horizontal flips and shifts, to enhance the model's ability to generalize.

The training process is run for 60 epochs with a batch size of 64, and the model's performance is evaluated on both the training and validation sets.

### Evaluation

The training and validation accuracy and loss are visualized using Matplotlib. The final trained model is saved as 'model_optimal.h5'.

### Emotion Detection Example

An example of emotion detection is provided using a sample image from the test set. The image is loaded, preprocessed, and passed through the trained model to predict the emotion category. The predicted emotion label and the corresponding probability distribution are displayed.

### Results

After training, the model achieves a final training accuracy of []% and a validation accuracy of []%.

Feel free to explore and integrate this model into the EmoHarmony application for real-time emotion detection and music recommendation.
