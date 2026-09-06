# Real-Time MNIST Digit Classifier

A real-time handwritten digit classifier built with a Convolutional Neural Network (CNN), Keras, and Streamlit.

## 🚀 Live Demo

👉 **[Try the app](https://live-mnist-classifier.streamlit.app/)**

Draw a handwritten digit directly in the browser and see the model's prediction, confidence, class probabilities, and the processed 28×28 image.

## 📌 Overview

This project demonstrates an end-to-end MNIST image classification pipeline:

- Loading and preprocessing the MNIST dataset
- Building a Convolutional Neural Network
- Training and evaluating the model
- Making predictions on individual images
- Preprocessing handwritten drawings
- Real-time inference with Streamlit
- Deploying the application to Streamlit Community Cloud

## 🧠 Model

The classifier uses a Convolutional Neural Network with the following architecture:

```text
Input: 28 × 28 × 1

Conv2D(32) → MaxPooling
Conv2D(64) → MaxPooling
Flatten
Dense(64)
Dense(10, softmax)
```

The model achieves approximately **98.95% test accuracy** on the MNIST test set.

## 🔄 Inference Pipeline

The handwritten drawing goes through the following preprocessing pipeline before being passed to the CNN:

```text
User Drawing
     ↓
Grayscale Conversion
     ↓
Digit Detection
     ↓
Bounding Box Detection
     ↓
Crop
     ↓
Resize
     ↓
Center on 28×28 Canvas
     ↓
Normalize
     ↓
CNN
     ↓
Prediction + Probabilities
```

This ensures that the user's drawing is transformed into a format similar to the images used during model training.

## 📊 Results

| Metric | Result |
| --- | ---: |
| Dataset | MNIST |
| Classes | 10 |
| Input Size | 28 × 28 × 1 |
| Test Accuracy | ~98.95% |

## 🛠️ Tech Stack

- **Python**
- **Keras**
- **PyTorch**
- **NumPy**
- **Pillow**
- **Streamlit**
- **streamlit-drawable-canvas**
- **uv**

## 📂 Project Structure

```text
live-mnist/
├── app.py
├── pyproject.toml
├── uv.lock
├── README.md
├── models/
│   └── mnist_cnn.keras
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── drawing_preprocessing.py
└── notebooks/
    └── 01_explore_mnist.ipynb
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/live-mnist.git
cd live-mnist
```

Install the dependencies:

```bash
uv sync
```

Run the Streamlit application:

```bash
uv run streamlit run app.py
```

The application will then be available at the local Streamlit URL shown in the terminal.

## 📈 Training

The CNN was trained on the MNIST training set using:

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Epochs:** 10
- **Batch Size:** 64
- **Validation Split:** 10%

The trained model is saved as:

```text
models/mnist_cnn.keras
```

## 🔮 Prediction

The trained model outputs a probability for each digit from `0` to `9`.

The application displays:

- Predicted digit
- Prediction confidence
- Probability for each class
- The processed 28×28 image sent to the CNN

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud**.

### Live Application

👉 **<https://live-mnist-classifier.streamlit.app/>**

## 📄 License

This project is for educational purposes.
