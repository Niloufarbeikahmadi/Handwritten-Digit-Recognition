# Handwritten-Digit-Recognition

📌 Project Overview

This repository contains an end-to-end implementation of Handwritten Digit Recognition using Deep Learning. It covers:

Binary Classification: Classifying between digits 0 and 1.

Multiclass Classification: Recognizing all digits (0-9).

Regularization: Preventing overfitting to improve model generalization.

🚀 Methods and Techniques

The project applies different deep learning techniques to solve the problem of digit classification efficiently.

1️⃣ Binary Classification (0 vs. 1)

Dataset: Uses a subset of the MNIST dataset containing digits 0 and 1.

Neural Network Implementation:

Implemented using both NumPy (manual forward propagation) and TensorFlow/Keras.

A simple feedforward neural network with ReLU activation and sigmoid output.

Loss Function: Binary Cross-Entropy.

Optimization: Stochastic Gradient Descent (SGD).

2️⃣ Multiclass Classification (0-9)

Dataset: Full MNIST dataset.

Neural Network Implementation:

Built using TensorFlow/Keras with multiple Dense layers.

Uses Softmax activation for multi-class classification.

Loss Function: Categorical Cross-Entropy.

Optimization: Adam Optimizer.

3️⃣ Overfitting Prevention with Regularization

To enhance model generalization, several regularization methods were applied:

L2 Regularization (Ridge Regression)

Dropout Layers to randomly deactivate neurons during training.

Batch Normalization for stable training and better convergence.

🔧 Technologies Used

Python
TensorFlow / Keras
NumPy
Matplotlib (for Visualization)
Jupyter Notebook

📊 Results & Insights

The binary classifier successfully differentiates between 0 and 1 with high accuracy.

The multiclass model generalizes well across all digits.

Regularization significantly improves performance on test data, reducing overfitting.

🛠 How to Run the Project
Clone this repository:

bash
Copy
Edit
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
Install required dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run Jupyter Notebooks to explore the models:
bash
Copy
Edit
jupyter notebook
Open and execute:
Handwritten Digit Recognition, Binary using NN.ipynb
Handwritten Digit Recognition, Multiclass.ipynb
Preventing Overfitting with Regularization.ipynb

📌 Key Takeaways

Deep Learning fundamentals applied to real-world classification tasks.

Implementation of Neural Networks from scratch using NumPy.

TensorFlow/Keras models with activation functions, optimizers, and regularization.

Practical solutions to overfitting in deep learning.

