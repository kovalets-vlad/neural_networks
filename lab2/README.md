# 🖼️ Convolutional Neural Networks (CNN) | Lab 2

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![Keras](https://img.shields.io/badge/Keras-3.0-red.svg)
![Computer Vision](https://img.shields.io/badge/Task-Computer--Vision-green.svg)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-orange.svg)

This project focuses on building an optimized **Convolutional Neural Network** for handwritten digit recognition. It includes an integrated **Data Augmentation** pipeline and extensive hyperparameter benchmarking to achieve maximum validation accuracy.

---

## 📌 Project Overview

The goal is to classify 28x28 grayscale images of handwritten digits (0-9).
Key highlights of this implementation:

- **Preprocessing:** Pixel normalization (1/255) and image reshaping.
- **On-the-fly Augmentation:** Integrated `RandomRotation`, `RandomZoom`, and `RandomTranslation` layers.
- **Architecture:** Multi-stage `Conv2D` + `MaxPool2D` with `Dropout` for regularization.
- **Analysis:** Comparing how kernel sizes, model depth, and augmentation intensity affect the final result.

---

## 🧪 Experiments & Benchmarking

The script systematically evaluates the following parameters:

### 1. Optimizer Race

Testing `Adam`, `RMSprop`, `SGD`, `Nadam`, and `Adamax` to find the most stable convergence for spatial data.

### 2. Kernel Size Impact

Comparing different receptive fields: **3x3, 5x5, 7x7, 9x9**. This experiment shows how the model captures local vs. global features.

### 3. Model Depth

Benchmarking a **Base (2-layer)** vs. **Deep (3-layer)** CNN architecture to see if more abstraction layers improve digit recognition.

### 4. Augmentation Stress-Test

Adjusting the `Augmentation Factor` (0.05, 0.1, 0.2) to find the balance between model robustness and training stability.

---

## 📊 Results & Visualization

Every training session produces two types of insights:

1. **Accuracy Curves:** Comparison of validation performance over epochs.
2. **Confusion Matrix:** Detailed breakdown of misclassified digits (e.g., identifying if the model confuses '4' with '9').

|             Training Comparison              |                    Confusion Matrix                    |
| :------------------------------------------: | :----------------------------------------------------: |
| ![Accuracy](results/accuracy_comparison.png) | ![Confusion Matrix](results/lab2/confusion_matrix.png) |

---

## 📁 Project Structure

```text
.
├── src/
│   └── cnn_mnist.py         # Main CNN implementation & experiments
├── input/
│   ├── train.csv            # Training data (ignored by git)
│   └── test.csv             # Test data (ignored by git)
├── results/lab2/            # Comparison plots and heatmaps
├── README.md
└── requirements.txt
```

## ⚙️ Installation & Usage

Prepare Data: Place train.csv and test.csv in the input/ folder.

Install Dependencies:

Bash
pip install -r requirements.txt
Run Full Analysis:

Bash
python src/cnn_mnist.py

## 💡 Key Findings

Data Augmentation: Small factors (0.05 - 0.1) significantly improve generalization, reducing the gap between training and validation accuracy.

Kernel Size: Larger kernels (7x7) can be useful in early layers for simple shapes, but 3x3 stacks are generally more efficient.

Dropout: A rate of 0.25 after pooling and 0.5 after dense layers is crucial to prevent the model from memorizing the training set.

## 📂 The Dataset: MNIST Digits

The model is trained on the **MNIST** (Modified National Institute of Standards and Technology) dataset, which is the "Hello World" of computer vision.

- **Content:** 70,000 grayscale images of handwritten digits (0-9).
- **Format:** Each image is **28x28 pixels**.
- **Visuals:** The digits are represented as white pixels (high intensity) on a black background (zero intensity).
- **Task:** Multiclass classification with 10 output nodes (one for each digit).

|                                      Sample Digits                                      |        Data Representation         |
| :-------------------------------------------------------------------------------------: | :--------------------------------: |
| ![MNIST Samples](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png) | **28x28 Matrix (0-255 grayscale)** |

> **Note:** In the preprocessing stage, we reshape the data to `(28, 28, 1)` and normalize pixel values to the `[0, 1]` range to improve gradient descent stability.
