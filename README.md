# ECG Anomaly Detection Using Deep Learning

This repository contains a Deep Learning project that implements a semi-supervised **Autoencoder** to detect life-threatening cardiac anomalies in real-time ECG time-series data. By learning the mathematical distribution of a healthy heartbeat, the model can automatically flag dangerous arrhythmias based on reconstruction failure, serving as an automated triage system.

[Dataset on Kaggle (ECG5000)](https://www.kaggle.com/datasets/devavratkalyanpur/ecg5000)

## The Idea: Why Semi-Supervised Learning?

In real-world medical monitoring, normal heartbeats are abundant, but critical anomalies (like an R-on-T Premature Ventricular Contraction) are rare and unpredictable. Traditional supervised classification models struggle with this highly imbalanced data.

Instead of teaching the model what an anomaly looks like, this project flips the approach:

1. **Learn the Norm:** We train a Deep Autoencoder _exclusively_ on healthy heartbeats (Class 1). The model learns to compress and perfectly reconstruct this normal wave.
2. **Flag the Unknown:** When fed an abnormal heartbeat, the model mathematically fails to reconstruct the jagged, unexpected spikes.
3. **Medical Triage:** We calculate the Mean Absolute Error (MAE) of the reconstruction. If the error crosses a calculated threshold, the system triggers an anomaly alert.

## The Implementation

- **Data Preprocessing:** Filtered the dataset to isolate Normal beats (Class 1), highly critical R-on-T PVC beats (Class 2), and noisy Unclassified beats (Class 5). The data was Min-Max scaled to fit a `[0, 1]` distribution.
- **Architecture:** A 1D Dense Autoencoder funneling 140 time-steps down to an 8-dimensional bottleneck latent space, forcing non-linear dimensionality reduction.
- **Evaluation:** Established an anomaly threshold at the 95th percentile of normal training loss. During testing, the model successfully isolated Class 2 anomalies with a massive reconstruction error gap, proving its effectiveness for critical care alerts.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

## Requirements

To run this project, you will need **Python 3.8+** and the following libraries:

- `tensorflow` (Deep Learning framework)
- `scikit-learn` (Data splitting and preprocessing)
- `pandas` (Data manipulation)
- `numpy` (Numerical operations)
- `matplotlib` (Data visualization)

_A complete list of dependencies with specific versions is available in the `requirements.txt` file._

## Easy Setup & Installation

**1. Clone the Repository**
Open your terminal and run:

```bash
git clone https://github.com/Aditya-Kayasth/Anomalies-detection-Autoencoder.git
```
