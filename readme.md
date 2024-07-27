# DDoS Botnet Attack Detection

This repository contains the implementation and analysis of various machine learning models for detecting DDoS botnet attacks, specifically focusing on DNS-based botnets. The project is part of a study conducted at the University of Engineering and Technology, Lahore.

## Project Overview

The primary aim of this project is to develop a machine learning-based approach to identify malicious traffic in IoT networks. The project explores various models, including Decision Tree Classifier and Bagging Classifier, to improve the detection accuracy of DDoS attacks.

### Key Objectives
- **Understanding Machine Learning Models**: Gain insights into the working of Decision Tree and Bagging Classifiers.
- **Model Evaluation**: Analyze the models based on metrics like accuracy, precision, recall, F1 score, and ROC curve.
- **Implementation**: Implement the models and perform cross-validation to ensure generalization.

## Dataset

The project utilizes the **IoT-23 Dataset**, which is designed for DDoS analysis, containing detailed packet features from IoT devices. The dataset was provided by the University of New Brunswick.

## Methodology

### Data Preprocessing
1. **Data Encoding**: Data was preprocessed using Min-Max Scaler for normalization.
2. **Data Splitting**: The dataset was split into 80% training and 20% testing sets.

### Machine Learning Models
- **Bagging Classifier**: An ensemble technique that improves accuracy and reduces overfitting by training multiple base models on bootstrapped data samples.
- **Decision Tree Classifier**: A model that uses a tree-like structure to make decisions based on input features.

## Evaluation Metrics

The models were evaluated using the following metrics:
- **Accuracy**: Measures the overall correctness of the model.
- **Precision**: Indicates the accuracy of the positive predictions.
- **Recall**: Measures the ability of the model to identify all relevant instances.
- **F1 Score**: A balanced metric between precision and recall.
- **ROC Curve**: A graphical representation of the model's performance at various threshold settings.
- **Confusion Matrix**: Provides a comprehensive overview of the model's performance.

## Results and Findings

- **Bagging Classifier**: Achieved an accuracy of 99.9%, demonstrating strong predictive performance.
- **Decision Tree**: Provided a clear, interpretable structure for decision-making, although it may suffer from overfitting.

## Limitations and Future Work

### Limitations
The study's primary limitation is the training of models on a specific subset of the dataset, which may lead to overfitting. Future work should include training on a more comprehensive dataset and applying the models in real-world IoT contexts.

### Future Enhancements
- **Model Expansion**: Train models on the entire dataset to potentially improve generalization.
- **Real-World Application**: Implement models in real-world IoT environments to evaluate practical feasibility.

## References

1. Tong Anh Tuan et al., "Performance evaluation of Botnet DDoS attack detection using machine learning," 2019. [Link](https://link.springer.com/article/10.1007/s12065-019-00310-w)
2. S. M, "DDoS Botnet Attack on IOT Devices," 2020. [Kaggle](https://www.kaggle.com/siddharthm1698/ddos-botnet-attack-on-iot-devices)
3. SOLARMAINFRAME, "IDS 2018 Intrusion CSVs (CSE-CIC-IDS2018)," 2021. [Kaggle](https://www.kaggle.com/solarmainframe/ids-intrusion-csv)
4. GitHub Repository: [Mahdi77N/DDoS-Botnet-Attack-Detection](https://github.com/Mahdi77N/DDoS-Botnet-Attack-Detection)

## Installation and Usage

### Prerequisites
- Python 3.x
- Required libraries: sklearn, pandas, matplotlib, etc.

### Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/Mahdi77N/DDoS-Botnet-Attack-Detection.git
cd DDoS-Botnet-Attack-Detection
pip install -r requirements.txt
