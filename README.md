# Wildlife Detection and Deterrence System

## Overview

The Wildlife Detection and Deterrence System is an AI-powered solution designed to detect and classify wildlife animals in real time using Deep Learning and Computer Vision techniques. The system utilizes an ESP32-CAM module for image acquisition and a MobileNetV2-based Convolutional Neural Network (CNN) model for animal classification.

The primary objective of this project is to reduce human-wildlife conflicts by providing early detection and alert mechanisms when wild animals enter agricultural fields or residential areas.

## Key Features

* Real-time animal image capture using ESP32-CAM
* Wildlife classification using MobileNetV2 Transfer Learning
* Detection of 20 animal classes
* Automated alert and deterrence mechanism
* Lightweight and cost-effective deployment
* Suitable for agricultural and forest-border monitoring

## Technology Stack

* Python
* TensorFlow & Keras
* OpenCV
* NumPy
* Matplotlib
* ESP32-CAM
* Arduino IDE

## Dataset

The model was trained using a filtered animal image dataset derived from Kaggle's "90 Different Animals" dataset. Twenty animal categories were selected for training and evaluation.

## Model Performance

* Model: MobileNetV2 (Transfer Learning)
* Input Size: 224 × 224 RGB Images
* Validation Accuracy: 93.96%
* Loss Function: Categorical Cross-Entropy
* Optimizer: Adam

## System Workflow

1. ESP32-CAM captures animal images.
2. Images are preprocessed and sent to the trained model.
3. MobileNetV2 extracts features and predicts the animal class.
4. If a potentially dangerous wildlife animal is detected, the deterrence mechanism is activated.
5. Alerts are generated to notify nearby users.

## Applications

* Wildlife Intrusion Monitoring
* Crop Protection
* Forest Border Surveillance
* Rural Safety Systems
* Smart Agriculture

## Team Members

* K. Aarthi
* B. Afsana
* S. Arumuga Selvi
* P. Bhaveena
* G. A. Sri Varsha

## Acknowledgement

We express our sincere gratitude to the Department of Artificial Intelligence and Data Science, Mepco Schlenk Engineering College, for their guidance and support throughout the development of this project.

## Future Enhancements

* Real-time cloud-based monitoring
* SMS/Mobile App notifications
* Edge AI deployment
* Multi-camera surveillance support
* Species-specific deterrence strategies



---

Developed as a Mini Project for Deep Learning Laboratory, Department of Artificial Intelligence and Data Science, Mepco Schlenk Engineering College.
