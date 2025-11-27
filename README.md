ASL Finger-Spelling Recognition System -

A real-time American Sign Language finger-spelling recognizer that uses MediaPipe Hand Tracking for landmark extraction and a neural network classifier trained on the Kaggle ASL alphabet dataset.
Runs entirely on CPU and provides live inference using your webcam.

Features-

Real-time ASL alphabet classification (A–Z, static letters only)
Hand landmark extraction using MediaPipe Hands
Lightweight neural network classifier trained on normalized 2D keypoints
Streaming webcam UI built with Streamlit
CPU-friendly deployment with low latency

System Architecture-

Webcam → MediaPipe Hands → Normalized Landmarks → MLP Classifier → Predicted Letter

Model Details:

Component	Choice-

Feature extractor	MediaPipe Hands
Input vector	42-D hand landmark features
Model	MLPClassifier (scikit-learn)
Hidden layers	(128, 64) ReLU
Classes	24 static ASL letters (J & Z omitted)
Accuracy	99% on held-out test set
Confusion matrix and evaluation included in training script.
