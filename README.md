# Emotion-Recognition-System
# Mutimodal Emotion Recognition System

**Course:**  AIT301 - Advanced Machine Learning (AML)

## Project Overview

This project is a **Multimodal Emotion Recognition System** that detects emotion from two different inputs:

* **Text Input** (Text Emotion Recognition)
* **Speech Input** (Speech Emotion Recognition)

The system applies Advanced Machine Learning (AML) approaches to analyze emotions from written text and spoken audio.

## Why Multimodal?

The system is considered **multimodal** because it processes more than one type of data:

* Text data
* Speech (audio) data

Each modality uses a different machine learning model to improve emotion recognition performance and robustness.

## Models Used

### Text Emotion Recognition

The text-based emotion recognition module adopts a hybrid and sequence-based approach, consisting of:

* Sequence-Based Text Modelling
* Rule-Based Emoji Detection
* LSTM (Long Short-Term Memory)
* Relative EA - LSTM (Custom Relative Multi-Head Attention Mechanism integrated with LSTM + EAC)

The rule-based component handles emoji-based emotional cues, while the LSTM model focuses on contextual and sequential textual features.

### Speech Emotion Recognition

The speech-based emotion recognition module uses a deep learning approaches tailored for audio signals:

* Convolutional Recurrent Neural Network (CRNN)
* Log-Mel Spectrogram Feature Extraction

The CRNN combines convolutional layers for spatial feature extraction and recurrent layers (LSTM) for temporal modeling of speech signals.

## Features

* Emotion prediction from text input
* Emotion prediction from speech
* Confidence score for prediction
* Web-based user interface with real-time interaction
* Support for audio file upload and browser-based audio recording

## Technologies Used

* **Python:** Model development and backend processing
* **TensorFlow / Keras:**  Text emotion recognition model development
* **PyTorch:** Speech emotion recognition model development
* **Flask:** Web-based application framework
* **Librosa:** Speech signal processing and feature extraction
* **HTML, CSS, JavaScript:** Frontend user interface
* **Web Audio API:** Browser-based audio recording

## How to Run

1. Navigate to the project folder

   ```
   cd emotion_recognition
   ```
2. (Optional) Create and activate virtual environment

   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install required libraries

   ```
   pip install flask tensorflow==2.20.0 librosa numpy soundfile nltk contractions nrclex torch
   ```
4. Run the application

   ```
   python app.py
   ```
5. Open in browser

   ```
   http://127.0.0.1:5000
   ```

## Usage

### Text Emotion Recognition

1. Enter text into the input box
2. Click **Analyze**
3. The predicted emotion and confidence score will be displayed

### Speech Emotion Recognition

1. Upload an audio file or Record audio using microphone
2. Click **Analyze**
3. The predicted emotion and confidence score will be displayed

## Notes

* Recommended Audio format: **WAV**
* Microphone permission must be enabled in the browser
* Confidence scores are derived from Softmax probablities
* Designed for **academic demonstration purposes** in AML

## Course Context

This project is developed for **AIT301 - Advanced Machine Learning (AML)** and demonstrates:

* Multimodal Learning
* Sequence Modelling
* Attention Mechanisms
* Hybrid rule-based and LSTM-based Approaches (Text)
* CRNN-based Deep Learning Model (Speech)
* Emotion Recognition from Text and Speech

## Authors

* Wong Hui Xuan
* Chee Hui Sheen
* Koh Zhi Ling
* Kong Yenly
* Pang Siao Xuan

**Course:** AIT301 - Advanced Machine Learning (AML)
