# Hate Speech Detector

A comprehensive solution to detect and filter hate speech using a trained machine learning model. This project integrates a CNN for text classification with a Flask API, a React-based web interface, and a browser extension for real-time hate speech detection.

---

## Features
- **Hate Speech Detection:** Uses NLP techniques and a CNN model to classify text into hate speech and non-hate speech categories.
- **Flask API:** A RESTful backend to serve the model and provide predictions.
- **React Web Interface:** A user-friendly platform for uploading text/commentaries for analysis.
- **Browser Extension:** Filters hate speech on web pages and blurs them dynamically.

---

## Technologies Used
- **Backend:** Python, Flask, TensorFlow, Keras, Scikit-learn.
- **Frontend:** React for the web interface.
- **Browser Extension:** JavaScript and HTML for filtering hate speech on web pages.

---

## Screenshots

### Flask API
![Flask API Example](server/flask_ui.png)

### React Web Interface
![React Web Interface](images/react_ui.png)

### Browser Extension
![Browser Extension Example](images/browser_demo.png)

---

## Installation and Usage

### **1. Clone the Repository**
```bash
git clone https://github.com/nkkb/hate-speech-detector.git
cd hate-speech-detector
