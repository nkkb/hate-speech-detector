# Hate Speech Detector

A comprehensive solution to detect and filter hate speech using a trained machine learning model. This project leverages a Convolutional Neural Network (CNN) for text classification, integrated with a Flask API, a React-based web interface, and a browser extension to enhance user experience.

---

## Features
- **Hate Speech Detection:** Uses NLP techniques and a CNN model to classify text into hate speech and non-hate speech categories.
- **Flask API:** A RESTful backend to serve the model and provide predictions.
- **React Web Interface:** A user-friendly platform for uploading text/commentaries for analysis.
- **Browser Extension:** Filters hate speech on web pages and blurs them dynamically.
- **Comprehensive Reporting:** Includes accuracy metrics and a classification report.

---

## Technologies Used
- **Python Libraries:**
  - `pandas` and `numpy` for data manipulation and preprocessing.
  - `re` for text cleaning.
  - `tensorflow` and `keras` for building and training the CNN model.
  - `scikit-learn` for model evaluation.
- **Frontend:**
  - React for building the web interface.
- **Backend:**
  - Flask for serving the model via an API.
- **Browser Extension:**
  - JavaScript and HTML for detecting and filtering hate speech on web pages.

---

## Project Structure
hate-speech-detector/ ├── model/ │ ├── train_model.py # Model training script │ ├── saved_model/ # Directory for saved model weights │ ├── flask_api/ │ ├── app.py # Flask server script │ ├── requirements.txt # Python dependencies │ ├── react_website/ │ ├── src/ # React application source code │ ├── package.json # React dependencies │ ├── browser_extension/ │ ├── content_script.js # Main script for browser extension │ ├── manifest.json # Browser extension configuration │ ├── styles.css # Styling for blurred content │ ├── README.md # Project documentation └── images/ ├── flask_ui.png # Screenshot of Flask API ├── react_ui.png # Screenshot of React web interface ├── browser_demo.png # Screenshot of browser extension
