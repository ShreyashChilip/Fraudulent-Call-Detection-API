# Fraudulent Call Detection API

An API designed to detect fraudulent phone calls by analyzing audio inputs using machine learning models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dependencies](#dependencies)
- [License](#license)

## Overview

The Fraudulent Call Detection API processes audio recordings of phone calls to determine the likelihood of fraudulent activity. By leveraging advanced machine learning techniques, the API provides a probabilistic assessment, aiding in the prevention of scams and unauthorized activities.

## Features

- **Audio Analysis:** Processes audio files to extract relevant features indicative of fraudulent behavior.
- **Machine Learning Model Integration:** Utilizes pre-trained models to classify calls as fraudulent or legitimate.
- **API Interface:** Provides endpoints for submitting audio files and retrieving analysis results.

## Project Structure

```
Fraudulent-Call-Detection-API/
├── models/
│   ├── model.py
│   └── ...
├── venv/
├── app.py
├── index.html
├── requirements.txt
└── temp_audio.wav
```

- `models/`: Contains the machine learning models and related scripts.
- `venv/`: Virtual environment for managing project dependencies.
- `app.py`: Main application script that sets up the API endpoints.
- `index.html`: Frontend interface for interacting with the API.
- `requirements.txt`: Lists the Python dependencies required for the project.
- `temp_audio.wav`: Sample audio file for testing purposes.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/ShreyashChilip/Fraudulent-Call-Detection-API.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd Fraudulent-Call-Detection-API
   ```
3. **Set Up Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Application:**
   ```bash
   python app.py
   ```
2. **Access the Web Interface:** Open `http://localhost:5000` in a web browser.
3. **Analyze an Audio File:**
   - Use the provided interface to upload an audio file or submit a URL to an audio recording.
   - The API will process the audio and return a probabilistic assessment of its legitimacy.

## Model Details

- **Model Architecture:** The project employs a machine learning model trained to distinguish between fraudulent and legitimate calls based on audio features.
- **Training Data:** The model is trained on a dataset of labeled call recordings, encompassing various indicators of fraudulent activity.

## Dependencies

- **Flask:** Web framework for Python.
- **Librosa:** Library for analyzing and processing audio.
- **NumPy:** Library for numerical computations.
- **Scikit-learn:** Machine learning library for Python.
- **Other Dependencies:** Listed in `requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE).
