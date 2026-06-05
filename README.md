# Mental Health Risk Prediction

This project predicts Mental Health Risk using a KNN Machine Learning Model.

## Features

- Upload CSV dataset
- Automatic preprocessing
- Predict Mental Health Risk
- Download prediction results

## Project Structure

├── app.py
├── knn_model.pkl
├── knn_scaler.pkl
├── label_encoder.pkl
├── requirements.txt
└── README.md

## Installation

Clone repository

```bash
git clone <repository-url>
cd project-folder
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run Application

```bash
streamlit run app.py
```

## Input Dataset

Dataset should contain the same columns used during training.

Example:

- age
- gender
- occupation
- sleep_hours
- stress_level
- physical_activity_days
- social_support_score
- etc.

## Output

The model predicts:

- Low Risk
- Moderate Risk
- High Risk

Results can be downloaded as CSV.

## Model

Algorithm: K-Nearest Neighbors (KNN)

## Author

Pratiksha Renge
