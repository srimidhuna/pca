Wine Quality Prediction App

This is a Gradio-based web application that predicts the quality of wine based on its physicochemical properties. It also provides the probability of each quality class if the model supports it.

Features

Predict wine quality (integer values 3–9).

Display probabilities for each quality class.

Uses PCA for dimensionality reduction and scaling for preprocessing.

Simple and interactive user interface with Gradio.

Requirements

Python 3.8+

Packages:

pandas

joblib

gradio

scikit-learn

Install dependencies with:

pip install pandas joblib gradio scikit-learn

Files

app.py – Main Gradio application.

best_model.pkl – Trained machine learning model for wine quality prediction.

preprocessing_artifacts.pkl – Contains scaler, pca, and feature_columns.

How to Run

Make sure all files (app.py, best_model.pkl, preprocessing_artifacts.pkl) are in the same directory.

Run the Gradio app:

python app.py


Open the provided local URL in your browser.

Input Features

Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

pH

Sulphates

Alcohol

Output

Predicted Wine Quality – The predicted quality score (3–9).

Class Probabilities – Probabilities for each possible quality class (if available).

Deployment

APP LINK : https://huggingface.co/spaces/srimidhuna/wine_quality
