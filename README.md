# Iris Flower Classification

An end-to-end machine learning application that classifies Iris flowers into Setosa, Versicolor, or Virginica using their sepal and petal measurements.

## Project Overview

This project demonstrates a complete machine learning workflow:

- Exploratory data analysis
- Data cleaning
- Train/test splitting
- Feature scaling using StandardScaler
- Logistic Regression and SVM comparison
- 5-fold cross-validation
- Model evaluation and error analysis
- Scikit-learn Pipeline for preprocessing and inference
- Model serialization using Joblib
- Separate prediction/inference layer
- Interactive Streamlit application
- Automated prediction tests

## Model Performance

| Model | Test Accuracy | 5-Fold CV Mean Accuracy |
|---|---:|---:|
| Logistic Regression | 93.33% | 96.00% |
| SVM | 96.67% | 96.67% |

SVM was selected as the final model.

On the test set, the final SVM correctly classified 29 out of 30 samples. The only misclassification was a Versicolor sample predicted as Virginica.

## Project Structure

    app.py                  Streamlit application
    predict.py              Model loading and inference
    train.py                Training and evaluation pipeline
    eda.py                  Exploratory data analysis
    iris.csv                Dataset
    iris_model.joblib       Serialized trained pipeline
    requirements.txt        Python dependencies
    tests/test_predict.py   Automated prediction tests

## Running the Project

Install dependencies:

    pip install -r requirements.txt

Run the Streamlit application:

    streamlit run app.py

Run tests:

    python -m pytest

## Tech Stack

Python, Pandas, scikit-learn, SVM, StandardScaler, Joblib, Streamlit, Pytest