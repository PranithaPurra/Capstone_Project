# Titanic Analytics and Machine Learning

## Overview

This module performs exploratory data analysis and machine learning using the Titanic dataset.

The dataset is saved as `titanic.csv` as an offline fallback.

## Files

- analytics_01_EDA.ipynb
- analytics_02_modeling.ipynb
- titanic.csv

## EDA

- Dataset loading and profiling
- Missing-value analysis
- Age histogram and box plot
- Fare histogram and box plot
- IQR outlier detection
- Fare mean, median and mode
- Survival rate by sex
- Survival rate by passenger class
- Survival rate by sex and passenger class
- Age vs survival analysis
- Correlation matrix
- Correlation heatmap
- Exploratory z-score standardization

## Classification

Models used:
- Logistic Regression
- Decision Tree
- Random Forest

Evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC

## Class Imbalance

- Baseline
- class_weight='balanced'
- SMOTE

## Random Forest Tuning

GridSearchCV is used for Random Forest tuning with OOB score evaluation.

## Fare Regression

Fare regression includes MAE, RMSE, R² and Adjusted R², along with residual analysis.

## Model Persistence

The best classification pipeline is saved and reloaded using Joblib.