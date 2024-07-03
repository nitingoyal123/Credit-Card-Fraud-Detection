# Credit Card Fraud Detection

## Overview

This project aims to build a machine learning model to detect fraudulent credit card transactions using a dataset from Kaggle. Given the highly imbalanced nature of the dataset, various techniques were employed to handle this imbalance effectively. The project involves using different machine learning algorithms and evaluating their performance.

## Dataset

The dataset used for this project is sourced from Kaggle and contains credit card transactions over a period of time. You can find the dataset [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Project Structure

1. Data Loading and Preprocessing
2. Handling Class Imbalance
   - Undersampling
   - Oversampling using SMOTE
3. Model Building and Evaluation
   - Logistic Regression
   - Decision Tree Classifier
   - Random Forest Classifier
4. Results Analysis

## Handling Class Imbalance

### Undersampling

Undersampling is performed by randomly selecting an equal number of samples from the majority class (non-fraudulent transactions) to match the number of samples in the minority class (fraudulent transactions).

```python
# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE

# Data Loading and Preprocessing
data = pd.read_csv('https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/creditcard.csv', on_bad_lines='skip')
data['Amount'] = StandardScaler().fit_transform(pd.DataFrame(data['Amount']))

def data_cleaning(data):
    data = data.drop_duplicates().drop('Time', axis=1)
    X = data.drop('Class', axis=1)
    Y = data['Class']
    return pd.concat([X, Y], axis=1).dropna(subset=[Y.name])

# Handling Class Imbalance
new_data = data_cleaning(data)
normal = new_data[new_data['Class'] == 0]
fraud = new_data[new_data['Class'] == 1]
normal_sample = normal.sample(fraud.shape[0])
new_data = pd.concat([normal_sample, fraud])

X_clean = new_data.drop('Class', axis=1)
Y_clean = new_data['Class']

X_train, X_test, Y_train, Y_test = train_test_split(X_clean, Y_clean, test_size=0.2, random_state=42)

# Oversampling using SMOTE
new_data = data_cleaning(data)
X = new_data.drop('Class', axis=1)
Y = new_data['Class']

X_res, Y_res = SMOTE().fit_resample(X, Y)
X_train_res, X_test_res, Y_train_res, Y_test_res = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)

# Model Building and Evaluation
param_grid = {
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__solver': ['lbfgs', 'liblinear']
    },
    'Decision Tree Classifier': {
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    },
    'Random Forest Classifier': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
}

def create_pipeline(clf):
    imputer = SimpleImputer()
    return Pipeline([
        ('imputer', imputer),
        ('classifier', clf)
    ])

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier()
}

def evaluate_model(Y_pred, Y_test, name):
    accuracy = accuracy_score(Y_pred, Y_test)
    precision = precision_score(Y_pred, Y_test)
    recall = recall_score(Y_pred, Y_test)
    f1 = f1_score(Y_pred, Y_test)
    
    print(f"\n============={name}============")
    print(f"\nAccuracy : {accuracy}")
    print(f"\nPrecision : {precision}")
    print(f"\nRecall : {recall}")
    print(f"\nF1 Score : {f1}")
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Undersampling Evaluation
undersampling_results = {}
print("Undersampling Results")
for name, clf in classifiers.items():
    grid_search = GridSearchCV(create_pipeline(clf), param_grid[name], cv=5, scoring='roc_auc')
    grid_search.fit(X_train, Y_train)
    Y_pred = grid_search.predict(X_test)
    undersampling_results[name] = evaluate_model(Y_pred, Y_test, name)

# Oversampling Evaluation
oversampling_results = {}
print("Oversampling Results")
for name, clf in classifiers.items():
    grid_search = GridSearchCV(create_pipeline(clf), param_grid[name], cv=5, scoring='roc_auc')
    grid_search.fit(X_train_res, Y_train_res)
    Y_pred = grid_search.predict(X_test_res)
    oversampling_results[name] = evaluate_model(Y_pred, Y_test_res, name)

# Results Analysis

## Undersampling Results

### Logistic Regression

- Accuracy: 0.974
- Precision: 0.962
- Recall: 0.981
- F1 Score: 0.971

### Decision Tree Classifier

- Accuracy: 0.966
- Precision: 0.962
- Recall: 0.962
- F1 Score: 0.962

### Random Forest Classifier

- Accuracy: 0.966
- Precision: 0.943
- Recall: 0.980
- F1 Score: 0.962

## Oversampling Results

### Logistic Regression

- Accuracy: 0.982
- Precision: 0.978
- Recall: 0.987
- F1 Score: 0.982

### Decision Tree Classifier

- Accuracy: 0.999
- Precision: 0.999
- Recall: 0.999
- F1 Score: 0.999

### Random Forest Classifier

- Accuracy: 0.999
- Precision: 1.0
- Recall: 0.999
- F1 Score: 0.999

