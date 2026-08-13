# California Housing Price Prediction

## Project Overview

This project focuses on predicting California house prices using machine learning regression techniques.

The project uses the California Housing dataset and compares multiple regression models to identify the best-performing model based on MAE, RMSE and R² Score.

## Machine Learning Models

The following models were trained and evaluated:

- Linear Regression
- Ridge Regression
- Lasso Regression

## Project Structure

```text
California-Housing-Price-Prediction/
│
├── data/
│   └── california_housing.csv
│
├── notebooks/
│   └── California_Housing.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── linear_regression.pkl
│   ├── ridge_regression.pkl
│   └── lasso_regression.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── best-model.csv
│
├── requirements.txt
├── .gitignore
└── README.md
```
## Workflow

```text
Dataset
   ↓
Data Preprocessing
   ↓
Train/Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Best Model Selection
```

## Model Evaluation

The models were evaluated using the following metrics:

- **MAE (Mean Absolute Error)** – measures the average absolute difference between actual and predicted values.
- **RMSE (Root Mean Squared Error)** – measures prediction error while giving more weight to larger errors.
- **R² Score** – measures how much of the variation in house prices is explained by the model.

### Model Comparison

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 0.533200 | 0.745581 | 0.575788 |
| Ridge Regression | 0.533200 | 0.745581 | 0.575788 |
| Lasso Regression | 0.533152 | 0.744824 | 0.576650 |

### Best Performing Model

**Lasso Regression**

- MAE: **0.533152**
- RMSE: **0.744824**
- R² Score: **0.576650**

Lasso Regression currently provides the best performance among the three evaluated models based on the R² Score.

## Results

The `results/` directory contains:

- `model_comparison.csv` — comparison of all trained models.
- `best-model.csv` — evaluation metrics of the selected best-performing model.

## Requirements

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

```bash
python src/train.py
```

### 4. Evaluate the models

```bash
python src/evaluate.py
```

## Author

Pravash