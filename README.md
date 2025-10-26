# Loan-Amount-Prediction

This repository contains a complete machine learning pipeline for predicting loan amounts based on
applicant demographics, financial details, housing conditions, and historical loan data.

Features
Descriptive Analysis: Summary statistics, missing value analysis, and correlation heatmaps
Exploratory Data Analysis (EDA): Visualizations for personal, financial, housing, and loan-related features

Feature Engineering:
debt_capacity = annual_income - 12 × monthly_expenses
income_per_occupant = annual_income / occupants_count
Categorical bucketing for rare levels


Preprocessing:
Imputation (median for numeric, constant for categorical)
Scaling and one-hot encoding
Modular pipelines using ColumnTransformer and Pipeline

Modeling:
ElasticNet, RandomForest, LightGBM, XGBoost
Hyperparameter tuning via GridSearchCV
Evaluation using MAE, RMSE, and R²

Visualization:
Distribution plots, box plots, and metric comparisons across models
