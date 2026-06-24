# Walmart Weekly Sales Prediction
A machine learning project to predict Walmart weekly sales, built following MLOps best practices; from data exploration to model training, experiment tracking, automated testing, and CI/CD deployment.
## Project overview
This project predicts weekly sales for 45 Walmart stores using historical data from Kaggle. It covers the full ML pipeline:
- Exploratory Data Analysis (EDA)
- Feature engineering (date decomposition, store encoding)
- Feature selection using MRMR (Mutual Information + Redundancy)
- Model comparison and selection
- Experiment tracking with MLflow via DagsHub
- Automated testing with pytest
- CI/CD pipeline via GitHub Actions

PS: No user-facing interface; the project is a backend ML pipeline designed to be run and reproduced end-to-end.

## Dataset
Source: [Walmart sales](https://www.kaggle.com/datasets/mikhail1681/walmart-sales/data) 
The dataset contains weekly sales data for 45 stores, along with macroeconomic features:

| Feature | Description |
| --------- | ----------|
| Store | Store number (1–45) |
