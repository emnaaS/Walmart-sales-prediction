# Walmart Weekly Sales Prediction
A machine learning project to predict Walmart weekly sales, built following MLOps best practices; from data exploration to model training, experiment tracking, automated testing, and CI/CD deployment.
##Project overview
This project predicts weekly sales for 45 Walmart stores using historical data from Kaggle. It covers the full ML pipeline:
- Exploratory Data Analysis (EDA)
- Feature engineering (date decomposition, store encoding, holiday interaction features)
- Feature selection using MRMR (Mutual Information + Redundancy)
- Model comparison and selection
- Experiment tracking with MLflow via DagsHub
- Automated testing with pytest
- CI/CD pipeline via GitHub Actions
