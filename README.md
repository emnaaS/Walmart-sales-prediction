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
| Date | Week of sales |
| Weekly sales | Sales for the given store/week (target) |
| Holoday flag | Whether the week includes a public holiday |
| Temperature | Average regional temperature |
| Fuel_price | Cost of fuel in the region |
| CPI | Consumer Price Index |
| Unemployement | Regional unemployment rate |

## Experiment tracking
All experiments are logged and stored on DagsHub using MLflow:
- Logged parameters: n_estimators, train/test shapes
- Logged metrics: R², MSE, RMSE, MAE
- Logged artifacts: trained model

Dagshub repository: [emnaaS/Walmart_sales](https://dagshub.com/emnaaS/Walmart_sales)

## CI/CD pipeline
The project uses **GitHub Actions** to automate testing on every push. The workflow runs the **pytest** test suite (**test_pipeline.py**) to validate the pipeline steps before any changes are merged.

## Setup & Usage
```
 pip install -r requirements.txt
```

Key dependencies: *pandas*, *scikit-learn*, *mlflow*, *dagshub*, *seaborn*, *matplotlib*, *python-dotenv*

## Environment Variables
Create a .env file at the project root:
```
DagsHub_username=your_username
DagsHub_token=your_token
```

