import matplotlib.pyplot as plt
import seaborn as sns

def plot_mi_scores(mi_df):
    plt.figure(figsize=(12, 10))
    plt.barh(mi_df['Feature'], mi_df['MI_Score'])
    plt.xlabel('MI Score')
    plt.title('Mutual Information Scores')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_redundancy(redundancy_matrix):
    plt.figure(figsize=(16, 14))
    sns.heatmap(redundancy_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Redundancy Matrix (MI between features)')
    plt.tight_layout()
    plt.show()

def plot_predictions(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 — Actual vs Predicted scatter
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=10)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 color='red', linewidth=2, linestyle='--')
    axes[0].set_xlabel('Actual Sales')
    axes[0].set_ylabel('Predicted Sales')
    axes[0].set_title('Actual vs Predicted')

    # Plot 2 — Residuals
    residuals = y_test - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[1].axhline(0, color='red', linewidth=2, linestyle='--')
    axes[1].set_xlabel('Predicted Sales')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals Plot')

    plt.tight_layout()
    plt.show()

def plot_holiday_predictions(y_test, y_pred, X_test):
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    }, index=y_test.index)

    # Add month back from original df
    results_df['month'] = X_test['month'].values

    # Filter November and December
    holiday_months = results_df[results_df['month'].isin([11, 12])]

    plt.figure(figsize=(12, 5))
    plt.scatter(range(len(holiday_months)), holiday_months['Actual'],
                label='Actual', alpha=0.7, s=20)
    plt.scatter(range(len(holiday_months)), holiday_months['Predicted'],
                label='Predicted', alpha=0.7, s=20)
    plt.title('Actual vs Predicted — November & December only')
    plt.xlabel('Sample')
    plt.ylabel('Weekly Sales')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Print error metrics for those months only
    from sklearn.metrics import mean_absolute_error, r2_score
    print("Holiday months MAE:", f"${mean_absolute_error(holiday_months['Actual'], holiday_months['Predicted']):,.0f}")
    print("Holiday months R² :", f"{r2_score(holiday_months['Actual'], holiday_months['Predicted']):.4f}")