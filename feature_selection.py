from sklearn.feature_selection import mutual_info_regression
import pandas as pd

def compute_mutual_info(df):
    X = df.drop(columns=['Weekly_Sales', 'Date'])
    y = df['Weekly_Sales']

    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'Feature': X.columns,
        'MI_Score': mi_scores
    }).sort_values('MI_Score', ascending=False).reset_index(drop=True)

    return mi_df


def compute_redundancy(df):
    X = df.drop(columns=['Weekly_Sales', 'Date'])

    features = X.columns
    redundancy_df = pd.DataFrame(index=features, columns=features, dtype=float)

    for col in features:
        mi = mutual_info_regression(X, X[col], random_state=42)
        redundancy_df.loc[col] = mi

    return redundancy_df


def select_features(df, mi_scores, relevance_threshold=0.01):
    # Drop features below relevance threshold
    low_relevance = mi_scores[mi_scores['MI_Score'] < relevance_threshold]['Feature'].tolist()

    # Drop redundant features manually based on MRMR analysis
    redundant = ['Fuel_Price', 'week', 'year', 'Holiday_Flag']

    features_to_drop = list(set(low_relevance + redundant))

    X = df.drop(columns=['Weekly_Sales', 'Date'] + features_to_drop, errors='ignore')
    y = df['Weekly_Sales']

    print("Dropped features:", features_to_drop)
    print("Remaining features:", X.shape[1])

    return X, y


def select_features_v2(df, mi_scores, relevance_threshold=0.01):
    # Drop redundant features based on MRMR analysis
    redundant = ['Fuel_Price', 'week', 'year']

    X = df.drop(columns=['Weekly_Sales', 'Date'] + redundant, errors='ignore')
    y = df['Weekly_Sales']

    print("Dropped features:", redundant)
    print("Remaining features:", X.shape[1])

    return X, y

def add_holiday_interactions(df):
    store_cols = [col for col in df.columns if col.startswith('Store_')]

    for store in store_cols:
        df[f'Holiday_{store}'] = df['Holiday_Flag'] * df[store]

    print(f"Added {len(store_cols)} interaction features")
    print(f"New shape: {df.shape}")

    return df