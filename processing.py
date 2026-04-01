from sklearn.model_selection import train_test_split
import pandas as pd

def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)

    return X_train, X_test, y_train, y_test


from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    print("Scaling done!")
    print("X_train mean (sample):", X_train_scaled[['Temperature', 'CPI', 'Unemployment']].mean().round(5).to_dict())
    print("X_train std  (sample):", X_train_scaled[['Temperature', 'CPI', 'Unemployment']].std().round(5).to_dict())

    return X_train_scaled, X_test_scaled, scaler
