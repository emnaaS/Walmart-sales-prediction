import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import sys
sys.path.append('..')
from data_engineering import encode_store
from feature_selection import select_features_v2
from processing import scale_features, scale_features
from modeling import train_model
from processing import split_data


# ── Fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture
def raw_data():
    return pd.read_csv('data/Walmart_Sales.csv')

@pytest.fixture
def encoded_data(raw_data):
    raw_data['Date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
    raw_data['month'] = raw_data['Date'].dt.month
    return encode_store(raw_data)

@pytest.fixture
def X_y(encoded_data):
    X, y = select_features_v2(encoded_data, mi_scores=None)
    return X, y

@pytest.fixture
def split(X_y):
    X, y = X_y
    return split_data(X, y)

@pytest.fixture
def scaled(split):
    X_train, X_test, y_train, y_test = split
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# ── Data Tests ──────────────────────────────────────────────────────────────────
def test_data_loads(raw_data):
    assert raw_data is not None
    assert len(raw_data) > 0

def test_no_missing_values(raw_data):
    assert raw_data.isnull().sum().sum() == 0

def test_no_duplicates(raw_data):
    assert raw_data.duplicated().sum() == 0

def test_expected_columns(raw_data):
    expected = ['Store', 'Date', 'Weekly_Sales', 'Holiday_Flag',
                'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
    assert list(raw_data.columns) == expected

def test_sales_are_positive(raw_data):
    assert (raw_data['Weekly_Sales'] > 0).all()

def test_store_count(raw_data):
    assert raw_data['Store'].nunique() == 45

def test_holiday_flag_binary(raw_data):
    assert set(raw_data['Holiday_Flag'].unique()).issubset({0, 1})

# ── Encoding Tests ──────────────────────────────────────────────────────────────
def test_encode_store_shape(encoded_data):
    assert encoded_data.shape[1] > 8

def test_encode_store_no_store_column(encoded_data):
    assert 'Store' not in encoded_data.columns

def test_store_dummies_binary(encoded_data):
    store_cols = [col for col in encoded_data.columns if col.startswith('Store_')]
    assert encoded_data[store_cols].isin([0, 1]).all().all()

# ── Feature Selection Tests ─────────────────────────────────────────────────────
def test_select_features_drops_date(X_y):
    X, y = X_y
    assert 'Date' not in X.columns

def test_select_features_drops_redundant(X_y):
    X, y = X_y
    for col in ['Fuel_Price', 'week', 'year']:
        assert col not in X.columns

def test_target_is_weekly_sales(X_y):
    X, y = X_y
    assert y.name == 'Weekly_Sales'

def test_no_missing_after_selection(X_y):
    X, y = X_y
    assert X.isnull().sum().sum() == 0

# ── Split Tests ─────────────────────────────────────────────────────────────────
def test_split_proportions(split):
    X_train, X_test, y_train, y_test = split
    total = len(X_train) + len(X_test)
    assert abs(len(X_test) / total - 0.2) < 0.01

def test_split_no_overlap(split):
    X_train, X_test, y_train, y_test = split
    assert len(set(X_train.index) & set(X_test.index)) == 0

# ── Scaling Tests ────────────────────────────────────────────────────────────────
def test_scaling_mean_near_zero(scaled):
    X_train_scaled, _, _, _, _ = scaled
    means = X_train_scaled[['Temperature', 'CPI', 'Unemployment']].mean()
    assert (means.abs() < 0.01).all()

def test_scaling_std_near_one(scaled):
    X_train_scaled, _, _, _, _ = scaled
    stds = X_train_scaled[['Temperature', 'CPI', 'Unemployment']].std()
    assert ((stds - 1).abs() < 0.01).all()

def test_scaler_not_fitted_on_test(scaled):
    X_train_scaled, X_test_scaled, _, _, _ = scaled
    # test set mean should NOT be zero since scaler was fit on train only
    test_means = X_test_scaled[['Temperature', 'CPI', 'Unemployment']].mean()
    assert not (test_means.abs() < 0.001).all()

# ── Model Tests ──────────────────────────────────────────────────────────────────
def test_model_trains(scaled):
    X_train_scaled, _, y_train, _, _ = scaled
    model = train_model(X_train_scaled, y_train)
    assert model is not None

def test_predictions_shape(scaled):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = scaled
    model = train_model(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    assert len(y_pred) == len(y_test)

def test_predictions_are_positive(scaled):
    X_train_scaled, X_test_scaled, y_train, y_test, _ = scaled
    model = train_model(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    assert (y_pred > 0).all()

def test_model_r2_above_threshold(scaled):
    from sklearn.metrics import r2_score
    X_train_scaled, X_test_scaled, y_train, y_test, _ = scaled
    model = train_model(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    assert r2 >= 0.90, f"R² too low: {r2:.4f}"