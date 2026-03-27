import pandas as pd

def todate(df):
    df = df.copy()
    # convert to datetime
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")

    # extract time features
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week

    return df

def encode_store(df):
    df_encoded = pd.get_dummies(df, columns=['Store'], drop_first=True)
    print(df_encoded.shape)
    return df_encoded