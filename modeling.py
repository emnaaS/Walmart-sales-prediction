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