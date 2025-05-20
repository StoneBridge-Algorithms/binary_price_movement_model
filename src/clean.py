import pandas as pd

def clean_dataset(df):
    """
    Drops rows with any NaNs and resets the index.
    """
    df_cleaned = df.dropna().reset_index(drop=True)
    return df_cleaned
