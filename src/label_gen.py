def generate_labels(df):
    """
    Creates 'PUP' column: 1 if next day's return > 0, else 0.
    """
    df = df.copy()
    
    # Create label using Daily_Return (assumes this column already exists)
    df['PUP'] = (df['Daily_Return'] > 0).astype(int)

    # Shift label upward to align with today's features
    df['PUP'] = df['PUP'].shift(-1)

    # Drop last row (label now NaN)
    df = df.dropna(subset=['PUP']).reset_index(drop=True)

    # Ensure label is integer after shift/drop
    df['PUP'] = df['PUP'].astype(int)

    return df
