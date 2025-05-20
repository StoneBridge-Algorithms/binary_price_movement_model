import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import entropy as scipy_entropy

# === Custom Functions === #

def hurst_exponent(ts):
    lags = range(2, 20)
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]

def petrosian_fd(x):
    diff = np.diff(x)
    N = len(x)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes)))

def fisher_transform(x):
    # Apply to scaled returns ([-1, 1] safe range)
    x = np.clip(x, -0.999, 0.999)
    return 0.5 * np.log((1 + x) / (1 - x))

def rolling_entropy(series, window):
    return series.rolling(window).apply(lambda x: scipy_entropy(np.histogram(x, bins=5)[0] + 1), raw=False)

def rolling_hurst(series, window):
    return series.rolling(window).apply(hurst_exponent, raw=False)

def rolling_petrosian(series, window):
    return series.rolling(window).apply(petrosian_fd, raw=False)

# === Main Feature Engineering === #

def add_features(df):
    df = df.copy()
    df.sort_values("Date", inplace=True)

    # --- Price Derivatives ---
    df['Daily_Return'] = df['Close'].pct_change()
    df['Velocity'] = df['Close'].diff()
    df['Acceleration'] = df['Velocity'].diff()

    # --- Autocorrelation ---
    def autocorr(x):
        return x.autocorr(lag=1)
    df['Autocorrelation_1'] = df['Daily_Return'].rolling(5).apply(autocorr, raw=False)

    # --- Petrosian Fractal Dimension ---
    df['Petrosian_FD'] = rolling_petrosian(df['Close'], window=20)

    # --- Fisher Transform on scaled returns ---
    df['Fisher_Transform'] = fisher_transform(df['Daily_Return'])

    # --- Savitzky-Golay Filter ---
    try:
        df['SG_Filtered'] = savgol_filter(df['Close'], window_length=11, polyorder=2)
    except:
        df['SG_Filtered'] = np.nan

    # --- Rolling Entropy over returns ---
    df['Rolling_Entropy_5'] = rolling_entropy(df['Daily_Return'], window=5)

    return df
