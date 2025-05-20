import joblib
import pandas as pd

def predict_next_day(df, model_path="models/trained_model.pkl"):
    """
    Predict the PUP label (0/1) for the next day using today's features.
    Assumes df is already cleaned, labeled, and has all features.
    """
    # Load trained model
    model = joblib.load(model_path)

    # Define feature columns
    feature_cols = [col for col in df.columns if col not in ['Date', 'PUP']]

    # Get the last row (most recent features)
    latest_row = df[feature_cols].iloc[-1].values.reshape(1, -1)

    # Predict
    prediction = model.predict(latest_row)[0]
    prob = model.predict_proba(latest_row)[0][1]

    print(f"\n📈 Prediction for next day (PUP): {prediction} (Prob = {prob:.4f})")
    return prediction, prob
