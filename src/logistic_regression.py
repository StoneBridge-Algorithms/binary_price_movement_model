import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_logistic_model(df, model_path="models/trained_model.pkl"):
    # Define features (exclude Date, PUP, etc.)
    feature_cols = [col for col in df.columns if col not in ['Date', 'PUP']]

    X = df[feature_cols]
    y = df['PUP']

    # Time-aware split (no shuffling)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predict & evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n🔍 Logistic Regression Accuracy: {acc:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved to {model_path}")

    return model
