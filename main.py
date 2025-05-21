import pandas as pd
from src.load_data import load_price_data
from src.feature_engineering import add_features
from src.clean import clean_dataset
from src.label_gen import generate_labels
from src.logistic_regression import train_logistic_model
from src.predict import predict_next_day

df = load_price_data(ticker="MSFT")
if df.empty:
    print("Aborting pipeline: no data fetched.")
    exit()
df = add_features(df)  
df = clean_dataset(df)
df = generate_labels(df)
model = train_logistic_model(df)
#prediction, prob = predict_next_day(df)
#print(f"\n📊 Tomorrow's prediction → PUP = {prediction} | Confidence = {prob:.2%}")