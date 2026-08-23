import os
import joblib
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from s3_utils import upload_model_to_s3

def train_and_save_model(ticker: str) -> str:
    ticker = ticker.upper().strip()
    
    # 1. Fetch historical price data
    df = yf.download(ticker, start="2018-01-01", progress=False)
    if df.empty:
        raise ValueError(f"No price history found for symbol '{ticker}'.")
        
    # Handle multi-index columns if returned by newer yfinance versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df.dropna(inplace=True)

    X = df[['Close_lag1', 'Close_lag2', 'MA_5', 'MA_20']]
    y = df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 2. Ensure the directory exists before saving
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{ticker}_model.pkl"
    local_path = os.path.join(save_dir, filename)
    
    # 3. Save model locally
    joblib.dump(model, local_path)
    
    # 4. Upload artifact to Amazon S3
    s3_uploaded = upload_model_to_s3(local_path, f"models/{filename}")
    
    status_msg = f"Successfully trained model for '{ticker}'."
    if s3_uploaded:
        status_msg += " Model artifact persisted to Amazon S3."
    return status_msg