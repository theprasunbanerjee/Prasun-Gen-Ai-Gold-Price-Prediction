import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from babel.numbers import format_currency

# --- Sidebar Configuration ---
st.sidebar.header("Gold Price Predictor Settings")
usd_to_inr = st.sidebar.number_input("USD to INR Conversion Rate", min_value=1.0, value=80.0, step=0.5)
local_premium = st.sidebar.number_input("Local Premium Factor", min_value=1.0, value=1.57, step=0.1)

# Choose data source option
data_option = st.sidebar.radio(
    "Select Data Option", 
    ("Use Original Data", "Expand Dataset with Generative AI")
)

# --- Main App Title ---
st.title("Prasun's Gold Price Prediction (India)")

# --- Date Input Widget ---
selected_date = st.date_input("Select Date", value=pd.Timestamp.today())

# --- Data Download Function ---
@st.cache_data
def get_data(end_date):
    """Fetch gold price data up to the specified end_date."""
    data = yf.download('GC=F', start='2010-01-01', end=end_date)
    if 'Close' not in data.columns:
        return pd.DataFrame()
    return data[['Close']].reset_index()

# --- Data Preprocessing Function ---
def preprocess_data(df):
    df = df.copy()
    if 'Date' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    for i in range(1, 31):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# --- Synthetic Data Expansion Function ---
def expand_features_dataset(df, expansion_factor=2):
    synthetic_dfs = [df]
    num_synthetic_copies = expansion_factor - 1
    for _ in range(num_synthetic_copies):
        synthetic = df.copy()
        for col in synthetic.columns:
            if pd.api.types.is_numeric_dtype(synthetic[col]):
                noise_std = 0.05 * synthetic[col].std()
                synthetic[col] = synthetic[col] + np.random.normal(0, noise_std, size=len(synthetic))
        synthetic_dfs.append(synthetic)
    expanded_df = pd.concat(synthetic_dfs)
    expanded_df = expanded_df.sample(frac=1, random_state=42)
    return expanded_df

# --- Main Application ---
if st.button("Predict"):
    status = st.empty()
    progress_bar = st.progress(0)
    end_date = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

    # Step 1: Fetching Data
    status.info("Step 1: Fetching data...")
    raw_data = get_data(end_date)
    progress_bar.progress(20)
    
    if raw_data.empty:
        st.error(f"No data available up to {end_date}.")
    else:
        # Step 2: Preprocessing Data
        status.info("Step 2: Preprocessing data...")
        processed_data = preprocess_data(raw_data)
        progress_bar.progress(40)
        
        if processed_data.empty:
            st.error("Not enough data to generate features for prediction.")
        else:
            # Step 3: Temporal Split
            original_processed_data = processed_data.copy()
            train_size = int(0.8 * len(original_processed_data))
            train_data = original_processed_data.iloc[:train_size]
            test_data = original_processed_data.iloc[train_size:]

            # Step 4: Data Expansion
            if data_option == "Expand Dataset with Generative AI":
                status.info("Step 3: Expanding dataset with synthetic data...")
                train_data = expand_features_dataset(train_data, expansion_factor=2)
            else:
                status.info("Step 3: Using original dataset...")
            progress_bar.progress(60)

            # Step 5: Model Training
            status.info("Step 4: Training the model...")
            X_train = train_data.drop('Close', axis=1)
            y_train = train_data['Close']
            X_test = test_data.drop('Close', axis=1)
            y_test = test_data['Close']

            model = XGBRegressor(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.7,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            progress_bar.progress(80)

            # Step 6: Prediction
            status.info("Step 5: Generating prediction...")
            latest_features = original_processed_data.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
            prediction_usd_per_ounce = model.predict(latest_features)[0]
            prediction_inr_per_ounce = prediction_usd_per_ounce * usd_to_inr * local_premium
            formatted_price_per_ounce = format_currency(round(prediction_inr_per_ounce, 2), 'INR', locale='en_IN')
            progress_bar.progress(100)
            status.success("All steps completed!")

            # --- Display Results ---
            st.success(f"**Predicted Gold Price on {end_date}:**")
            st.write(f"**Price per ounce:** {formatted_price_per_ounce}")
            st.markdown("---")
            st.info(f"**Model Accuracy Metrics:** MAE = {mae:.2f} USD, R² = {r2:.2f}")

            if r2 >= 0.95:
                st.warning("Notice: High R² score detected. While this might indicate good performance, ensure there's no data leakage and consider real-world market volatility.")