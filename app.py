import streamlit as st
import pandas as pd
import numpy as np
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
st.title("Prasun's Gold Price Prediction With Integrated GEN-AI (India)")

# --- Date Input Widget ---
selected_date = st.date_input("Select Date", value=pd.Timestamp.today())

# --- Data Loading Function ---
@st.cache_data
def get_data(end_date):
    """
    Load gold price data from a local CSV file ('daily.csv') up to the specified end_date.
    The CSV file is expected to have the following headers (among others):
      Date, USD, EUR, JPY, GBP, CAD, ...
    Here we assume the USD column represents the gold price in USD per ounce.
    """
    # Read the CSV file. 'thousands' handles numbers with commas.
    df = pd.read_csv('daily.csv', parse_dates=['Date'], thousands=',')
    
    # Rename the USD column to 'Close' to be compatible with the rest of the code.
    df.rename(columns={'USD': 'Close'}, inplace=True)
    
    # Convert the 'Close' column to numeric (this will turn non-numeric entries like "#N/A" into NaN).
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    
    # Filter the dataframe to include only rows with Date <= selected end_date.
    df = df[df['Date'] <= pd.to_datetime(end_date)]
    
    # Return only the Date and Close columns (adjust if you need more columns)
    return df[['Date', 'Close']]

# --- Data Preprocessing Function ---
def preprocess_data(df):
    df = df.copy()
    if 'Date' not in df.columns:
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    # Create lag features (1 to 30 days) based on the 'Close' column.
    for i in range(1, 31):
        df[f'lag_{i}'] = df['Close'].shift(i)
    df.dropna(inplace=True)
    return df

# --- Synthetic Data Expansion Function ---
def expand_features_dataset(df, expansion_factor=2):
    """
    Creates synthetic copies of the data by adding Gaussian noise to numeric features.
    Returns a tuple: (expanded_dataset, synthetic_data_preview)
    where synthetic_data_preview is one synthetic copy for UI preview.
    """
    synthetic_dfs = []
    for _ in range(expansion_factor - 1):
        synthetic = df.copy()
        for col in synthetic.columns:
            if pd.api.types.is_numeric_dtype(synthetic[col]):
                noise_std = 0.05 * synthetic[col].std()
                synthetic[col] = synthetic[col] + np.random.normal(0, noise_std, size=len(synthetic))
        synthetic_dfs.append(synthetic)
    
    # Combine original data with synthetic copies.
    expanded_df = pd.concat([df] + synthetic_dfs)
    expanded_df = expanded_df.sample(frac=1, random_state=42)  # Shuffle the data
    
    # Use the first synthetic copy as the preview (if available).
    synthetic_preview = synthetic_dfs[0] if synthetic_dfs else None
    return expanded_df, synthetic_preview

# --- Main Application ---
if st.button("Predict"):
    status = st.empty()
    progress_bar = st.progress(0)
    
    # Format the selected date to a string (YYYY-MM-DD)
    end_date = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

    # Step 1: Load Data from CSV
    status.info("Step 1: Loading data from daily.csv...")
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

            # Step 4: Data Expansion (if chosen)
            if data_option == "Expand Dataset with Generative AI":
                status.info("Step 3: Expanding dataset with synthetic data...")
                train_data, synthetic_preview = expand_features_dataset(train_data, expansion_factor=2)
                # Display a preview of the synthetic data in the UI.
                st.markdown("### Synthetic Data Preview")
                st.dataframe(synthetic_preview.head(5))
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

            # Step 6: Prediction using the latest available features.
            status.info("Step 5: Generating prediction...")
            latest_features = original_processed_data.drop('Close', axis=1).iloc[-1].values.reshape(1, -1)
            prediction_usd_per_ounce = model.predict(latest_features)[0]
            # Convert USD prediction to INR using the provided conversion rate and local premium factor.
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
