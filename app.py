import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.callback import TrainingCallback
from sklearn.metrics import mean_absolute_error, r2_score
from babel.numbers import format_currency

# --- Sidebar: Dataset Source Options ---
st.sidebar.header("Dataset Options")
dataset_source = st.sidebar.radio(
    "Select Dataset Source",
    ("Default (daily.csv)", "Upload your own dataset")
)
uploaded_file = None
if dataset_source == "Upload your own dataset":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# --- Sidebar: Gold Price Predictor Settings ---
st.sidebar.header("Gold Price Predictor Settings")
usd_to_inr = st.sidebar.number_input("USD to INR Conversion Rate", min_value=1.0, value=80.0, step=0.5)
local_premium = st.sidebar.number_input("Local Premium Factor", min_value=1.0, value=1.57, step=0.1)

# --- Sidebar: Data Expansion Option ---
data_option = st.sidebar.radio(
    "Select Data Option", 
    ("Use Original Data", "Expand Dataset with Generative AI")
)

# --- Main App Title ---
st.title("Prasun's Gold Price Prediction With Integrated GEN-AI (India)")

# --- Example Dataset Preview ---
with st.expander("Example Dataset Preview (daily.csv)", expanded=False):
    try:
        @st.cache_data
        def get_daily_preview():
            df = pd.read_csv('daily.csv', parse_dates=['Date'], thousands=',')
            # Rename USD to Close if applicable
            if "USD" in df.columns and "Close" not in df.columns:
                df.rename(columns={'USD': 'Close'}, inplace=True)
            return df.head(3)
        preview_df = get_daily_preview()
        st.write(preview_df)
    except Exception as e:
        st.warning("Could not load daily.csv for preview.")

# --- Date Input Widget ---
selected_date = st.date_input("Select Date", value=pd.Timestamp.today())

# --- Data Loading Function for Default Dataset ---
@st.cache_data
def get_data(end_date):
    """
    Load gold price data from a local CSV file ('daily.csv') up to the specified end_date.
    The CSV file is expected to have a 'Date' column and a 'USD' column (which is renamed to 'Close').
    """
    df = pd.read_csv('daily.csv', parse_dates=['Date'], thousands=',')
    if "USD" in df.columns and "Close" not in df.columns:
        df.rename(columns={'USD': 'Close'}, inplace=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df.dropna(subset=['Close'], inplace=True)
    df = df[df['Date'] <= pd.to_datetime(end_date)]
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
    expanded_df = expanded_df.sample(frac=1, random_state=42)  # Shuffle the data
    return expanded_df

# --- Custom Callback for XGBoost Training ---
class StreamlitProgressCallback(TrainingCallback):
    def __init__(self, progress_bar, status, total_rounds, start_progress=60, end_progress=80):
        self.progress_bar = progress_bar
        self.status = status
        self.total_rounds = total_rounds
        self.start_progress = start_progress
        self.end_progress = end_progress

    def after_iteration(self, model, epoch, evals_log):
        progress = (epoch + 1) / self.total_rounds
        new_progress = self.start_progress + int(progress * (self.end_progress - self.start_progress))
        self.progress_bar.progress(new_progress)
        self.status.info(f"Step 4: Training the model... (iteration {epoch + 1}/{self.total_rounds})")
        return False  # Continue training

# --- Main Application ---
if st.button("Predict"):
    status = st.empty()
    progress_bar = st.progress(0)
    
    # Format the selected date to a string (YYYY-MM-DD)
    end_date = pd.to_datetime(selected_date).strftime('%Y-%m-%d')

    # Step 1: Load Data
    status.info("Step 1: Loading data...")
    if dataset_source == "Upload your own dataset":
        if uploaded_file is None:
            st.error("Please upload a CSV file.")
            st.stop()
        else:
            try:
                raw_data = pd.read_csv(uploaded_file, parse_dates=['Date'], thousands=',')
            except Exception as e:
                st.error("Error reading the uploaded CSV file.")
                st.stop()
            # If the CSV has a 'USD' column instead of 'Close', rename it.
            if "USD" in raw_data.columns and "Close" not in raw_data.columns:
                raw_data.rename(columns={'USD': 'Close'}, inplace=True)
            raw_data['Close'] = pd.to_numeric(raw_data['Close'], errors='coerce')
            raw_data.dropna(subset=['Close'], inplace=True)
            raw_data = raw_data[raw_data['Date'] <= pd.to_datetime(end_date)]
            raw_data = raw_data[['Date', 'Close']]
    else:
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
                train_data = expand_features_dataset(train_data, expansion_factor=2)
            else:
                status.info("Step 3: Using original dataset...")
            progress_bar.progress(60)

            # Step 5: Model Training with Progress Updates using xgboost.train()
            status.info("Step 4: Training the model...")

            X_train = train_data.drop('Close', axis=1)
            y_train = train_data['Close']
            X_test = test_data.drop('Close', axis=1)
            y_test = test_data['Close']

            # Convert the training and testing data to DMatrix format.
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            params = {
                "objective": "reg:squarederror",
                "learning_rate": 0.05,
                "max_depth": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.7,
                "seed": 42
            }
            num_round = 200

            # Create an instance of our custom callback.
            progress_callback = StreamlitProgressCallback(progress_bar, status, total_rounds=num_round, start_progress=60, end_progress=80)

            booster = xgb.train(
                params,
                dtrain,
                num_boost_round=num_round,
                evals=[(dtest, "test")],
                callbacks=[progress_callback]
            )

            # Evaluate the model on the test set.
            y_pred = booster.predict(dtest)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            progress_bar.progress(80)

            # Step 6: Prediction using the latest available features.
            status.info("Step 5: Generating prediction...")
            # Use the DataFrame directly to preserve feature names.
            latest_features = original_processed_data.drop('Close', axis=1).iloc[[-1]]
            dlatest = xgb.DMatrix(latest_features)
            prediction_usd_per_ounce = booster.predict(dlatest)[0]
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
