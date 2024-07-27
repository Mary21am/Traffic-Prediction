import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from joblib import load
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Load model and scalers
feature_scaler = load('models/feature_scaler_FV_2.pkl')
target_scaler = load('models/target_scaler_FV_2.pkl')
model = load_model('models/final_ts_bilstm_model.h5', custom_objects={'mse': MeanSquaredError()})

def calculate_manhattan_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the Manhattan distance between two points (lat1, lon1) and (lat2, lon2).
    Convert the distance to miles.
    
    Parameters:
    - lat1, lon1: Coordinates of the first point.
    - lat2, lon2: Coordinates of the second point.
    
    Returns:
    - Manhattan distance in miles.
    """
    # Approximate conversion factors
    lat_to_miles = 69.0  # 1 degree latitude ~ 69 miles
    lon_to_miles = 54.6  # 1 degree longitude ~ 54.6 miles (varies by latitude)
    
    # Calculate Manhattan distance
    distance = (abs(lat1 - lat2) * lat_to_miles) + (abs(lon1 - lon2) * lon_to_miles)
    
    return distance

def prepare_dataset(df):
    """
    Prepares the dataset by extracting temporal features, calculating distance, and dropping unnecessary columns.

    Parameters:
    - df: The original DataFrame containing the dataset.

    Returns:
    - A DataFrame with the prepared dataset.
    """
    logging.info("Starting to prepare the dataset")
    start_time = time.time()
    
    # Convert trip_start_timestamp to datetime
    df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])

    # Calculate Manhattan distance
    df['trip_distance_miles'] = df.apply(lambda row: calculate_manhattan_distance(row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

    # Extract hour of the day (cosine and sine)
    df['hour_cos'] = np.cos(2 * np.pi * df['trip_start_timestamp'].dt.hour / 24)
    df['hour_sin'] = np.sin(2 * np.pi * df['trip_start_timestamp'].dt.hour / 24)

    # Extract day of the week (cosine and sine)
    df['day_cos'] = np.cos(2 * np.pi * df['trip_start_timestamp'].dt.dayofweek / 6)
    df['day_sin'] = np.sin(2 * np.pi * df['trip_start_timestamp'].dt.dayofweek / 6)

    # Extract quarter of the hour (cosine and sine)
    df['quarter_cos'] = np.cos(2 * np.pi * df['trip_start_timestamp'].dt.minute / 15)
    df['quarter_sin'] = np.sin(2 * np.pi * df['trip_start_timestamp'].dt.minute / 15)
    
    # Extract is weekend
    df['is_weekend'] = (df['trip_start_timestamp'].dt.dayofweek >= 5).astype(int)

    # Extract is morning rush hour (6-9 AM)
    df['is_morning_rush'] = ((df['trip_start_timestamp'].dt.hour >= 6) & (df['trip_start_timestamp'].dt.hour < 9)).astype(int)

    # Extract is evening rush hour (4-7 PM)
    df['is_evening_rush'] = ((df['trip_start_timestamp'].dt.hour >= 16) & (df['trip_start_timestamp'].dt.hour < 19)).astype(int)

    logging.info(f"Dataset prepared in {time.time() - start_time:.2f} seconds")
    
    return df

class PredictionDataPreparer:
    def __init__(self, feature_scaler, target_scaler, input_width):
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.input_width = input_width

    def prepare_data(self, data):
        logging.info("Starting to prepare the data for prediction")
        start_time = time.time()

        timestamps = data['trip_start_timestamp'].values
        feature_columns = [col for col in data.columns if col not in ['trip_start_timestamp']]
        logging.info(f"Feature columns: {feature_columns}")
        logging.info(f"Data before scaling: {data[feature_columns].head()}")
        
        data[feature_columns] = self.feature_scaler.transform(data[feature_columns])
        
        logging.info(f"Data after scaling: {data[feature_columns].head()}")

        sequence_data = []
        sequence_timestamps = []
        if len(data) >= self.input_width:
            for start_idx in range(len(data) - self.input_width + 1):
                end_idx = start_idx + self.input_width
                sequence_data.append(data.iloc[start_idx:end_idx][feature_columns].values)
                sequence_timestamps.append(timestamps[start_idx:end_idx])
        else:
            logging.warning("Not enough data to create sequences. Required: %d, Available: %d", self.input_width, len(data))
            return np.array([]), np.array([])

        sequence_data = np.array(sequence_data, dtype=np.float32)
        sequence_timestamps = np.array(sequence_timestamps)

        logging.info(f"Data prepared for prediction in {time.time() - start_time:.2f} seconds")
        
        return sequence_data, sequence_timestamps

def predict_travel_time(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, trip_start_timestamp):
    logging.info("Starting the prediction process")
    start_time = time.time()
    
    df = pd.DataFrame([{
        'pickup_latitude': pickup_latitude,
        'pickup_longitude': pickup_longitude,
        'dropoff_latitude': dropoff_latitude,
        'dropoff_longitude': dropoff_longitude,
        'trip_start_timestamp': trip_start_timestamp
    }])
    logging.info(f"Input DataFrame: {df}")
    
    df = prepare_dataset(df)
    logging.info(f"Prepared DataFrame: {df}")
    
    preparer = PredictionDataPreparer(feature_scaler, target_scaler, input_width=8)
    prepared_data, _ = preparer.prepare_data(df)
    logging.info(f"Prepared data for model: {prepared_data.shape}")
    
    # Check if prepared_data is empty
    if prepared_data.size == 0:
        logging.error("Prepared data is empty. Skipping prediction.")
        return None

    predictions = model.predict(prepared_data)
    logging.info(f"Raw model predictions: {predictions}")
    
    predictions_reshaped = predictions.reshape(-1, predictions.shape[2])
    original_scale_predictions = target_scaler.inverse_transform(predictions_reshaped)
    original_scale_predictions = original_scale_predictions.reshape(predictions.shape)
    
    logging.info(f"Prediction completed in {time.time() - start_time:.2f} seconds")
    
    return original_scale_predictions[-1, -1, 0]
