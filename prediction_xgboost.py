import requests
import pandas as pd
import numpy as np
from geopy.distance import distance as geopy_distance
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import joblib
import logging

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def prepare_dataset(df):
    """
    Prepares the dataset by extracting temporal features and dropping unnecessary columns.
    Parameters:
    - df: The original DataFrame containing the dataset.
    Returns:
    - A DataFrame with the prepared dataset.
    """
    # Convert trip_start_timestamp to datetime
    df['trip_start_timestamp'] = pd.to_datetime(df['trip_start_timestamp'])

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
    df['is_weekend'] = df['trip_start_timestamp'].dt.dayofweek >= 5
    df['is_weekend'] = df['is_weekend'].astype(int)  # Convert to 0 or 1

    # Extract is morning rush hour (6-9 AM)
    df['is_morning_rush'] = (df['trip_start_timestamp'].dt.hour >= 6) & (df['trip_start_timestamp'].dt.hour < 9)
    df['is_morning_rush'] = df['is_morning_rush'].astype(int)  # Convert to 0 or 1

    # Extract is evening rush hour (4-7 PM)
    df['is_evening_rush'] = (df['trip_start_timestamp'].dt.hour >= 16) & (df['trip_start_timestamp'].dt.hour < 19)
    df['is_evening_rush'] = df['is_evening_rush'].astype(int)  # Convert to 0 or 1

    return df

def get_road_distance_osrm(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
    """
    Calculate the road distance between two points using the OSRM API.
    """
    # Create a request to the OSRM API
    url = f"http://router.project-osrm.org/route/v1/driving/{pickup_longitude},{pickup_latitude};{dropoff_longitude},{dropoff_latitude}?overview=false"

    response = requests.get(url)
    data = response.json()

    # Extract the distance from the response
    if data['code'] == 'Ok':
        distance_meters = data['routes'][0]['distance']
        distance_miles = distance_meters * 0.000621371  # Convert meters to miles
        return distance_miles
    else:
        logging.error(f"Error: {data['code']}")
        return None

def predict_trip_details(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude, trip_start_timestamp):
    """
    Predict the trip details including distance and travel time.
    """
    # Create DataFrame for the new trip
    new_trip = pd.DataFrame({
        'pickup_latitude': [pickup_latitude],
        'pickup_longitude': [pickup_longitude],
        'dropoff_latitude': [dropoff_latitude],
        'dropoff_longitude': [dropoff_longitude],
        'trip_start_timestamp': [trip_start_timestamp]
    })

    # Prepare the dataset
    new_trip = prepare_dataset(new_trip)

    # Calculate road distance using the OSRM API
    new_trip['trip_distance_miles'] = new_trip.apply(lambda row: get_road_distance_osrm(
        row['pickup_latitude'], row['pickup_longitude'], row['dropoff_latitude'], row['dropoff_longitude']), axis=1)

    # Load the travel time scaler and model
    travel_time_scaler = joblib.load('models/xgboost/xgboost_tt_scaler_2.pkl')
    travel_time_model = joblib.load('models/xgboost/xgboost_tt_model_2.pkl')

    # Define feature columns
    feature_columns = [
        'pickup_latitude', 'pickup_longitude', 'dropoff_latitude',
        'dropoff_longitude', 'trip_distance_miles',
        'hour_cos', 'hour_sin', 'day_cos', 'day_sin', 'quarter_cos',
        'quarter_sin', 'is_weekend', 'is_morning_rush', 'is_evening_rush'
    ]

    # Scale features for travel time prediction
    scaled_features = travel_time_scaler.transform(new_trip[feature_columns])

    # Convert the scaled features to DMatrix for XGBoost
    dmatrix_features = xgb.DMatrix(scaled_features, feature_names=feature_columns)

    # Predict travel time
    predicted_travel_time = travel_time_model.predict(dmatrix_features)
    new_trip['predicted_trip_travel_time'] = predicted_travel_time

    # Round the predicted travel time to 3 decimal places
    new_trip['predicted_trip_travel_time'] = new_trip['predicted_trip_travel_time'].round(3)

    # Return the predictions as a dictionary
    return new_trip.to_dict('records')[0]
