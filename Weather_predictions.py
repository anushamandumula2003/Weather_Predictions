import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Visual Crossing API Configuration
API_KEY = os.getenv("VISUAL_CROSSING_API_KEY")
if not API_KEY:
    print("Warning: VISUAL_CROSSING_API_KEY not found in environment variables or .env file.")

BASE_URL = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/India?unitGroup=metric&key={API_KEY}&contentType=json"

# File paths for dataset and model
DATASET_FILE = 'weather_data.csv'  # Replace with your dataset path
MODEL_FILE = 'temperature_predictor_model.pkl'

# Weather Prediction Using ML
def train_weather_model():
    """Train a machine learning model for weather prediction."""
    if not os.path.exists(DATASET_FILE):
        print(f"Dataset file '{DATASET_FILE}' not found. Please provide a valid CSV file.")
        return None

    # Load historical weather data
    data = pd.read_csv(DATASET_FILE)
    data['Date'] = pd.to_datetime(data['Date'])

    # Verify dataset structure
    print("Dataset columns:", data.columns)

    # Define features and target
    # Update feature names based on actual dataset structure
    features = ['Humidity_%', 'Pressure_hPa', 'Wind_Speed_kmph', 'Wind_Direction_deg']
    target = 'Temperature_C'

    # Check if all required features exist in the dataset
    for feature in features:
        if feature not in data.columns:
            print(f"Error: Column '{feature}' not found in dataset. Please check the CSV file.")
            return None

    if target not in data.columns:
        print(f"Error: Target column '{target}' not found in dataset. Please check the CSV file.")
        return None

    X = data[features]
    y = data[target]

    # Handle missing values
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model trained with MAE: {mae}")

    # Save the model
    joblib.dump(model, MODEL_FILE)
    print("Model saved successfully!")
    return model


def predict_future_weather(input_data):
    """
    Predict future weather temperature using the trained model.
    - input_data: Dictionary containing input features ('Humidity_%', 'Pressure_hPa', 'Wind_Speed_kmph', 'Wind_Direction_deg').
    """
    if not os.path.exists(MODEL_FILE):
        return "Model not found. Please train the model first."

    # Load the trained model
    model = joblib.load(MODEL_FILE)

    # Prepare input data for prediction
    features = ['Humidity_%', 'Pressure_hPa', 'Wind_Speed_kmph', 'Wind_Direction_deg']
    input_values = [input_data.get(feature, np.nan) for feature in features]

    # Check for missing values in input
    if any(np.isnan(input_values)):
        return "Input data is missing some required features."

    # Predict temperature
    prediction = model.predict([input_values])[0]
    return round(prediction, 2)


# Load or train model
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
else:
    model = train_weather_model()

def fetch_weather_from_visual_crossing(date=None):
    """
    Fetch weather for India using the Visual Crossing API.
    - date: Date in 'YYYY-MM-DD' format (optional for current weather).
    """
    try:
        # Construct the URL with the date if provided
        if date:
            url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/India/{date}?unitGroup=metric&key={API_KEY}&contentType=json"
        else:
            url = BASE_URL  # Current weather URL

        response = requests.get(url)
        data = response.json()

        if response.status_code == 200:
            # Extract weather details
            if date:
                # Fetch weather details for the given date
                day_weather = data.get('days', [{}])[0]
                temp = day_weather.get('temp', 'N/A')
                humidity = day_weather.get('humidity', 'N/A')
                pressure = day_weather.get('pressure', 'N/A')
                weather_desc = day_weather.get('conditions', 'N/A')
            else:
                # Fetch current weather details
                current_weather = data.get('currentConditions', {})
                temp = current_weather.get('temp', 'N/A')
                humidity = current_weather.get('humidity', 'N/A')
                pressure = current_weather.get('pressure', 'N/A')
                weather_desc = current_weather.get('conditions', 'N/A')

            return {
                "Temperature (°C)": temp,
                "Humidity (%)": humidity,
                "Pressure (hPa)": pressure,
                "Weather Description": weather_desc
            }
        else:
            return {"Error": f"Failed to fetch weather data. Response: {response.text}"}
    except Exception as e:
        return {"Error": str(e)}

# Multi-location weather fetch
def fetch_weather_for_multiple_locations(locations, country, date=None):
    """
    Fetch weather for multiple locations in a specified country using the Visual Crossing API.
    - locations: List of location names (e.g., ["Delhi", "Mumbai", "Chennai"]).
    - country: Name of the country (e.g., "India").
    - date: Date in 'YYYY-MM-DD' format (optional for current weather).
    """
    results = {}
    for location in locations:
        try:
            # Construct the location query as "City, Country"
            location_query = f"{location},{country}"

            # Construct the URL for the specified location and date
            if date:
                url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_query}/{date}?unitGroup=metric&key={API_KEY}&contentType=json"
            else:
                url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location_query}?unitGroup=metric&key={API_KEY}&contentType=json"

            response = requests.get(url)
            data = response.json()

            if response.status_code == 200:
                # Extract weather details
                if date:
                    # Fetch weather details for the given date
                    day_weather = data.get('days', [{}])[0]
                    temp = day_weather.get('temp', 'N/A')
                    humidity = day_weather.get('humidity', 'N/A')
                    pressure = day_weather.get('pressure', 'N/A')
                    wind_speed = day_weather.get('windspeed', 'N/A')
                    wind_direction = day_weather.get('winddir', 'N/A')
                    weather_desc = day_weather.get('conditions', 'N/A')
                else:
                    # Fetch current weather details
                    current_weather = data.get('currentConditions', {})
                    temp = current_weather.get('temp', 'N/A')
                    humidity = current_weather.get('humidity', 'N/A')
                    pressure = current_weather.get('pressure', 'N/A')
                    wind_speed = current_weather.get('windspeed', 'N/A')
                    wind_direction = current_weather.get('winddir', 'N/A')
                    weather_desc = current_weather.get('conditions', 'N/A')

                # Store the weather data for the location
                results[location] = {
                    "Temperature (°C)": temp,
                    "Humidity (%)": humidity,
                    "Pressure (hPa)": pressure,
                    "Wind Speed (km/h)": wind_speed,
                    "Wind Direction (degrees)": wind_direction,
                    "Weather Description": weather_desc
                }
            else:
                results[location] = {"Error": f"Failed to fetch data. Response: {response.text}"}
        except Exception as e:
            results[location] = {"Error": str(e)}

    return results

# Interactive System
def query_weather_system():
    print("\nWelcome to the Weather Prediction and Query System!")
    while True:
        print("\nOptions:")
        print("1. Predict future weather using ML")
        print("2. Get weather for multiple locations in any country (current or historical)")
        print("3. Exit")

        choice = input("Enter your choice (1, 2, or 3): ")
        if choice == '1':
            print("\nEnter the input data for prediction:")
            try:
                humidity = float(input("Humidity (%): "))
                pressure = float(input("Pressure (hPa): "))
                wind_speed = float(input("Wind Speed (km/h): "))
                wind_direction = float(input("Wind Direction (degrees): "))
                input_data = {
                    "Humidity_%": humidity,
                    "Pressure_hPa": pressure,
                    "Wind_Speed_kmph": wind_speed,
                    "Wind_Direction_deg": wind_direction
                }
                prediction = predict_future_weather(input_data)
                print(f"Predicted Temperature (°C): {prediction}")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
        elif choice == '2':
            country = input("Enter the country : ").strip()
            locations = input(f"Enter locations in {country} separated by commas : ").split(',')
            locations = [loc.strip() for loc in locations]
            date = input("Enter the date (YYYY-MM-DD) or leave blank for current weather: ")
            weather_data = fetch_weather_for_multiple_locations(locations, country, date if date else None)
            print("\nWeather Data:")
            for location, data in weather_data.items():
                print(f"\nLocation: {location}, {country}")
                for key, value in data.items():
                    print(f"{key}: {value}")
        elif choice == '3':
            print("Exiting the system. Have a great day!")
            break
        else:
            print("Invalid choice. Please try again.")

# Run the system
if __name__ == "__main__":
    query_weather_system()
