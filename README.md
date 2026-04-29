# Weather Predictions

This project uses machine learning to predict future temperatures and fetches current/historical weather data for multiple locations using the Visual Crossing API.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**:
   - Create a `.env` file in the project root.
   - You can use `.env.example` as a template.
   - Add your Visual Crossing API key:
     ```env
     VISUAL_CROSSING_API_KEY=your_api_key_here
     ```

## Usage

Run the main script to start the interactive weather system:

```bash
python Weather_predictions.py
```

### Options:
1. **Predict future weather using ML**: Input humidity, pressure, wind speed, and wind direction to get a temperature prediction.
2. **Get weather for multiple locations**: Fetch current or historical weather data for multiple cities in a country.
3. **Exit**: Close the system.

## Data
The project uses `weather_data.csv` for training the machine learning model (Random Forest Regressor).
