import joblib
import numpy as np
import pandas as pd
from config.paths_config import *
from config import *
from utils.common_functions import read_yaml
from flask import Flask, render_template, request
from src.logger import get_logger
from src.custom_exception import CustomException

app = Flask(__name__)
logger = get_logger(__name__)

# Load the trained models
try:
    standard_scaler_model = joblib.load(STANDARD_SCALER_PATH)
    ohe_model = joblib.load(OHE_ENCODER_PATH)
    prediction_model = joblib.load(MODEL_OUTPUT_PATH)
    logger.info("All models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise CustomException("Failed to load trained models")

@app.route("/", methods=['GET', 'POST'])
def index():
    prediction_result = None
    confidence_score = None
    error_message = None
    
    if request.method == 'POST':
        try:
            # Extract form data
            lead_time = int(request.form["lead_time"])
            no_of_adults = int(request.form["no_of_adults"])
            no_of_children = int(request.form["no_of_children"])
            no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
            no_of_week_nights = int(request.form["no_of_week_nights"])
            type_of_meal_plan = request.form["type_of_meal_plan"]
            required_car_parking_space = int(request.form["required_car_parking_space"])
            room_type_reserved = request.form["room_type_reserved"]
            arrival_year = int(request.form["arrival_year"])
            arrival_month = int(request.form["arrival_month"])
            arrival_date = int(request.form["arrival_date"])
            market_segment_type = request.form["market_segment_type"]
            repeated_guest = int(request.form["repeated_guest"])
            no_of_previous_cancellations = int(request.form["no_of_previous_cancellations"])
            no_of_previous_bookings_not_canceled = int(request.form["no_of_previous_bookings_not_canceled"])
            avg_price_per_room = float(request.form["avg_price_per_room"])
            no_of_special_requests = int(request.form["no_of_special_requests"])
            
            logger.info("Form data extracted successfully.")
            
            # Create input DataFrame with the same structure as training data
            input_data = pd.DataFrame({
                'no_of_adults': [no_of_adults],
                'no_of_children': [no_of_children],
                'no_of_weekend_nights': [no_of_weekend_nights],
                'no_of_week_nights': [no_of_week_nights],
                'type_of_meal_plan': [type_of_meal_plan],
                'required_car_parking_space': [required_car_parking_space],
                'room_type_reserved': [room_type_reserved],
                'arrival_year': [arrival_year],
                'arrival_month': [arrival_month],
                'arrival_date': [arrival_date],
                'market_segment_type': [market_segment_type],
                'repeated_guest': [repeated_guest],
                'lead_time': [lead_time],
                'no_of_previous_cancellations': [no_of_previous_cancellations],
                'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
                'avg_price_per_room': [avg_price_per_room],
                'no_of_special_requests': [no_of_special_requests]
            })
            
            logger.info("Input DataFrame created.")
            
            config = read_yaml(CONFIG_PATH)
            # Define categorical and numerical columns (same as in training)
            cat_cols = config['data_processing']['categorical_columns']
            num_cols = config['data_processing']['numerical_columns']

            # Apply One-Hot Encoding to categorical columns
            ohe_df = ohe_model.transform(input_data[cat_cols])
            ohe_features = pd.DataFrame(ohe_df, columns=ohe_model.get_feature_names_out(cat_cols), index=input_data.index)
            
            # Combine numerical columns with one-hot encoded features
            processed_data = pd.concat([input_data[num_cols], ohe_features], axis=1)
            
            logger.info("One-hot encoding applied successfully.")
            
            # Apply Standard Scaling to numerical columns
            # Only scale the numerical columns that exist in the processed data
            num_cols_to_scale = [col for col in num_cols if col in processed_data.columns]
            if num_cols_to_scale:
                processed_data[num_cols_to_scale] = standard_scaler_model.transform(processed_data[num_cols_to_scale])
            
            logger.info("Standard scaling applied successfully.")
            
            # Make prediction
            prediction_proba = prediction_model.predict_proba(processed_data)[0]
            prediction = prediction_model.predict(processed_data)[0]
            
            # Calculate confidence percentage
            confidence = max(prediction_proba) * 100
            
            # Interpret the result
            if prediction == 1:
                prediction_result = "This booking is likely to be CONFIRMED (Not Canceled)"
            else:
                prediction_result = "This booking is likely to be CANCELED"
            
            confidence_score = f"{confidence:.2f}%"
            
            logger.info(f"Prediction made successfully: {prediction_result} with {confidence_score} confidence.")
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            error_message = "An error occurred while processing your request. Please check your input and try again."
    
    return render_template('index.html', 
                         prediction=prediction_result, 
                         confidence=confidence_score,
                         error=error_message)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
