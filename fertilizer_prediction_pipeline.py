import pickle
import numpy as np
import pandas as pd
import os

# Define paths
MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
ENCODERS_PATH = os.path.join('models', 'fertilizer_encoders.pkl')

# Load the trained model and encoders
try:
    with open(MODEL_PATH, 'rb') as file:
        fertilizer_model = pickle.load(file)
    with open(ENCODERS_PATH, 'rb') as file:
        encoders = pickle.load(file)
        fertilizer_name_encoder = encoders['fertilizer_name']
        feature_columns = encoders['feature_columns']
except FileNotFoundError:
    print(f"Error: Model or encoders file not found. Please run fertilizer_training_pipeline.py first.")
    fertilizer_model = None
    encoders = None
    feature_columns = []

def predict_fertilizer(data_dict):
    """
    Takes a dictionary of features, applies one-hot encoding, and returns the predicted fertilizer.
    
    Args:
        data_dict (dict): A dictionary with keys for user input.
        
    Returns:
        str: The predicted fertilizer name.
    """
    if fertilizer_model is None or not feature_columns:
        return "Fertilizer prediction model not loaded. Please train the model first."
        
    try:
        # --- Feature Preparation ---
        # Create a DataFrame from the input dictionary, matching the original feature names
        input_df = pd.DataFrame([data_dict])
        
        # Apply one-hot encoding
        input_encoded = pd.get_dummies(input_df)
        
        # Align the columns of the input data with the columns from the training data
        # This is a crucial step to ensure consistency.
        # It adds any missing dummy columns and fills them with 0.
        # It also ensures the column order is identical to what the model expects.
        final_input = input_encoded.reindex(columns=feature_columns, fill_value=0)
        
        # --- Prediction ---
        # The pipeline handles scaling internally before prediction
        prediction_encoded = fertilizer_model.predict(final_input)
        
        # Inverse transform the prediction to get the fertilizer name
        prediction_name = fertilizer_name_encoder.inverse_transform(prediction_encoded)
        
        return prediction_name[0]
    except Exception as e:
        print(f"An error occurred during fertilizer prediction: {e}")
        return f"Error: Could not make a prediction. Ensure all inputs are correct."