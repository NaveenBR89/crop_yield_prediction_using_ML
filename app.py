from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import database
import json 

from prediction_pipeline import predict_crop
from yield_prediction_pipeline import predict_crop_yield
from fertilizer_prediction_pipeline import predict_fertilizer

app = Flask(__name__)
database.init_db()

# --- MODIFICATION: Use the AUGMENTED file for yield dropdowns ---
YIELD_DATA_PATH = os.path.join('data', 'crop_production.csv')
FERTILIZER_DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')

yield_df = pd.read_csv(YIELD_DATA_PATH).dropna(subset=["Production"])
fertilizer_df = pd.read_csv(FERTILIZER_DATA_PATH)

# Get unique values for dropdowns
unique_states = sorted(yield_df['State_Name'].unique())
unique_yield_seasons = sorted(yield_df['Season'].unique())
unique_yield_crops = sorted(yield_df['Crop'].unique())
# --- MODIFICATION: Now get Soil Type from the yield_df as well ---
unique_soil_types = sorted(yield_df['Soil Type'].unique())
unique_crop_types = sorted(fertilizer_df['Crop Type'].unique())


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/crop-recommend')
def crop_recommend():
    return render_template('crop.html')

@app.route('/yield-predict')
def yield_predict():
    return render_template(
        'yield.html',
        states=unique_states,
        seasons=unique_yield_seasons,
        crops=unique_yield_crops,
        soil_types=unique_soil_types
    )

@app.route('/get_districts/<state_name>')
def get_districts(state_name):
    districts = sorted(yield_df[yield_df['State_Name'] == state_name]['District_Name'].unique())
    return jsonify({'districts': districts})

@app.route('/fertilizer-recommend')
def fertilizer_recommend():
    return render_template(
        'fertilizer.html',
        soil_types=unique_soil_types,
        crop_types=unique_crop_types
    )
    
@app.route('/predict', methods=['POST'])
def predict():
    # ... (no changes) ...
    try:
        data = [float(request.form[key]) for key in ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        crop_prediction = predict_crop(data)
        database.log_crop_prediction(data, crop_prediction)
        return jsonify({'prediction': crop_prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/predict_yield', methods=['POST'])
def predict_yield_endpoint():
    if request.method == 'POST':
        try:
            # --- MODIFICATION: Pass ALL new fields to the prediction function ---
            data_for_prediction = {
                'Area': float(request.form['Area']),
                'State_Name': request.form['State_Name'],
                'District_Name': request.form['District_Name'],
                'Season': request.form['Season'],
                'Crop': request.form['Crop'],
                'Rainfall': float(request.form['Rainfall']),
                'Soil Type': request.form['Soil_Type']
            }
            
            yield_prediction = predict_crop_yield(data_for_prediction)
            database.log_yield_prediction(data_for_prediction, yield_prediction)
            
            return jsonify({'prediction': yield_prediction})
        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/predict_fertilizer', methods=['POST'])
def predict_fertilizer_endpoint():
    """Handles fertilizer recommendation requests."""
    if request.method == 'POST':
        try:
            # --- MODIFICATION: Match exact column names from the training data ---
            data_for_prediction = {
                'Temperature': float(request.form['Temperature']),
                'Humidity': float(request.form['Humidity']), # Removed trailing space from 'Humidity '
                'Moisture': float(request.form['Moisture']),
                'Soil Type': request.form['Soil_Type'],
                'Crop Type': request.form['Crop_Type'],
                'Nitrogen': int(request.form['Nitrogen']),
                'Potassium': int(request.form['Potassium']),
                'Phosphorous': int(request.form['Phosphorous'])
            }
            
            fertilizer_prediction = predict_fertilizer(data_for_prediction)
            
            database.log_fertilizer_prediction(data_for_prediction, fertilizer_prediction)
            
            return jsonify({'prediction': fertilizer_prediction})

        except Exception as e:
            return jsonify({'error': str(e)})

@app.route('/history')
def history():
    # ... (no changes) ...
    crop_logs = database.fetch_crop_logs()
    yield_logs = database.fetch_yield_logs()
    fertilizer_logs = database.fetch_fertilizer_logs()
    return render_template(
        'history.html', 
        crop_history=crop_logs,
        yield_history=yield_logs,
        fertilizer_history=fertilizer_logs
    )


@app.route('/delete_log/<log_type>/<int:log_id>', methods=['DELETE'])
def delete_log(log_type, log_id):
    """API endpoint to delete a single log entry."""
    try:
        database.delete_log_entry(log_type, log_id)
        return jsonify({'success': True, 'message': 'Log deleted successfully.'})
    except Exception as e:
        print(f"Error deleting log: {e}")
        return jsonify({'success': False, 'message': 'An error occurred.'}), 500

@app.route('/delete_all/<log_type>', methods=['DELETE'])
def delete_all(log_type):
    """API endpoint to delete all logs of a certain type."""
    try:
        database.delete_all_logs_for_type(log_type)
        return jsonify({'success': True, 'message': 'All logs deleted successfully.'})
    except Exception as e:
        print(f"Error clearing logs: {e}")
        return jsonify({'success': False, 'message': 'An error occurred.'}), 500

@app.route('/model_details')
def model_details():
    """Renders the model details page, loading data from the performance JSON file."""
    performance_data = {}
    # --- MODIFICATION: Update path to models directory ---
    performance_file_path = os.path.join('models', 'model_performance.json')
    try:
        with open(performance_file_path, 'r') as f:
            performance_data = json.load(f)
    except FileNotFoundError:
        print("Performance file not found. It will be created after the first training run.")
    
    return render_template('model_details.html', performance=performance_data)

# --- NEW ROUTE: Trigger Training ---
@app.route('/train_models', methods=['POST'])
def train_models_endpoint():
    """API endpoint to trigger the training of all models."""
    try:
        # Import the training functions here to avoid circular imports if structured differently
        from training_pipeline import train_crop_model
        from yield_training_pipeline import train_yield_model
        from fertilizer_training_pipeline import train_fertilizer_model
        
        print("--- Starting Crop Model Training ---")
        train_crop_model()
        print("--- Starting Yield Model Training ---")
        train_yield_model()
        print("--- Starting Fertilizer Model Training ---")
        train_fertilizer_model()
        
        return jsonify({'success': True, 'message': 'All models trained successfully.'})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)