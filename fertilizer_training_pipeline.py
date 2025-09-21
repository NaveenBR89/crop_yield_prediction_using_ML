import pandas as pd
import pickle
import os
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- File Paths ---
DATA_PATH = os.path.join('data', 'Fertilizer Prediction.csv')
MODEL_PATH = os.path.join('models', 'fertilizer_model.pkl')
ENCODERS_PATH = os.path.join('models', 'fertilizer_encoders.pkl')
PERFORMANCE_FILE = os.path.join('models', 'model_performance.json')

def train_fertilizer_model():
    """
    Trains a RandomForestClassifier for fertilizer recommendation using a robust
    pipeline with One-Hot Encoding for features and Label Encoding for the target.
    """
    print("Starting FERTILIZER model training with hyperparameter tuning...")
    
    data = pd.read_csv(DATA_PATH)
    data.rename(columns={
        "Temparature": "Temperature",
        "Humidity ": "Humidity"
    }, inplace=True)
    
    # Target Variable Encoding
    fertilizer_name_encoder = LabelEncoder()
    data["Fertilizer Name"] = fertilizer_name_encoder.fit_transform(data["Fertilizer Name"])

    # Feature Engineering
    X = data.drop("Fertilizer Name", axis=1)
    y = data["Fertilizer Name"]
    X_encoded = pd.get_dummies(X, columns=['Soil Type', 'Crop Type'])
    
    feature_columns = X_encoded.columns.tolist()
    encoders = {
        'fertilizer_name': fertilizer_name_encoder,
        'feature_columns': feature_columns
    }

    # Split the data, using stratify to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)

    # --- FIX: Set a smaller k_neighbors for SMOTE ---
    # The error indicates some classes have very few samples in the CV folds.
    # Setting k_neighbors=3 is a safe value that is less than the error-causing sample size of 4.
    pipeline = ImbPipeline(steps=[
        ('smote', SMOTE(k_neighbors=3, random_state=42)),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [10, 20]
    }
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    
    print("Running GridSearchCV for Fertilizer model...")
    grid_search.fit(X_train, y_train)
    
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_

    y_pred = best_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Best Fertilizer Model Accuracy: {accuracy:.4f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(best_pipeline, file)
    with open(ENCODERS_PATH, 'wb') as file:
        pickle.dump(encoders, file)
    print(f"Fertilizer pipeline and encoders saved.")

    performance_data = {}
    if os.path.exists(PERFORMANCE_FILE):
        with open(PERFORMANCE_FILE, 'r') as f:
            performance_data = json.load(f)
            
    performance_data['fertilizer_recommendation'] = {
        'model_type': 'Random Forest Classifier with SMOTE',
        'accuracy': f"{accuracy:.4f}",
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'best_parameters': best_params,
        'last_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(PERFORMANCE_FILE, 'w') as f:
        json.dump(performance_data, f, indent=4)
    print(f"Fertilizer model performance saved to {PERFORMANCE_FILE}")

    return performance_data['fertilizer_recommendation']

if __name__ == '__main__':
    train_fertilizer_model()