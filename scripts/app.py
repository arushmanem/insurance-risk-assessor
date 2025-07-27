from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Get absolute path to models folder
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

def load_model_artifacts():
    """Load the trained model and scaler with absolute paths"""
    artifacts = {}
    
    # Define absolute paths
    model_path = os.path.join(MODELS_DIR, 'random_forest_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    
    print(f"Looking for model at: {model_path}")
    print(f"Looking for scaler at: {scaler_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing at {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file missing at {scaler_path}")
    
    print("Files found, loading...")
    with open(model_path, 'rb') as f:
        artifacts['model'] = pickle.load(f)
        
    with open(scaler_path, 'rb') as f:
        artifacts['scaler'] = pickle.load(f)
        
    return artifacts

# Load model artifacts
try:
    ml_artifacts = load_model_artifacts()
    print("✅ Successfully loaded model artifacts!")
except Exception as e:
    print(f"❌ Error loading model artifacts: {str(e)}")
    raise

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Validate input
        data = request.json
        required_fields = ['age', 'bmi', 'children', 'smoker', 'region']
        for field in required_fields:
            if field not in data:
                raise BadRequest(f"Missing required field: {field}")
        
        # 2. Prepare the input data
        input_data = pd.DataFrame([{
            'age': float(data['age']),
            'bmi': float(data['bmi']),
            'children': int(data['children']),
            'smoker': str(data['smoker']).lower().strip(),
            'region': str(data['region']).title().strip(),
            'age_group': data.get('age_group', 'middle_aged'),  # default if not provided
            'bmi_category': data.get('bmi_category', 'normal')  # default if not provided
        }])
        
        # 3. Feature engineering (same as training)
        input_data['smoker_age_interaction'] = input_data['age'] * (input_data['smoker'] == 'yes')
        input_data['bmi_age_interaction'] = input_data['bmi'] * input_data['age']
        
        # 4. Convert categoricals to dummies
        input_data = pd.get_dummies(input_data, 
                                 columns=['smoker', 'region', 'age_group', 'bmi_category'], 
                                 drop_first=True)
        
        # 5. Ensure all expected columns exist
        expected_columns = [
            'age', 'bmi', 'children',
            'smoker_yes', 'region_northwest', 'region_southeast', 'region_southwest',
            'age_group_middle_aged', 'age_group_senior', 'age_group_elderly',
            'bmi_category_normal', 'bmi_category_overweight', 'bmi_category_obese',
            'smoker_age_interaction', 'bmi_age_interaction'
        ]
        
        # Add missing columns with 0
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
                
        # 6. Scale numerical features
        numerical_cols = ['age', 'bmi', 'children', 'smoker_age_interaction', 'bmi_age_interaction']
        input_data[numerical_cols] = ml_artifacts['scaler'].transform(input_data[numerical_cols])
        
        # 7. Make prediction
        prediction = ml_artifacts['model'].predict(input_data[expected_columns])
        
        return jsonify({
            'prediction': float(prediction[0]),
            'status': 'success',
            'message': 'Prediction successful'
        })
        
    except BadRequest as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f"Prediction failed: {str(e)}"
        }), 500

@app.route('/')
def home():
    return """
    <h1>Insurance Cost Predictor</h1>
    <p>Use the /predict endpoint with POST request</p>
    <p>Example curl command:</p>
    <pre>
    curl -X POST http://localhost:5001/predict \\
    -H "Content-Type: application/json" \\
    -d '{
        "age":35,
        "bmi":28,
        "children":2,
        "smoker":"yes",
        "region":"northeast"
    }'
    </pre>
    """

if __name__ == '__main__':
    app.run(debug=True, port=5001, host='0.0.0.0')