from flask_cors import CORS
from flask import Flask, request, jsonify
from app import ExoplanetModel
from tess import ExoplanetModel_TESS
import pandas as pd
import numpy as np


app = Flask(__name__)
CORS(app)

exoplanet_model = ExoplanetModel()
exoplanet_model_tess = ExoplanetModel_TESS()

@app.route('/predict', methods=['POST'])
def predict_route():
    global exoplanet_model
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        
        if not user_input:
            return jsonify({'error': 'user_input is required'}), 400
        
        # Get prediction
        result = exoplanet_model.predict(user_input)
        
        if isinstance(result, dict):
            print(f"The predicted class for the input data is: {result['predicted_label']} (Code: {result['predicted_class']})")
            return jsonify(result)
        else:
            return jsonify({'error': result}), 400
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        # Predict for each row
        predictions = []
        for _, row in df.iterrows():
            result = exoplanet_model.predict(row.to_dict())
            predictions.append(result)
        print(predictions)
        return jsonify(predictions)

    elif file.filename.endswith('.json'):
        df = pd.read_json(file)
        # (Handle as above if needed)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    
@app.route('/predict_tess', methods=['POST'])
def predict_tess_route():
    global exoplanet_model_tess
    try:
        data = request.get_json()
        user_input = data.get('user_input')
        
        if not user_input:
            return jsonify({'error': 'user_input is required'}), 400
        
        # Get prediction
        result = exoplanet_model_tess.predict(user_input)
        
        if isinstance(result, dict):
            print(f"The predicted class for the input data is: {result['predicted_label']} (Code: {result['predicted_class']})")
            return jsonify(result)
        else:
            return jsonify({'error': result}), 400
            
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/upload_tess', methods=['POST'])
def upload_tess_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        # Predict for each row
        predictions = []
        for _, row in df.iterrows():
            result = exoplanet_model_tess.predict(row.to_dict())
            predictions.append(result)
        print(predictions)
        return jsonify(predictions)

    elif file.filename.endswith('.json'):
        df = pd.read_json(file)
        # (Handle as above if needed)
    else:
        return jsonify({"error": "Unsupported file type"}), 400
    
if __name__ == "__main__":
    print("----------------python file executing------------------")
    app.run(port=5000, debug=True)