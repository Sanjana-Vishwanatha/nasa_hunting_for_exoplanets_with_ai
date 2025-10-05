from flask_cors import CORS
from flask import Flask, request, jsonify, send_file
from kepler import ExoplanetModel
from tess import ExoplanetModel_TESS
import pandas as pd
import numpy as np
from io import BytesIO


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
    
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file type. Use .csv or .json"}), 400
        
        # Predict for each row and collect results
        predicted_classes = []
        predicted_labels = []
        
        for _, row in df.iterrows():
            result = exoplanet_model.predict(row.to_dict())
            predicted_classes.append(result['predicted_class'])
            predicted_labels.append(result['predicted_label'])
        
        # Add prediction columns to the original dataframe
        df['predicted_class'] = predicted_classes
        df['predicted_label'] = predicted_labels
        
        # Convert dataframe to CSV in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        print(f"Processed {len(df)} Kepler predictions")
        
        # Return CSV file
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='kepler_predictions.csv'
        )
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
@app.route('/predict_tess', methods=['POST'])
def predict_tess_route():
    print("TESS predict route called")
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
    
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file)
        else:
            return jsonify({"error": "Unsupported file type. Use .csv or .json"}), 400
        
        # Predict for each row and collect results
        predicted_classes = []
        predicted_labels = []
        
        for _, row in df.iterrows():
            result = exoplanet_model_tess.predict(row.to_dict())
            predicted_classes.append(result['predicted_class'])
            predicted_labels.append(result['predicted_label'])
        
        # Add prediction columns to the original dataframe
        df['predicted_class'] = predicted_classes
        df['predicted_label'] = predicted_labels
        
        # Convert dataframe to CSV in memory
        output = BytesIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        print(f"Processed {len(df)} TESS predictions")
        
        # Return CSV file
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='tess_predictions.csv'
        )
        
    except Exception as e:
        print(f"Error processing TESS file: {str(e)}")
        return jsonify({"error": str(e)}), 500

    
if __name__ == "__main__":
    print("----------------python file executing------------------")
    app.run(host='0.0.0.0', debug=True)