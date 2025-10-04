import pandas as pd
import numpy as np
import joblib
from flask_cors import CORS
from flask import Flask, request, jsonify
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)
CORS(app)

class ExoplanetModel_TESS:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_name = "./models/random_forest_exoplanet_model_tess.joblib"
        # Store median values for filling nulls
        self.median_values = {}
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        self.data_cleaning()
        self.scale_data()
        self.load_model()

    def load_data(self):
        self.tess_data = pd.read_csv(self.data_path, skiprows=69)
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Drop the unnecessary columns like TIC ID, TESS Name, etc.
        self.tess_data = self.tess_data.drop(columns=['toi', 'tid', 'toi_created', 'rowupdate'])  # Drop the columns that are not needed
        
        # Convert the 'koi_disposition' column to categorical type
        self.tess_data['tfopwg_disp'] = self.tess_data['tfopwg_disp'].astype('category')
        self.categories = self.tess_data['tfopwg_disp'].cat.categories

        # Map the categories to numerical codes and replace the original column 0 -> 'CANDIDATE', 2 -> 'FALSE POSITIVE', 1 -> 'CONFIRMED'
        self.tess_data['tfopwg_disp'] = self.tess_data['tfopwg_disp'].cat.codes

        print("Data preprocessing completed.")

    def data_cleaning(self):
        # Check any character columns and convert them to categorical type
        for column in self.tess_data.select_dtypes(include=['object']).columns:
            print(f"Column '{column}' is of type 'object'")
            self.tess_data[column] = self.tess_data[column].astype('category')
            self.tess_data[column] = self.tess_data[column].cat.codes

        # Store median values before filling
        self.median_values = self.tess_data.median(numeric_only=True).to_dict()

        # Fill null values with median values
        self.tess_data.fillna(self.median_values, inplace=True)
        print("Data cleaning completed.")

    def scale_data(self):
        self.features = self.tess_data.drop(columns=['tfopwg_disp'])

        # Identify the columns contains -inf and inf values
        inf_columns = self.features.columns.to_series()[np.isinf(features).any()]
        # Replace -inf and inf values with NaN
        self.features[inf_columns] = self.features[inf_columns].replace([np.inf, -np.inf], np.nan)
        # Fill NaN values with median values
        self.features.fillna(self.median_values, inplace=True)
        print(f"Columns with -inf or inf values replaced: {list(inf_columns)}")
        # Scale the features
        self.scaled_features = self.scaler.fit_transform(self.features)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.scaled_features, 
            self.tess_data['tfopwg_disp'], 
            test_size=0.2, 
            random_state=42
        )
        print("Data scaling completed.")