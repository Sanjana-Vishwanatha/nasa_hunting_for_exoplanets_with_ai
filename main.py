import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)

class ExoplanetModel:
    def __init__(self, data_path):
        self.data_path = data_path
        self.scaler = RobustScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_name = "random_forest_exoplanet_model.joblib"
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        self.data_cleaning()
        self.scale_data()
        self.load_model()

    def load_data(self):
        self.kepler_data = pd.read_csv(self.data_path, skiprows=53)
        print("Data loaded successfully.")

    def preprocess_data(self):
        
        # Drop the unnecessary columns like KEPID, KOI Name, Kepler Name, koi_tce_delivname.
        self.kepler_data = self.kepler_data.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname'])
        
        # Convert the 'koi_disposition' column to categorical type
        self.kepler_data['koi_disposition'] = self.kepler_data['koi_disposition'].astype('category')
        self.categories = self.kepler_data['koi_disposition'].cat.categories

        # Map the categories to numerical codes and replace the original column 0 -> 'CANDIDATE', 2 -> 'FALSE POSITIVE', 1 -> 'CONFIRMED'
        self.kepler_data['koi_disposition'] = self.kepler_data['koi_disposition'].cat.codes

        # Convert 'koi_pdisposition' to Categorical type
        self.kepler_data['koi_pdisposition'] = self.kepler_data['koi_pdisposition'].astype('category')
        self.kepler_data['koi_pdisposition'] = self.kepler_data['koi_pdisposition'].cat.codes

        print("Data preprocessing completed.")

    def data_cleaning(self):
        # Check any character columns and convert them to categorical type
        for column in self.kepler_data.select_dtypes(include=['object']).columns:
            print(f"Column '{column}' is of type 'object'")
            self.kepler_data[column] = self.kepler_data[column].astype('category')
            self.kepler_data[column] = self.kepler_data[column].cat.codes

        # Fill all numeric columns' missing values with their median
        self.kepler_data.fillna(
            self.kepler_data.median(numeric_only=True),
            inplace=True
        )

    def scale_data(self):
        self.data_to_scale = self.kepler_data.drop(columns=['koi_disposition', 'koi_pdisposition'])

        # Identify columns containing -inf values
        inf_columns = []
        for column in self.data_to_scale.columns:
            if np.isneginf(self.data_to_scale[column]).any():
                inf_columns.append(column)
        print("Columns containing -inf values:", inf_columns)

        # Display rows for each column that contain -inf values
        for column in inf_columns:
            inf_rows = self.data_to_scale[np.isneginf(self.data_to_scale[column])]
            print(f"Rows with -inf in column '{column}':")
            print(inf_rows[[column]])
            # Replace -inf values with NaN
            self.data_to_scale[column].replace(-np.inf, np.nan, inplace=True)
            # Fill NaN values with the median of the column
            median_value = self.data_to_scale[column].median()
            self.data_to_scale[column].fillna(median_value, inplace=True)


        # Apply RobustScaler
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(self.data_to_scale)
        # Convert scaled_data (numpy array) back to DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_data, columns=self.data_to_scale.columns)

        # add back the label columns for ML training:
        self.final_df = pd.concat([scaled_df, self.kepler_data[['koi_disposition', 'koi_pdisposition']].reset_index(drop=True)], axis=1)
        # print(final_df.head(3))
        print("Data scaling completed.")

    def train_model(self):
        # Split the data into features and labels
        X = self.final_df.drop("koi_disposition", axis=1)
        y = self.final_df["koi_disposition"]

        X = pd.get_dummies(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train the Random Forest Classifier
        clf = RandomForestClassifier(
            n_estimators=100,    # number of trees
            max_depth=None,      # let it grow deep (or set manually)
            random_state=42
        )

        clf.fit(X_train, y_train)
        # Make predictions aginst the test set
        y_pred = clf.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        self.model_name = "random_forest_exoplanet_model.joblib"

        joblib.dump(clf, self.model_name)

        return self.model_name
    
    def load_model(self):
        self.model = joblib.load(self.model_name)
        print("Model loaded successfully.")

    def predict(self, input_data):
        # Ensure input_data is a DataFrame
        if input_data is None:
            input_data = {
                'koi_score': 0.9,
                'koi_fpflag_nt': 1,
                'koi_fpflag_ss': 0,
                'koi_fpflag_co': 0,
                'koi_fpflag_ec': 0,
                'koi_period': 10.5,
                'koi_period_err1': 0.01,
                'koi_period_err2': -0.01,
                'koi_time0bk': 130.5,
                'koi_time0bk_err1': 0.1,
                'koi_time0bk_err2': -0.1,
                'koi_impact': 0.2,
                'koi_impact_err1': 0.01,
                'koi_impact_err2': -0.01,
                'koi_duration': 5.0,
                'koi_duration_err1': 0.2,
                'koi_duration_err2': -0.2,
                'koi_depth': 1500,
                'koi_depth_err1': 100,
                'koi_depth_err2': -100,
                'koi_prad': 1.2,
                'koi_prad_err1': 0.05,
                'koi_prad_err2': -0.05,
                'koi_teq': 500,
                'koi_teq_err1': 10,
                'koi_teq_err2': -10,
                'koi_insol': 1.1,
                'koi_insol_err1': 0.1,
                'koi_insol_err2': -0.1,
                'koi_model_snr': 15.0,
                'koi_tce_plnt_num': 1,
                'koi_steff': 5700,
                'koi_steff_err1': 50,
                'koi_steff_err2': -50,
                'koi_slogg': 4.4,
                'koi_slogg_err1': 0.1,
                'koi_slogg_err2': -0.1,
                'koi_srad': 1.0,
                'koi_srad_err1': 0.05,
                'koi_srad_err2': -0.05,
                'ra': 290.0,
                'dec': 44.5,
                'koi_kepmag': 14.0,
                'koi_pdisposition': 1  # If this was used as a feature
            }
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])

        # Preprocess input data (handle categorical variables and scaling)
        input_data = pd.get_dummies(input_data)
        input_data = input_data.reindex(columns=self.final_df.columns.drop('koi_disposition'), fill_value=0)

        # Make prediction
        prediction = self.model.predict(input_data)
        predicted_class = self.categories[prediction[0]]
        return predicted_class

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.get_json()
    user_input = data.get('user_input')
    file_path  = "dataset\kepler_exoplanet_data.csv"
    if not file_path:
        return jsonify({'error': 'file_path is required'}), 400
    
    exoplanet_model = ExoplanetModel(file_path)
    # user_input = None
    prediction = exoplanet_model.predict(user_input)
    print(f"The predicted class for the input data is: {prediction}")

    return jsonify({'predicted_class': prediction})

if __name__ == "__main__":
    print("----------------python file executing------------------")
    app.run(port=5000)

# if __name__ == "__main__":
#     data_path  = "dataset\kepler_exoplanet_data.csv"
#     exoplanet_model = ExoplanetModel(data_path)
    
#     # Train the model
#     model_path = exoplanet_model.train_model()
#     # Load the trained model
#     exoplanet_model.load_model(model_path)
#     # Example user input for prediction
#     user_input = None
#     prediction = exoplanet_model.predict(user_input)
#     print(f"The predicted class for the input data is: {prediction}")
