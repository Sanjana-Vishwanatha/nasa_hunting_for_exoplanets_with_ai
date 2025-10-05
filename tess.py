import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ExoplanetModel_TESS:
    def __init__(self):
        self.data_path = "dataset/TESS_exoplanet_data.csv"
        self.scaler = RobustScaler()
        self.model = None
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
        self.tess_data = pd.read_csv(self.data_path, skiprows=90)
        print("Data loaded successfully.")

    def preprocess_data(self):
        # Drop the unnecessary columns like TIC ID, TESS Name, etc.
        self.tess_data = self.tess_data.drop(columns=['toi', 'tid', 'toi_created', 'rowupdate', 'rastr', 'decstr'])  # Drop the columns that are not needed
        
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
        inf_columns = self.features.columns.to_series()[np.isinf(self.features).any()]
        # Replace -inf and inf values with NaN
        self.features[inf_columns] = self.features[inf_columns].replace([np.inf, -np.inf], np.nan)
        # Fill NaN values with median values
        self.features.fillna(self.median_values, inplace=True)
        print(f"Columns with -inf or inf values replaced: {list(inf_columns)}")
        scaled_data = self.scaler.fit_transform(self.features)
        # Convert scaled_data (numpy array) back to DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_data, columns=self.features.columns)

        # add back the label columns for ML training:
        self.final_df = pd.concat([scaled_df, self.tess_data[['tfopwg_disp']].reset_index(drop=True)], axis=1)
       
        print("Data scaling completed.")

    def train_model(self):
        X = self.final_df.drop(columns=['tfopwg_disp'])
        y = self.final_df['tfopwg_disp']

        X = pd.get_dummies(X)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        tess_model = RandomForestClassifier(n_estimators=200, random_state=42)
        tess_model.fit(X_train, y_train)
        y_pred = tess_model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        report = classification_report(y_test, y_pred, target_names=self.categories)
        print(f"Model trained with accuracy: {accuracy:.4f}")
        print(f"Classification Report:{report}")

        joblib.dump(tess_model, self.model_name)

    def load_model(self):
        try:
            self.model = joblib.load(self.model_name)
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("Model file not found. Please train the model first.")

    def predict(self, input_data):
        if self.model is None:
            return {"error": "Model not loaded. Please train the model first."}
        
         # Convert to DataFrame if it's a dictionary
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])

        # Fill null values with median values from training data
        for column in input_data.columns:
            if column in self.median_values:
                input_data[column] = input_data[column].fillna(self.median_values[column])
            else:
                input_data[column] = input_data[column].fillna(0)

        # Align columns with training data
        input_data = input_data.reindex(columns=self.final_df.columns.drop('tfopwg_disp'), fill_value=0)

        # Scale the input data
        scaled_input = self.scaler.transform(input_data)

        prediction = self.model.predict(scaled_input)
        predicted_category = self.categories[prediction[0]]
        return {"predicted_class": int(prediction[0]), "predicted_label": predicted_category}

# test = ExoplanetModel_TESS()
# test.train_model()