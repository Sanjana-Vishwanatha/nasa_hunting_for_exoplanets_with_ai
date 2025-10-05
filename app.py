import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ExoplanetModel:
    def __init__(self):
        self.data_path = "./dataset/kepler_exoplanet_data.csv"
        self.scaler = RobustScaler()
        self.model = None
        self.model_name = "./models/random_forest_exoplanet_model.joblib"
        # Store median values for filling nulls
        self.median_values = {}
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

        # Store median values before filling
        self.median_values = self.kepler_data.median(numeric_only=True).to_dict()
        
        # Fill all numeric columns' missing values with their median
        self.kepler_data.fillna(self.median_values, inplace=True)

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
        scaled_data = self.scaler.fit_transform(self.data_to_scale)

        # Convert scaled_data (numpy array) back to DataFrame with original column names
        scaled_df = pd.DataFrame(scaled_data, columns=self.data_to_scale.columns)

        # add back the label columns for ML training:
        self.final_df = pd.concat([scaled_df, self.kepler_data[['koi_disposition', 'koi_pdisposition']].reset_index(drop=True)], axis=1)
        
        # Store the feature columns for prediction
        self.feature_columns = self.final_df.columns.drop('koi_disposition').tolist()
        
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
            random_state=42
        )

        clf.fit(X_train, y_train)
        # Make predictions aginst the test set
        y_pred = clf.predict(X_test)

        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        self.model_name = "./models/random_forest_exoplanet_model.joblib"

        joblib.dump(clf, self.model_name)

        return self.model_name
    
    def load_model(self):
        self.model = joblib.load(self.model_name)
        print("Model loaded successfully.")

    def prepare_predict_data(self, input_data):
        # Convert to DataFrame if it's a dictionary
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
        
        # Fill null values with median values from training data
        for column in input_data.columns:
            if column in self.median_values:
                input_data[column].fillna(self.median_values[column], inplace=True)
            else:
                # If column not in median_values, fill with 0 or appropriate default
                input_data[column].fillna(0, inplace=True)
        
        # Preprocess input data (handle categorical variables and scaling)
        input_data = pd.get_dummies(input_data)
        
        # Align columns with training data
        input_data = input_data.reindex(columns=self.final_df.columns.drop('koi_disposition'), fill_value=0)
        
        return input_data
    
    def predict(self, input_data):
        # Ensure input_data is provided
        if input_data is None:
            return "Error: No input data provided"
        
        # Convert to DataFrame if it's a dictionary
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])
    
        # Fill null values with median values from training data
        for column in input_data.columns:
            if column in self.median_values:
                input_data[column] = input_data[column].fillna(self.median_values[column])
            else:
                input_data[column] = input_data[column].fillna(0)
    
        # Handle categorical columns the same way as in training
        if 'koi_pdisposition' in input_data.columns:
            # Use the same categories as in training
            input_data['koi_pdisposition'] = input_data['koi_pdisposition'].astype('category')
            # Align categories with training data
            input_data['koi_pdisposition'] = input_data['koi_pdisposition'].cat.set_categories(
                self.kepler_data['koi_pdisposition'].astype('category').cat.categories
            )
            input_data['koi_pdisposition'] = input_data['koi_pdisposition'].cat.codes
            # If unknown category, will be set to -1;
    
        # Align columns with training data
        input_data = input_data.reindex(columns=self.final_df.columns.drop('koi_disposition'), fill_value=0)
    
        # Drop 'koi_pdisposition' before scaling
        data_to_scale = input_data.drop(columns=['koi_pdisposition'])
        
        # Scale the input data
        scaled_input_data = self.scaler.transform(data_to_scale)
        
        # Convert scaled data back to DataFrame with correct columns
        scaled_df = pd.DataFrame(scaled_input_data, columns=data_to_scale.columns)
        
        # Add back the 'koi_pdisposition' column
        scaled_df['koi_pdisposition'] = input_data['koi_pdisposition'].values
        
        # Now use scaled_df for prediction
        prediction = self.model.predict(scaled_df)
        predicted_class = self.categories[prediction[0]]
        predicted_code = int(prediction[0])
        
        return {
            'predicted_class': predicted_code,
            'predicted_label': predicted_class
        }