# %%
import pandas as pd
import numpy as np

# %%
# Load the Kepler exoplanet dataset, skipping the first 53 rows of metadata (Commented lines)
kepler_data = pd.read_csv('kepler_exoplanet_data.csv', skiprows=53)

# Drop the unnecessary columns like KEPID, KOI Name, Kepler Name, koi_tce_delivname.
kepler_data = kepler_data.drop(columns=['kepid', 'kepoi_name', 'kepler_name', 'koi_tce_delivname'])

# Convert the 'koi_disposition' column to categorical type
kepler_data['koi_disposition'] = kepler_data['koi_disposition'].astype('category')
categories = kepler_data['koi_disposition'].cat.categories

# Map the categories to numerical codes and replace the original column 0 -> 'CANDIDATE', 2 -> 'FALSE POSITIVE', 1 -> 'CONFIRMED'
kepler_data['koi_disposition'] = kepler_data['koi_disposition'].cat.codes

#convert 'koi_pdisposition' to Categorical type
kepler_data['koi_pdisposition'] = kepler_data['koi_pdisposition'].astype('category')
kepler_data['koi_pdisposition'] = kepler_data['koi_pdisposition'].cat.codes

#Check any character columns and convert them to categorical type
for column in kepler_data.select_dtypes(include=['object']).columns:
    print(f"Column '{column}' is of type 'object'")
    kepler_data[column] = kepler_data[column].astype('category')
    kepler_data[column] = kepler_data[column].cat.codes

# identify columns with missing values and fill them with the median of the respective columns
missing_value_columns = kepler_data.columns[kepler_data.isnull().any()]
for column in missing_value_columns:
    median_value = kepler_data[column].median()
    kepler_data[column].fillna(median_value, inplace=True)

# verify the missing values have been handled
# print(kepler_data.isnull().sum())


print(kepler_data.head())


# # Fill missing values in 'koi_prad' with the median value of the column
# kepler_data['koi_prad'].fillna(kepler_data['koi_prad'].median(), inplace=True)

# %%
# Identify skewed columns in kepler_data
# skew_values = kepler_data.skew(numeric_only=True)

# # List columns with high skewness
# skewed_columns = skew_values[abs(skew_values) > 1].index.tolist()
# # print("Highly skewed columns:", skewed_columns)

# # log-transform skewed features
# for column in skewed_columns:
#     # Apply RobustScaler to reduce the impact of outliers
#     kepler_data[column] = np.log1p(kepler_data[column])

# print(kepler_data.head())

# %%
data_to_scale = kepler_data.drop(columns=['koi_disposition', 'koi_pdisposition'])

# Identify columns containing -inf values
inf_columns = []
for column in data_to_scale.columns:
    if np.isneginf(data_to_scale[column]).any():
        inf_columns.append(column)
print("Columns containing -inf values:", inf_columns)

# Display rows for each column that contain -inf values
for column in inf_columns:
    inf_rows = data_to_scale[np.isneginf(data_to_scale[column])]
    print(f"Rows with -inf in column '{column}':")
    print(inf_rows[[column]])

# %%
%pip install scikit-learn
from sklearn.preprocessing import RobustScaler
import numpy as np

# Apply RobustScaler
scaler = RobustScaler()


scaled_data = scaler.fit_transform(data_to_scale)

# %%
# Convert scaled_data (numpy array) back to DataFrame with original column names
scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns)

# add back the label columns for ML training:
final_df = pd.concat([scaled_df, kepler_data[['koi_disposition', 'koi_pdisposition']].reset_index(drop=True)], axis=1)

print(final_df.head())

# %%
 


