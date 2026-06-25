import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os
import kagglehub
from data_validation import validate_dataframe

def load_and_preprocess_data():
    # Download latest version of the dataset
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    file_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df = pd.read_csv(file_path)

    # Convert 'Churn' column to numeric
    df['Churn'] = df['Churn'].replace(['Yes', 'No'], [1, 0])

    # Drop 'customerID' column
    df.drop('customerID', axis=1, inplace=True)

    # Convert 'TotalCharges' to numeric and handle missing values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

    # Convert all object columns to numeric using LabelEncoder
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scale numerical features
    cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Perform data validation before returning
    if validate_dataframe(X):
        print("Data validation passed in the preprocessing pipeline.")
    else:
        print("Warning: Data validation failed in the preprocessing pipeline.")

    return X, y

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print("Data preprocessing complete. X shape:", X.shape, "y shape:", y.shape)
    print(X.head())
