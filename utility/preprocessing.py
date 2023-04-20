import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime

def preprocess_data(df, label='label', missing_threshold=0.9):
    """
    preprocess the data, including: 1. 
    args:
        df: raw data with label
        label: the name of label field
        missing_thresold: default at 0.9. the thresold to remove features with too many null values
    
    returns:
        X_processed: processed features
        y: labels
    
    """
    
    
    
    # Create a copy of the input DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Drop columns with more than the specified missing threshold
    df = df.loc[:, df.isna().mean() < missing_threshold]
    print("Step 1: Dropped columns with missing values above the threshold")

    # Identify date columns
    date_columns = [col for col in df.columns if 'date' in col.lower()]

    # Calculate the duration between the date and today for date columns and rename them
    for col in date_columns:
        df[col] = (datetime.datetime.now() - pd.to_datetime(df[col])).dt.days
        df.rename(columns={col: col + '_days'}, inplace=True)
    print("Step 2: Processed date columns and renamed them")

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    print("Step 3: Identified numerical and categorical columns")

    # Exclude target variable from the list of numerical columns
    numerical_columns = [col for col in numerical_columns if col != label]

    # Handle missing values for numerical columns
    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    print("Step 4: Handled missing values for numerical columns")

    # Exclude extreme values for numerical columns
    lower_bound = df[numerical_columns].quantile(0.05)
    upper_bound = df[numerical_columns].quantile(0.95)
    for col in numerical_columns:
        df[col] = np.where((df[col] < lower_bound[col]) | (df[col] > upper_bound[col]), df[col].median(), df[col])
    print("Step 5: Excluded extreme values for numerical columns")

    # Extract target variable
    y = df[label]

    # Drop target variable from the DataFrame
    df = df.drop(columns=[label])

    # Create dummy variables for categorical columns
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=True, drop_first=True)
    print("Step 6: Created dummy variables for categorical columns")

    # Standardize numerical columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    print("Step 7: Standardized numerical columns")

    X_processed = df.values

    return X_processed, y