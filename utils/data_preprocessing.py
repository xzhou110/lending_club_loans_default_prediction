import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df, label='label', missing_threshold=0.9, max_unique_values_cat=50, correlation_threshold=0.9):
    """
    Preprocesses the input data for use in a machine learning model.This function performs various preprocessing steps, including: 
    1. Encode labels: Convert the target variable column into numerical values.
    2. Drop columns with missing values above a specified threshold.
    3. Process date columns: Calculate the number of days from the date to the current date and rename the columns accordingly.
    4. Handle missing values for numerical columns: Replace missing values with the median of the respective column.
    5. Handle extreme values for numerical columns: Replace values below a certain percentile with that percentile value, and values above another percentile with that percentile value.
    6. Create dummy variables for categorical columns: Convert categorical columns with a number of unique values below a specified threshold into dummy variables.
    7. Handle high correlation among numerical columns: Drop one column from each pair of highly correlated columns based on a specified correlation threshold.
    8. Standardize numerical columns: Scale the numerical columns to have a mean of 0 and a standard deviation of 1.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input data to preprocess.
    label : str, optional, default: 'label'
        The name of the target variable column in the input DataFrame.
    missing_threshold : float, optional, default: 0.9
        The threshold for the proportion of missing values in a column. Columns with a proportion of 
        missing values above this threshold will be dropped.
    max_unique_values_cat : int, optional, default: 50
        The maximum number of unique values allowed for a categorical column. Categorical columns with 
        more unique values than this threshold will be dropped.
    correlation_threshold : float, optional, default: 0.9
        The threshold for the correlation between numerical columns. Pairs of columns with a correlation 
        above this threshold will be handled by dropping one of the columns.

    Returns:
    --------
    result_df : pandas.DataFrame
        The preprocessed data, including the target variable column.

    """
    
    df = df.copy()
    
    # Encoding labels and printing out the encoding
    df, y, label_encoding = encode_labels(df, label)
    logging.info(f"Label encoding: {label_encoding}")

    # Dropping columns with missing values above the threshold
    df = drop_missing_value_columns(df, missing_threshold)
    logging.info("Dropped columns with missing values above the threshold")

    # Processing date columns and renaming them
    df = process_date_columns(df)
    logging.info("Processed date columns")

    # Handling missing values for numerical columns
    df = handle_missing_values_numerical(df)
    logging.info("Handled missing values for numerical columns")

    # Handling extreme values for numerical columns
    df = handle_extreme_values_numerical(df)
    logging.info("Handled extreme values for numerical columns")

    # Creating dummy variables for categorical columns
    df = create_dummy_variables(df, max_unique_values_cat)
    logging.info("Created dummy variables for categorical columns")

    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Handling high correlation among numerical columns
    df = handle_high_correlation(df, correlation_threshold)
    logging.info("Handled high correlation among numerical columns")
    
    # Standardizing numerical columns
    df = standardize_numerical_columns(df)
    logging.info("Standardized numerical columns")

    # Combine the processed features and the label into a single DataFrame
    df.reset_index(drop=True, inplace=True)
    result_df = pd.concat([df, pd.Series(y, name=label)], axis=1)

    return result_df


def encode_labels(df, label):
    y, unique_labels = pd.factorize(df[label])
    label_encoding = {label: idx for idx, label in enumerate(unique_labels)}
    df = df.drop(columns=[label])
    return df, y, label_encoding


def drop_missing_value_columns(df, missing_threshold):
    return df.loc[:, df.isna().mean() < missing_threshold]


def process_date_columns(df):
    date_columns = [col for col in df.columns if 'date' in col.lower()]

    for col in date_columns:
        df[col] = (datetime.datetime.now() - pd.to_datetime(df[col])).dt.days
        df.rename(columns={col: col + '_days'}, inplace=True)
    
    return df


def handle_missing_values_numerical(df):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)
    
    return df


def handle_extreme_values_numerical(df):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    lower_bound = df[numerical_columns].quantile(0.05)
    upper_bound = df[numerical_columns].quantile(0.95)

    for col in numerical_columns:
        df[col] = np.where(df[col] < lower_bound[col], lower_bound[col], df[col])
        df[col] = np.where(df[col] > upper_bound[col], upper_bound[col], df[col])
    
    return df


def create_dummy_variables(df, max_unique_values_cat):
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    categorical_columns_to_drop = [col for col in categorical_columns if df[col].nunique() > max_unique_values_cat]
    df.drop(columns=categorical_columns_to_drop, inplace=True)
    remaining_categorical_columns = list(set(categorical_columns) - set(categorical_columns_to_drop))
    df = pd.get_dummies(df, columns=remaining_categorical_columns, dummy_na=True, drop_first=True)
    
    return df


def handle_high_correlation(df, correlation_threshold):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    corr_matrix = df[numerical_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    columns_to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    df.drop(columns=columns_to_drop, inplace=True)
    return df

def standardize_numerical_columns(df):
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df
