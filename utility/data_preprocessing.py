import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import datetime
import logging

logging.basicConfig(level=logging.INFO)

def preprocess_data(df, label='label', missing_threshold=0.9, max_unique_values_cat=50, correlation_threshold=0.9):
    """
    Preprocesses the input data by performing the following steps:
        1. Dropping columns with a high proportion of missing values.
        2. Converting date columns to the number of days elapsed since the date.
        3. Handling missing values in numerical columns.
        4. Handling extreme values in numerical columns.
        5. Converting categorical columns with a limited number of unique values into dummy variables.
        6. Standardizing numerical columns using z-score normalization.
        7. Handling high correlation among numerical columns.

    Args:
        df (pd.DataFrame): Raw data with label column.
        label (str, optional): Name of the label field. Defaults to 'label'.
        missing_threshold (float, optional): Defaults to 0.9. Threshold for the proportion of missing values in a column.
        max_unique_values_cat (int, optional): Defaults to 50. Maximum number of unique values allowed for a categorical column.
        correlation_threshold (float, optional): Defaults to 0.9. Threshold for the absolute value of the correlation coefficient between numerical columns.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed features and labels.
    """

    df = df.copy()
    y = df[label]
    df = df.drop(columns=[label])

    # Step 1: Dropping columns with missing values above the threshold
    logging.info("Step 1: Dropping columns with missing values above the threshold")
    df = df.loc[:, df.isna().mean() < missing_threshold]

    # Step 2: Processing date columns and renaming them
    logging.info("Step 2: Processing date columns and renaming them")
    date_columns = [col for col in df.columns if 'date' in col.lower()]

    for col in date_columns:
        df[col] = (datetime.datetime.now() - pd.to_datetime(df[col])).dt.days
        df.rename(columns={col: col + '_days'}, inplace=True)

    # Step 3: Handling missing values for numerical columns
    logging.info("Step 3: Handling missing values for numerical columns")
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    for col in numerical_columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Step 4: Handling extreme values for numerical columns
    logging.info("Step 4: Handling extreme values for numerical columns")
    lower_bound = df[numerical_columns].quantile(0.05)
    upper_bound = df[numerical_columns].quantile(0.95)

    for col in numerical_columns:
        df[col] = np.where((df[col] < lower_bound[col]) | (df[col] > upper_bound[col]), df[col].median(), df[col])

    # Step 5: Creating dummy variables for categorical columns
    logging.info("Step 5: Creating dummy variables for categorical columns")
    max_unique_values = max_unique_values_cat
    categorical_columns_to_drop = [col for col in categorical_columns if df[col].nunique() > max_unique_values]
    df.drop(columns=categorical_columns_to_drop, inplace=True)
    remaining_categorical_columns = list(set(categorical_columns) - set(categorical_columns_to_drop))
    df = pd.get_dummies(df, columns=remaining_categorical_columns, dummy_na=True, drop_first=True)

    # Step 6: Standardizing numerical columns
    logging.info("Step 6: Standardizing numerical columns")
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Step 7: Handling high correlation among numerical columns
    logging.info("Step 7: Handling high correlation among numerical columns")
    corr_matrix = df[numerical_columns].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    columns_to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]

    df.drop(columns=columns_to_drop, inplace=True)

    # Combine the processed features and the label into a single DataFrame
    result_df = pd.concat([df, y], axis=1)

    return result_df
