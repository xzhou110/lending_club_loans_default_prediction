import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO)

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        logging.info(f"Entered Date Columns: DateTransformer")
        X = X.copy()
        for col in X.columns:
            X[col] = (pd.Timestamp.now() - pd.to_datetime(X[col])).dt.days
        return X
    
    def get_feature_names_out(self, input_features=None):
        return input_features 

    
class ExtremeValuesNumericalHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        
    def fit(self, X, y=None):
        logging.info(f"Entered Numerical Columns: ExtremeValuesNumericalHandler")
        X = pd.DataFrame(X)  
        self.lower_bound = X.quantile(0.05)
        self.upper_bound = X.quantile(0.95)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X) 
        for col in X.columns:
            X[col] = np.where(X[col] < self.lower_bound[col], self.lower_bound[col], X[col])
            X[col] = np.where(X[col] > self.upper_bound[col], self.upper_bound[col], X[col])
        return X.values
    
    def get_feature_names_out(self, input_features=None):
        return input_features


class DummyVariableCreator(BaseEstimator, TransformerMixin):
    def __init__(self, max_unique_values_cat=50):
        self.max_unique_values_cat = max_unique_values_cat

    def fit(self, X, y=None):
        X = pd.DataFrame(X) 
        self.columns_to_encode = [col for col in X.columns if X[col].nunique() <= self.max_unique_values_cat]
        self.columns_to_drop = [col for col in X.columns if X[col].nunique() > self.max_unique_values_cat]
        return self

    def transform(self, X, y=None):
        logging.info(f"Entered Categorical Columns: DummyVariableCreator")
        X = pd.DataFrame(X) 
        X = X.drop(columns=self.columns_to_drop)
        X = pd.get_dummies(X, columns=self.columns_to_encode, dummy_na=True, drop_first=True)
        self.feature_names_out_ = X.columns.tolist()  # get the output feature names after transformation
        return X
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_
    
# class CleanFeatureNamesForXgboost(BaseEstimator, TransformerMixin):
#     def __init__(self):
#         self.lower_bound = None
#         self.upper_bound = None
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         logging.info(f"Entered CleanFeatureNamesForXgboost")
#         X = pd.DataFrame(X) 
#         X.columns = X.columns.astype(str)\
#                  .str.replace('[', '_replace_bracket_open_', regex=True)\
#                  .str.replace(']', '_replace_bracket_close_', regex=True)\
#                  .str.replace('<', '_smaller_than_', regex=True)
#         return X
    
#     def get_feature_names_out(self, input_features=None):
#         return input_features

# class LoggingPipeline(Pipeline):
#     def fit(self, X, y=None, **fit_params):
#         message = X.shape
#         for name, step in self.steps[:-1]:
#             print(f'entered {name}')
#             X = step[1].fit_transform(X, y, **fit_params)
#             logging.info(f"Finished {name}, Input shape: {message}, Output shape: {X.shape}")
#             message = X.shape
#         self.steps[-1][1].fit(X, y, **fit_params)
#         logging.info(f"Finished {self.steps[-1][0]}, Input shape: {message}, Output shape: {X.shape}")
#         return self

def preprocessing_pipeline(date_columns, numerical_columns, categorical_columns):

    # Define the pipeline steps
    date_pipeline = Pipeline([
        ('to_days', DateTransformer()),
        # ('clean_feature_names_for_date_cols', CleanFeatureNamesForXgboost())
])

    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('winsorizer', ExtremeValuesNumericalHandler()),
        ('scaler', StandardScaler()),
        # ('clean_feature_names_for_num_cols', CleanFeatureNamesForXgboost())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        # ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ('onehot_with_limit', DummyVariableCreator()),
        # ('clean_feature_names_for_cat_cols', CleanFeatureNamesForXgboost())
    ])

    # Combine the pipelines
    full_pipeline = ColumnTransformer(transformers=[
        ('date', date_pipeline, date_columns),
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    return full_pipeline
