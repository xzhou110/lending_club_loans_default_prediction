from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

class DateTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = (pd.Timestamp.now() - pd.to_datetime(X[col])).dt.days
        return X


class ExtremeValuesNumericalHandler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.lower_bound = X.quantile(0.05)
        self.upper_bound = X.quantile(0.95)
        return self

    def transform(self, X, y=None):
        for col in X.columns:
            X[col] = np.where(X[col] < self.lower_bound[col], self.lower_bound[col], X[col])
            X[col] = np.where(X[col] > self.upper_bound[col], self.upper_bound[col], X[col])
        return X


class DropHighCorrelation(BaseEstimator, TransformerMixin):
    def __init__(self, correlation_threshold=0.9):
        self.correlation_threshold = correlation_threshold

    def fit(self, X, y=None):
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        self.columns_to_drop = [col for col in upper.columns if any(upper[col] > self.correlation_threshold)]
        return self

    def transform(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop)
        return X


class DummyVariableCreator(BaseEstimator, TransformerMixin):
    def __init__(self, max_unique_values_cat=50):
        self.max_unique_values_cat = max_unique_values_cat

    def fit(self, X, y=None):
        self.columns_to_encode = [col for col in X.columns if X[col].nunique() <= self.max_unique_values_cat]
        self.columns_to_drop = [col for col in X.columns if X[col].nunique() > self.max_unique_values_cat]
        return self

    def transform(self, X, y=None):
        X = X.drop(columns=self.columns_to_drop)
        X = pd.get_dummies(X, columns=self.columns_to_encode, dummy_na=True, drop_first=True)
        return X

class LoggingPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        message = X.shape
        for name, step in self.steps[:-1]:
            X = step[1].fit_transform(X, y, **fit_params)
            logging.info(f"Finished {name}, Input shape: {message}, Output shape: {X.shape}")
            message = X.shape
        self.steps[-1][1].fit(X, y, **fit_params)
        logging.info(f"Finished {self.steps[-1][0]}")
        return self

def preprocessing_pipeline(df):
    # Define the pipeline steps
    date_pipeline = LoggingPipeline([
    ('to_days', FunctionTransformer(convert_to_days))
])

    numerical_pipeline = LoggingPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('winsorizer', FunctionTransformer(winsorize_numerical, kw_args={'limits': [0.05, 0.05]}))
    ])

    categorical_pipeline = LoggingPipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Get the column names
    date_columns = [col for col in df.columns if 'date' in col.lower()]
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Combine the pipelines
    full_pipeline = ColumnTransformer(transformers=[
        ('date', date_pipeline, date_columns),
        ('num', numerical_pipeline, numerical_columns),
        ('cat', categorical_pipeline, categorical_columns)
    ])

    return full_pipeline
