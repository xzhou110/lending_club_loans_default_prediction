import time
import logging
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)

def select_features(X, y=None, n_features_to_select=30, ml_type='classification', estimator=None, n_samples=None):
    """
    Selects the most relevant features for a machine learning model using Recursive Feature Elimination with Cross-Validation (RFECV)
    with a default estimator based on the specified problem type.

    Parameters:
    -----------
    X : pandas.DataFrame
        The input features (independent variables) to perform feature selection on.
    y : pandas.Series or numpy.array, optional, default: None
        The target variable (dependent variable) corresponding to the input features. Required for
        classification and regression problems.
    n_features_to_select : int, optional, default: 30
        The number of top features to select.
    ml_type : str, optional, default: 'classification'
        The type of machine learning problem. Supported values are 'regression', 'classification', and 'clustering'.
    estimator : sklearn-compatible estimator, optional, default: None
        The estimator to use for feature selection. If None, an estimator based on the ml_type will be used.
    n_samples : int, optional, default: None
        The number of samples to downsample the data to. If None, no downsampling will be performed.

    Returns:
    --------
    selected_features : list
        The names of the selected features.

    """
    start_time = time.time()
    
    # Downsample the data if n_samples is provided
    if n_samples is not None:
        X = X.sample(n_samples, random_state=42)
        if y is not None:
            y = y.loc[X.index]

    if estimator is None:
        if ml_type == 'regression':
            estimator = LinearRegression()
        elif ml_type == 'classification':
            estimator = DecisionTreeClassifier()
        elif ml_type == 'clustering':
            estimator = KMeans()
        else:
            raise ValueError("Invalid ml_type. Supported values are 'regression', 'classification', and 'clustering'.")

    cv = 5 if ml_type != 'clustering' else None

    logging.info("Starting feature selection...")
    if ml_type == 'clustering':
        # Step = 1: Eliminate one feature at a time in each iteration.
        selector = RFECV(estimator, min_features_to_select=n_features_to_select, step=1, cv=cv)
        selector.fit(X)
    else:
        if y is None:
            raise ValueError("y cannot be None for classification or regression problems.")
        selector = RFECV(estimator, min_features_to_select=n_features_to_select, step=1, cv=cv)
        selector.fit(X, y)

    selected_features = X.columns[selector.support_].tolist()

    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Feature selection completed in {duration:.2f} seconds")

    return selected_features
