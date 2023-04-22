from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression
from sklearn.cluster import FeatureAgglomeration

def select_features(X, y=None, ml_type='classification', method='SelectKBest', n_features=30):
    """
    Perform feature selection on the input data.

    Parameters
    ----------
    X : array-like or pd.DataFrame of shape (n_samples, n_features)
        The input samples.
    y : array-like or pd.Series of shape (n_samples,), default=None
        The target values for supervised learning problems (classification or regression).
        Not used for unsupervised learning problems (clustering).
    ml_type : str, default='classification'
        The type of ML problem. Options: 'classification', 'regression', 'clustering'.
    method : str, default='SelectKBest'
        The feature selection method. Options: 'SelectKBest', 'MutualInfo', 'Agglomeration'.
    n_features : int, default=30
        The number of top features to select.

    Returns
    -------
    X_selected : array-like or pd.DataFrame of shape (n_samples, n_features_selected)
        The input data with only the selected features.
    """

    if ml_type == 'classification':
        if method == 'SelectKBest':
            selector = SelectKBest(f_classif, k=n_features)
        elif method == 'MutualInfo':
            selector = SelectKBest(mutual_info_classif, k=n_features)
        else:
            raise ValueError("Invalid method for classification problem.")
    elif ml_type == 'regression':
        if method == 'SelectKBest':
            selector = SelectKBest(f_regression, k=n_features)
        elif method == 'MutualInfo':
            selector = SelectKBest(mutual_info_regression, k=n_features)
        else:
            raise ValueError("Invalid method for regression problem.")
    elif ml_type == 'clustering':
        if method == 'Agglomeration':
            selector = FeatureAgglomeration(n_clusters=n_features)
        else:
            raise ValueError("Invalid method for clustering problem.")
    else:
        raise ValueError("Invalid problem_type. Options: 'classification', 'regression', 'clustering'.")

    if ml_type == 'clustering':
        X_selected = selector.fit_transform(X)
    else:
        X_selected = selector.fit_transform(X, y)

    return X_selected