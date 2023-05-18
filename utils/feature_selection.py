import time
import logging
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

def select_features(X_train, y_train=None, min_features_to_select=10, n_features_to_select=None, ml_type='classification', estimator=None, n_samples=None):
    """
    Selects the most relevant features for a machine learning model using Recursive Feature Elimination 
    with Cross-Validation (RFECV) with a default estimator based on the specified problem type.

    Parameters:
    -----------
    X_train : pandas.DataFrame
        The input training features (independent variables) to perform feature selection on.
    y_train : pandas.Series or numpy.array, optional, default: None
        The target training variable (dependent variable) corresponding to the input features. Required for
        classification and regression problems.
    min_features_to_select : int, default=10
        The minimum number of features to select. This number of features will always be scored.
    n_features_to_select : int, optional, default: None
        The number of top features to select. If None, an optimal number will be determined by cross-validation.
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
        X_train = X_train.sample(n_samples, random_state=42)
        if y_train is not None:
            y_train = y_train.loc[X_train.index]

    if estimator is None:
        if ml_type == 'regression':
            estimator = LinearRegression()
        elif ml_type == 'classification':
            estimator = DecisionTreeClassifier(class_weight='balanced')
        elif ml_type == 'clustering':
            estimator = KMeans()
        else:
            raise ValueError("Invalid ml_type. Supported values are 'regression', 'classification', and 'clustering'.")

    cv = 5 if ml_type != 'clustering' else None

    logging.info("Starting feature selection...")
    selector = RFECV(estimator, step=1, min_features_to_select=min_features_to_select, cv=cv)
    if ml_type == 'clustering':
        selector.fit(X_train)
    else:
        if y_train is None:
            raise ValueError("y_train cannot be None for classification or regression problems.")
        selector.fit(X_train, y_train)

    # Plot number of features VS. cross-validation scores
    plt.figure(figsize=(16, 9))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(selector.cv_results_['mean_test_score']) + 1), selector.cv_results_['mean_test_score'])
    plt.xticks(np.arange(0, len(selector.cv_results_['mean_test_score']) + 1, 10)) # 10 as step for x-axis
    plt.show()
    
    end_time = time.time()
    duration = end_time - start_time
    logging.info(f"Feature selection completed in {duration:.2f} seconds")
    
    optimal_num_features = selector.n_features_
    optimal_features_selected = X_train.columns[selector.support_].tolist()
    
    ranked_features = sorted(zip(selector.ranking_, X_train.columns))
    selected_features = [feature for _, feature in ranked_features[:selector.n_features_]]
    if n_features_to_select is not None and n_features_to_select < optimal_num_features:
        logging.info(f"Requested number of features ({n_features_to_select}) is less than the optimal number of features ({optimal_num_features}).")
        selected_features = selected_features[:n_features_to_select]
    elif n_features_to_select is not None and n_features_to_select > optimal_num_features:
        logging.info(f"Requested number of features ({n_features_to_select}) is greater than the optimal number of features ({optimal_num_features}). All features selected by RFECV are used.")

    logging.info(f"Number of features selected: {len(selected_features)}")
    logging.info(f"Selected features : {selected_features}")
    
    if n_features_to_select is not None and n_features_to_select < optimal_num_features:
        logging.info(f"Optimal number of features : {optimal_num_features}")
        logging.info(f"Optimal features selected : {optimal_features_selected}")
    
    return selected_features


