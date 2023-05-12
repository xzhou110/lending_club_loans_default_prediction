import time
import pickle
import concurrent.futures
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


def run_classification_models(X_processed, y, search_type=None, scoring_metric='ROC AUC'):
    """
    Train and evaluate multiple classification models on the given dataset.

    This function trains and evaluates five different classification models: Logistic Regression, K-Nearest Neighbors,
    Support Vector Machine, Random Forest, and XGBoost. It also performs hyperparameter tuning if specified and plots
    ROC curves for all models.

    Parameters
    ----------
    X_processed : pd.DataFrame
        The pre-processed feature matrix (X) as a pandas DataFrame.
    y : pd.Series
        The target variable (y) as a pandas Series.
    search_type : str, optional
        The type of hyperparameter search to perform. Choices are 'grid', 'random', or None. Default is None.
    scoring_metric : str, optional
        The scoring metric to optimize during hyperparameter tuning and to select the best model. 
        Some options are 'Accuracy', 'Precision', 'Recall', 'F1 Score', and 'ROC AUC'. Default is 'ROC AUC'.

    Returns
    -------
    best_model_instance : object
        The best performing model instance (fitted) according to the highest score of the scoring_metric.
    """
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    
    # Handling class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # Hyperparameter grids
    lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [500, 1000, 3000]}
    knn_grid = {'n_neighbors': list(range(1, 31))}
    svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
    xgb_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 6]}

    model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

    # Create a list of models to iterate through
    models = [
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000), lr_grid),
        ('KNN', KNeighborsClassifier(), knn_grid),
        ('SVM', SVC(random_state=42, probability=True), svm_grid),
        ('Random Forest', RandomForestClassifier(random_state=42), rf_grid),
        ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False), xgb_grid)
    ]

    plt.figure(figsize=(20, 12))
    ax = plt.gca()
    
    performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    if scoring_metric not in performance_metrics:
        raise ValueError(f"Invalid scoring_metric. Expected one of: {performance_metrics}")
        
    best_model_obj = None
    best_model_performance = None

    # Iterate through the models and evaluate their performance
    for model_name, model, grid in models:
        start_time = time.time()
        model = fit_model(search_type, model, grid, X_train, y_train, scoring_metric)
        exec_time = time.time() - start_time
        # Calculate execution time
        exec_time = time.time() - start_time
        print(f"{model_name}: Done (Execution Time: {exec_time:.2f} seconds)")

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Train and evaluate the model
        accuracy, precision, recall, f1, roc_auc = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        performance_metrics_values = [accuracy, precision, recall, f1, roc_auc]
        
        # # Append model performance metrics to the model_performance DataFrame
        # model_performance = model_performance.append({
        #     'Model': model_name,
        #     'Accuracy': accuracy,
        #     'Precision': precision,
        #     'Recall': recall,
        #     'F1 Score': f1,
        #     'ROC AUC': roc_auc
        # }, ignore_index=True)
        
        # Append model performance metrics to the model_performance DataFrame
        # Using pd.concat instead of DataFrame.append to avoid the FutureWarning
        performance_df = pd.DataFrame({
            'Model': [model_name],
            'Accuracy': [accuracy],
            'Precision': [precision],
            'Recall': [recall],
            'F1 Score': [f1],
            'ROC AUC': [roc_auc]
        })
        model_performance = pd.concat([model_performance, performance_df], ignore_index=True)
        
         # Update the best performing model
        if best_model_performance is None or performance_metrics_values[performance_metrics.index(scoring_metric)] > best_model_performance[scoring_metric]:
            best_model_obj = model
            best_model_performance = model_performance.iloc[-1]

        # Plot ROC curve
        plot_roc_curve(model_name, y_test, y_pred_proba, ax)

    # Display ROC curve comparison
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve Comparison", fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

    # Display model performance metrics
    print("Model Performance Metrics:\n")
    print(model_performance)

    # Print the best performing model
    print("\nBest Performing Model:\n")
    print(best_model_performance)

    # Print the best hyperparameters for the best performing model
    if search_type in ["grid", "random"]:
        print("\nBest Hyperparameters for the Best Performing Model:\n")
        print(best_model_obj.best_params_)
        
    # Pick the best performing model
    best_model_idx = model_performance['ROC AUC'].idxmax()
    best_model = model_performance.loc[best_model_idx]
    print("\nBest Performing Model:\n")
    print(best_model)
    
    # Save the best performing model to a pickle file
    best_model_name, best_model_instance, _ = models[best_model_idx]
    if search_type in ["grid", "random"]:
        best_model_instance = best_model_obj.best_estimator_
    with open(f"{best_model_name}_best_model.pkl", "wb") as file:
        pickle.dump(best_model_instance, file)
    print(f"\nSaved best model ({best_model_name}) to a pickle file.")
    
    # Return the best performing model object for future reuse
    return best_model_instance

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    return accuracy, precision, recall, f1, roc_auc

def downsample_data(X, y, target_records=50000, random_state=42):
    combined_data = pd.concat([X, y], axis=1)
    downsampled_data = combined_data.sample(n=target_records, random_state=random_state)
    X_downsampled = downsampled_data.iloc[:, :-1]
    y_downsampled = downsampled_data.iloc[:, -1]

    return X_downsampled, y_downsampled

def fit_model(search_type, model, grid, X_train, y_train, scoring_metric='ROC AUC'):
    if search_type == "grid":
        model = GridSearchCV(model, grid,
        scoring=scoring_metric.lower(), cv=5, n_jobs=-1, refit=True)
        model.fit(X_train, y_train)
    elif search_type == "random":
        model = RandomizedSearchCV(model, grid, n_iter=10, 
        scoring=scoring_metric.lower(), cv=5, n_jobs=-1, refit=True, random_state=42)
        model.fit(X_train, y_train)
    else:
        model.fit(X_train, y_train)
    return model

def plot_roc_curve(model_name, y_test, y_pred_proba, ax):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")