import os
import time
import datetime
import pickle
import logging
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import (cm, colors as mcolors)

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc)
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)


class MLPipeline:
    """
    A class used to preprocess data for machine learning.


    Attributes
    ----------
    data_raw : pandas.DataFrame
        Raw data before preprocessing.
    data_processed : pandas.DataFrame
        Data after preprocessing.
    label_encoding : dict
        Dictionary mapping original class labels to encoded labels.
    ml_type : str
        Type of machine learning problem ('classification', 'regression', 'clustering').
    selected_features : list
        List of selected features after feature selection process.
    best_model_instance: object
        The trained model object that is found to have the best performance during hyperparameter tuning process.
    
    Methods
    -------
    preprocess_data(missing_threshold, max_unique_values_cat, correlation_threshold, n_features_to_select, estimator, n_samples)
        Preprocess the data by handling missing values, encoding categorical variables, handling date columns, etc.
    encode_labels(df, label)
        Encode the class labels in the data.
    drop_missing_value_columns(df, missing_threshold)
        Drop columns with a percentage of missing values higher than a specified threshold.
    process_date_columns(df)
        Process date columns by converting them to the number of days relative to the current date.
    handle_missing_values_numerical(df)
        Handle missing values in numerical columns by replacing them with the median of the column.
    handle_extreme_values_numerical(df)
        Handle extreme values in numerical columns by winsorizing them.
    create_dummy_variables(df, max_unique_values_cat)
        Create dummy variables for categorical columns with a number of unique values lower than a specified threshold.
    handle_high_correlation(df, correlation_threshold)
        Remove one of a pair of features with a correlation higher than a specified threshold.
    standardize_numerical_columns(df)
        Standardize numerical columns to have a mean of 0 and a standard deviation of 1.
    clean_feature_names(df)
        Clean feature names to ensure compatibility with XGBoost.
    train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        Train and evaluate a model, returning several metrics.
    downsample_data(X, y, target_records=50000, random_state=42)
        Downsample the data to a specified number of records.
    fit_model(search_type, model, grid, X_train, y_train, scoring_metric='ROC AUC')
        Fit a model using either GridSearchCV or RandomizedSearchCV, or without any hyperparameter tuning.
    plot_roc_curve(model_name, y_test, y_pred_proba, ax)
        Plot the ROC curve for a model.
    plot_top_n_features(n=10)
        Plot the top n features of importance for the best model.
    """    
    
    def __init__(self, data, ml_type='classification'):
        self.data = data
        self.data_processed = None
        self.ml_type = ml_type
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.selected_features = None
        self.best_model_instance = None
        self.top_n_features = None
    
    def preprocess_data(self, label='label', missing_threshold=0.9, max_unique_values_cat=50, correlation_threshold=0.9):
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

        df = self.data.copy()

        # Separate out the target column and the feature columns
        y = df[label] if df[label] is not None else None
        X = df.drop(columns=[label])

        # Encoding labels and printing out the encoding
        y, label_encoding = self.encode_labels(y)
        logging.info(f"Label encoding: {label_encoding}")

        # Dropping columns with missing values above the threshold
        X = self.drop_missing_value_columns(X, missing_threshold)
        logging.info("Dropped columns with missing values above the threshold")

        # Processing date columns and renaming them
        X = self.process_date_columns(X)
        logging.info("Processed date columns")

        # Handling missing values for numerical columns
        X = self.handle_missing_values_numerical(X)
        logging.info("Handled missing values for numerical columns")

        # Handling extreme values for numerical columns
        X = self.handle_extreme_values_numerical(X)
        logging.info("Handled extreme values for numerical columns")

        # Creating dummy variables for categorical columns
        X = self.create_dummy_variables(X, max_unique_values_cat)
        logging.info("Created dummy variables for categorical columns")

        # Handling high correlation among numerical columns
        X = self.handle_high_correlation(X, correlation_threshold)
        logging.info("Handled high correlation among numerical columns")

        # Standardizing numerical columns
        X = self.standardize_numerical_columns(X)
        logging.info("Standardized numerical columns")

        # Clean feature names for XGBoost
        X = self.clean_feature_names_for_xgboost(X)
        logging.info("Cleaned feature names for XGBoost")

        # Combine the processed features and the label into a single DataFrame
        X.reset_index(drop=True, inplace=True)
        result_df = pd.concat([X, pd.Series(y, name=label)], axis=1)

        # Split the data into train and test sets
        X_processed = result_df.drop(columns=['label'])
        y_processed = result_df['label']
                                     
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_processed, y_processed, test_size=0.3, random_state=42)
        self.data_processed = result_df

        return result_df

    def select_features(self, min_features_to_select=10, n_features_to_select=None, ml_type='classification', estimator=None, n_samples=None):
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
        X_train = self.X_train
        y_train = self.y_train if self.y_train is not None else None
        ml_type = self.ml_type
 
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
        
        self.selected_features = selected_features
        return selected_features
    
    def run_classification_models(self, n_samples=None, search_type=None, scoring_metric='roc auc'):
        """
        Train and evaluate multiple classification models on the given dataset.

        This function trains and evaluate five different classification models: Logistic Regression, K-Nearest Neighbors,
        Support Vector Machine, Random Forest, and XGBoost. It also performs hyperparameter tuning if specified and plots
        ROC curves for all models. It evaluates model performance based on user defined scoring metric. It saves the best 
        performing model into results folder.

        Parameters
        ----------
        X_train : pd.DataFrame
            The training feature matrix (X) as a pandas DataFrame.
        y_train : pd.Series
            The target variable for training (y) as a pandas Series.
        X_test : pd.DataFrame
            The testing feature matrix (X) as a pandas DataFrame.
        y_test : pd.Series
            The target variable for testing (y) as a pandas Series.
        n_samples : int, optional, default: None
            Number of samples to downsample the data to. If None, no downsampling is performed.
        search_type : str, optional, default: None
            The type of hyperparameter search to perform. Choices are 'grid', 'random', or None. Default is None.
        scoring_metric : str, optional, default: 'roc auc'
            The scoring metric to optimize during hyperparameter tuning and to select the best model. 
            Some options are 'accuracy', 'precision', 'recall', 'f1', and 'roc auc'.

        Returns
        -------
        best_model_instance : object
            The best performing model instance (fitted) according to the highest score of the scoring_metric.
        """
        
        X_train, y_train, X_test, y_test = self.X_train[self.selected_features], self.y_train, self.X_test[self.selected_features], self.y_test
        
        # Handling class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # Downsample data if n_samples is provided
        if n_samples is not None:
            X_train, y_train = self.downsample_data(X_train, y_train, target_records=n_samples)

        # Hyperparameter grids
        lr_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [500, 1000, 3000]}
        knn_grid = {'n_neighbors': list(range(1, 31))}
        svm_grid = {'C': [0.1, 1, 10], 'gamma': [0.1, 1, 10]}
        rf_grid = {'n_estimators': [100, 200], 'max_depth': [None, 10, 20]}
        xgb_grid = {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200], 'max_depth': [3, 6]}

        # Create a list of models to iterate through
        models = [
            ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000), lr_grid),
            ('KNN', KNeighborsClassifier(), knn_grid),
            ('SVM', SVC(random_state=42, probability=True), svm_grid),
            ('Random Forest', RandomForestClassifier(random_state=42), rf_grid),
            ('XGBoost', XGBClassifier(random_state=42, use_label_encoder=False), xgb_grid)
        ]

        # Pre-process the scoring_metric argument
        scoring_metric = scoring_metric.title() if scoring_metric.lower() != 'roc auc' else 'ROC AUC'
        model_performance = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'])

        performance_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        if scoring_metric not in performance_metrics:
            raise ValueError(f"Invalid scoring_metric. Expected one of: {performance_metrics}")

        best_model_obj = None
        best_model_performance = None

        plt.figure(figsize=(16, 9))
        ax = plt.gca()

        # Iterate through the models and evaluate their performance
        for model_name, model, grid in models:
            start_time = time.time()
            model = self.fit_model(search_type, model, grid, X_train, y_train, scoring_metric)
            exec_time = time.time() - start_time
            print(f"{model_name}: Done (Execution Time: {exec_time:.2f} seconds)")

            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Train and evaluate the model
            accuracy, precision, recall, f1, roc_auc = self.train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
            performance_metrics_values = [accuracy, precision, recall, f1, roc_auc]

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
            self.plot_roc_curve(model_name, y_test, y_pred_proba, ax)

        # Display ROC curve comparison
        plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 0.5)")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14)
        plt.ylabel("True Positive Rate", fontsize=14)
        plt.title("ROC Curve Comparison", fontsize=16, weight='bold')
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

        # Create a timestamped filename
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create result directory if not exists
        result_dir = "./results"
        os.makedirs(result_dir, exist_ok=True)

        with open(f"./{result_dir}/{best_model_name}_best_model_{current_time}.pkl", "wb") as file:
            pickle.dump(best_model_instance, file)
        print(f"\nSaved best model ({best_model_name}) to a pickle file.")

        # Return the best performing model object for future reuse
        self.best_model_instance = best_model_instance
        
        return best_model_instance
    
    def encode_labels(self, y):
        y, unique_labels = pd.factorize(y)
        label_encoding = {label: idx for idx, label in enumerate(unique_labels)}
        return y, label_encoding

    def drop_missing_value_columns(self, X, missing_threshold):
        return X.loc[:, X.isna().mean() < missing_threshold]

    def process_date_columns(self, X):
        date_columns = [col for col in X.columns if 'date' in col.lower()]

        for col in date_columns:
            X[col] = (datetime.datetime.now() - pd.to_datetime(X[col])).dt.days
            X.rename(columns={col: col + '_days'}, inplace=True)

        return X

    def handle_missing_values_numerical(self, X):
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()

        for col in numerical_columns:
            X[col].fillna(X[col].median(), inplace=True)

        return X

    def handle_extreme_values_numerical(self, X):
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        lower_bound = X[numerical_columns].quantile(0.05)
        upper_bound = X[numerical_columns].quantile(0.95)

        for col in numerical_columns:
            X[col] = np.where(X[col] < lower_bound[col], lower_bound[col], X[col])
            X[col] = np.where(X[col] > upper_bound[col], upper_bound[col], X[col])

        return X

    def create_dummy_variables(self, X, max_unique_values_cat):
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        categorical_columns_to_drop = [col for col in categorical_columns if X[col].nunique() > max_unique_values_cat]
        X.drop(columns=categorical_columns_to_drop, inplace=True)
        remaining_categorical_columns = list(set(categorical_columns) - set(categorical_columns_to_drop))
        X = pd.get_dummies(X, columns=remaining_categorical_columns, dummy_na=True, drop_first=True)

        return X

    def handle_high_correlation(self, X, correlation_threshold):
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        corr_matrix = X[numerical_columns].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
        columns_to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
        X.drop(columns=columns_to_drop, inplace=True)
        return X

    def standardize_numerical_columns(self, X):
        numerical_columns = X.select_dtypes(include=['number']).columns.tolist()
        scaler = StandardScaler()
        X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
        return X

    def clean_feature_names_for_xgboost(self, X):
        """
        Clean column names to meet the requirements of XGBoost.
        XGBoost (at least the version used at the time of writing) does not accept feature names with special characters like '<', '[' or ']'.
        This function replaces these special characters with corresponding text representations.
        """
        X.columns = X.columns.astype(str).str.replace('[', '_replace_bracket_open_', regex=True).str.replace(']', '_replace_bracket_close_', regex=True).str.replace('<', '_smaller_than_', regex=True)
        return X

    def train_and_evaluate_model(self, model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        return accuracy, precision, recall, f1, roc_auc


    def downsample_data(self, X, y, target_records=10000, random_state=42):
        if len(X) <= target_records:
            return X, y

        combined_data = pd.concat([X, y], axis=1)
        downsampled_data = combined_data.sample(n=target_records, random_state=random_state)
        X_downsampled = downsampled_data.iloc[:, :-1]
        y_downsampled = downsampled_data.iloc[:, -1]

        return X_downsampled, y_downsampled


    def fit_model(self, search_type, model, grid, X_train, y_train, scoring_metric='ROC AUC'):
        # Replace spaces with underscores and convert to lowercase
        if scoring_metric.lower() == 'roc auc':
            scoring_metric = scoring_metric.replace(' ', '_').lower()
        elif scoring_metric.lower() == 'f1 score':
            scoring_metric = 'f1'
        else:
            scoring_metric = scoring_metric.lower()

        if search_type == "grid":
            model = GridSearchCV(model, grid,
            scoring=scoring_metric, cv=5, n_jobs=-1, refit=True)
            model.fit(X_train, y_train)
        elif search_type == "random":
            model = RandomizedSearchCV(model, grid, n_iter=10, 
            scoring=scoring_metric, cv=5, n_jobs=-1, refit=True, random_state=42)
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)
        return model


    def plot_roc_curve(self, model_name, y_test, y_pred_proba, ax):
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.4f})")


    def plot_top_n_features(self, n=10):
        """
        Plot the top n features of importance for a given model.

        Parameters
        ----------
        model : object
            The model object.
        X : pd.DataFrame
            The feature matrix (X) as a pandas DataFrame.
        n : int, optional, default: 10
            The number of top features to plot.

        Returns
        -------
        None
        """
        
        model = self.best_model_instance
        X = self.X_train
        
        try:
            importances = model.feature_importances_
        except AttributeError:
            try:
                # If the model is a linear model, use absolute value of coefficients as importances
                if isinstance(model, LogisticRegression) or (isinstance(model, SVC) and model.kernel == 'linear'):
                    importances = np.abs(model.coef_[0])
                else:
                    print("The model does not have 'feature_importances_' or 'coef_' attribute.")
                    return
            except AttributeError:
                print("The model does not have 'feature_importances_' or 'coef_' attribute.")
                return

        # Get feature importances and indices
        indices = np.argsort(importances)[::-1]
        top_n_indices = indices[:n]

        # Get the feature names from X_processed
        feature_names = X.columns.tolist()

        self.top_n_features = {feature_names[i]: importances[i] for i in top_n_indices}

        # Prepare colors. Create a custom colormap that goes from medium green to light blue.
        # cmap = mcolors.LinearSegmentedColormap.from_list("n",["#008000", "#0000CD"])
        colors = cm.summer(np.linspace(0, 1, n))[::-1]

        # Plot the feature importances of the model
        plt.figure(figsize=(12, 9))
        plt.title(f"Top {n} Feature Importances", fontsize=16,  weight='bold')
        plt.barh(range(n), importances[top_n_indices][::-1], color=colors, align='center')  # Only plot top n
        plt.yticks(range(n), [feature_names[i] for i in top_n_indices][::-1], fontsize=12)  # Only plot top n, reversed for aesthetics
        plt.xlabel("Relative Importance", fontsize=14)
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()

        return self.top_n_features
