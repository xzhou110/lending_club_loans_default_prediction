# Lending Club Loans Default Prediction

## Project Overview

This project aims to predict the outcomes of Lending Club loans, specifically whether a loan will default or not. Understanding the key features influencing loan outcomes can offer valuable insights into default patterns and aid in predicting future loan performance.

The project follows a structured data science pipeline which includes data preprocessing, feature selection, modeling, and evaluation. The pipeline utilizes Python and a range of data science libraries, such as pandas, numpy, and scikit-learn.

## Repository Structure

The repository is organized into main files (Jupyter notebooks) and utility scripts:

### Main Files:

1. **Data Cleansing**: `01_data_cleansing_new.ipynb` outlines the initial data exploration and cleansing process.

2. **Modeling**: `02_modeling_new.ipynb` focuses on feature selection, modeling, and model evaluation. 

3. **Modeling Using ML Pipeline**: `03_modeling_using_ml_pipeline.ipynb` demonstrates the use of the ML pipeline class, `ml_pipeline.py`, for a more streamlined modeling process.

### Utility Scripts:

1. **Data Preprocessing**: `data_preprocessing.py` handles the data cleanup, transformation, and preprocessing steps required to prepare the data for further analysis.

2. **Feature Selection**: `feature_selection.py` contains functions for selecting the most relevant features for the machine learning models.

3. **Modeling**: `classification_models.py` contains functions for training, evaluating, and selecting various classification models.

4. **Machine Learning Pipeline**: `ml_pipeline.py` is a Python class that consolidates all the individual functions from the preprocessing, feature selection, and modeling scripts. This class provides a unified and streamlined interface for the entire machine learning pipeline.

## Project Design

The project begins with data preprocessing to prepare the data for further analysis. The most relevant features are then selected using the Recursive Feature Elimination with Cross-Validation (RFECV) method. Multiple machine learning algorithms are then used to create models for the prediction task. The most promising models are selected for optimization, with multiple iterations performed for hyperparameter tuning.

## Data Source

The dataset used for this project is obtained from Kaggle, which consists of Lending Club loan data spanning from 2007 to 2015. The dataset comprises approximately 890K records.

## Algorithms Utilized

The project involved the use of multiple machine learning algorithms, including RandomForestClassifier, DecisionTreeClassifier, BaggingClassifier, SVC, LogisticRegression, and XGBoost.

## Results and Key Findings

The primary objective of this project is to predict loans likely to result in default. Consequently, the model performance was evaluated with a focus on F1 score, considering the class imbalance present in the dataset. The XGBoost model, without extensive optimization, emerged as the best-performing model, achieving an accuracy rate of 0.9686, and an F1 score of 0.741.

Key features influencing loan defaults were identified, including initial listing status, inquiries in the last 6 months, verification status, term of the loan, home ownership status, last payment amount, outstanding principal, loan purpose, and employment length. These features, along with others, provide a comprehensive view of the factors that influence loan default, thereby enabling more accurate predictions and risk assessment.

Detailed model performance metrics and best-performing model statistics are included in the `02_modeling_new.ipynb` and `03_modeling_using_ml_pipeline.ipynb` notebooks.

## Future Improvements and Recommendations

Despite its impressive performance, the XGBoost model showed a tendency towards overfitting, which could lead to under-identification of default loans in a real-world setting. To address this, future work

could involve refining the model through further hyperparameter tuning and employing techniques to better handle the class imbalance, such as oversampling the minority class or undersampling the majority class.

Increasing the computational resources could also allow for more extensive optimization and the exploration of more complex models, potentially leading to further improvements in performance.

## Tools Utilized

- **Programming**: The project utilizes Python and a range of data science libraries, including pandas, numpy, and scikit-learn, among others.

- **Modeling**: Scikit-learn was used for machine learning model creation. Recursive Feature Elimination with Cross-Validation (RFECV) was used for feature selection.

- **Visualization**: Matplotlib was used for data visualization. Additionally, ROC Curve, Confusion Matrix and Precision-Recall Curve were used for model performance visualization.

## Conclusion

Through systematic data preprocessing, feature selection, and model training, this project demonstrates the feasibility and potential effectiveness of using machine learning to predict loan defaults. The insights gained from the analysis of feature importances can also guide risk assessment and decision-making processes in lending practices. While there are areas for improvement, the results thus far are promising and indicate the value of further research and development in this area.

