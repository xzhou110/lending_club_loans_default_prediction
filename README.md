# Lending Club Loans Default Prediction

This project aims to predict the outcomes of Lending Club loans, specifically whether a loan will default or not. Understanding the key features influencing loan outcomes can offer valuable insights into default patterns and aid in predicting future loan performance.

## Project Structure

The project is divided into various scripts and classes for different stages of the data science pipeline, as follows:

1. **Data Preprocessing**: `data_preprocessing.py` handles the data cleanup, transformation, and preprocessing steps required to prepare the data for further analysis.
2. **Feature Selection**: `feature_selection.py` contains functions for selecting the most relevant features for the machine learning models.
3. **Modeling**: `classification_models.py` contains functions for training, evaluating and selecting various classification models.
4. **Machine Learning Pipeline**: `ml_pipeline.py` is a Python class that consolidates all the individual functions from the preprocessing, feature selection, and modeling scripts. This class provides a unified and streamlined interface for the entire machine learning pipeline.

## Project Design

The project begins with data preprocessing to prepare the data for further analysis. The most relevant features are then selected using the Recursive Feature Elimination with Cross-Validation (RFECV) method. Multiple machine learning algorithms are then used to create models for the prediction task. The most promising models are selected for optimization, with multiple iterations performed for hyperparameter tuning.

## Tools Utilized

- **Programming**: The project utilizes Python and a range of data science libraries, including pandas, numpy, scikit-learn, among others.
- **Modeling**: Scikit-learn was used for machine learning model creation. Recursive Feature Elimination with Cross-Validation (RFECV) was used for feature selection.
- **Visualization**: Matplotlib was used for data visualization. Additionally, ROC Curve, Confusion Matrix, and Precision-Recall Curve were used for model performance visualization.

## Data Source

The dataset used for this project is obtained from Kaggle, which consists of Lending Club loan data spanning from 2007 to 2015. The dataset comprises approximately 890K records.

## Algorithms Utilized

The project involved the use of multiple machine learning algorithms, including RandomForestClassifier, DecisionTreeClassifier, BaggingClassifier, SVC, LogisticRegression, and SGD-loc.

## Results and Key Findings

The primary objective of this project is to predict loans likely to result in default. The XGBoost model, without extensive optimization, emerged as the best-performing model, achieving an accuracy rate of 0.9686 and an F1 score of 0.741. 

### Top Influential Features 

- 'initial_list_status_w' had the highest influence on loan default prediction, indicating the importance of the initial listing status of the loan.
- 'inq_last_6mths', which represents the number of inquiries in the last six months (excluding auto and mortgage inquiries), is a key determinant of loan default, indicating that a higher number of recent credit inquiries may correlate with higher risk.
- Other notable features are 'term_ 60 months', suggesting that longer-term loans are more likely to default, and 'home_ownership_RENT', implying that borrowers who rent their homes might be more likely to default compared to homeowners.
- 'last_pymnt_amnt' and 'out_prncp', representing the last payment amount and the outstanding principal, respectively, also have significant influence on the prediction of loan default.

These features, along with others, provide a comprehensive view of the factors that influence loan default, thereby enabling more accurate predictions and risk assessment.

## Future Improvements and Recommendations

Despite its solid performance, the XGBoost model showed a tendency towards overfitting, which could lead to under-identification of default loans in a real-world setting. To address this, future work may involve further model optimization, including hyper-parameter tuning, to improve model generalizability.

Moreover, while the current model provides valuable insights, there are additional factors, such as macroeconomic indicators, that could potentially influence loan default rates. Incorporating such features could further enhance the predictive power of the model. 

Finally, the model could potentially be improved by using more sophisticated techniques, such as ensemble methods or deep learning, as well as by leveraging additional computational resources for more extensive model training and optimization. 