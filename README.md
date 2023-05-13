# Lending Club Loans Default Prediction

This project aims to predict the outcomes of Lending Club loans, specifically whether a loan will default or not. Understanding the key features influencing loan outcomes can offer valuable insights into default patterns and aid in predicting future loan performance.

## Project Structure

The project is divided into various notebooks and utility scripts for different stages of the data science pipeline, as follows:

1. **Data Cleansing and Transformation**: `data_cleansing_new.ipynb` script handles the data cleanup and transformation steps required to prepare the data for further analysis.

2. **Machine Learning Analysis and Visualization**: `modeling_new.ipynb` script contains the machine learning analysis and visualization of the processed data.

3. **Utilities**: These scripts provide auxiliary functions for the project.
   - `data_preprocessing.py`: Contains functions for data preprocessing.
   - `feature_selection.py`: Contains functions for feature selection.
   - `classification_models.py`: Contains functions for training and evaluating various classification models.

## Project Design

The project began with data cleanup and transformation to prepare the data for further analysis. Multiple machine learning algorithms were then used to create baseline models for the prediction task. The most promising models were selected for optimization, with multiple iterations performed for hyperparameter tuning.

## Tools Utilized

- **Programming**: The project utilizes Python and a range of data science libraries, including pandas, numpy, matplotlib, among others.
- **Modeling**: Scikit-learn was used for machine learning model creation.
- **Visualization**: Matplotlib was used for data visualization. Additionally, ROC Curve and Confusion Matrix were used for model performance visualization.

## Data Source

The dataset used for this project is obtained from Kaggle, which consists of Lending Club loan data spanning from 2007 to 2015. The dataset comprises approximately 890K records.

## Algorithms Utilized

The project involved the use of multiple machine learning algorithms, including Random Forest, Decision Tree, Bagging Classifier, SVC, Logistic Regression, and SGD-loc.

## Results and Key Findings

The primary objective of this project is to accurately predict loans likely to result in default. Consequently, the model performance was evaluated with a focus on the Recall or True Positive Rate. 

The best-performing model achieved an overall accuracy rate of 62.4%, with a True Positive Rate of 72%. This means the model was able to accurately predict 72% of loans that ended up in default.

Key features influencing loan defaults were identified, including high debt-to-income ratio, low grade level assigned by Lending Club, and the purpose of the loan (particularly for debt consolidation). Interestingly, the model indicated that using high-interest loans for refinancing, which is generally not a financially sound decision, is a negative indicator.

One potential area for improvement is that the model currently tends to over-predict negative outcomes, which could lead to a more conservative than desired investment portfolio.