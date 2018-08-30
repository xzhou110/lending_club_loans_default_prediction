# Lending Club Loans Outcomes Prediction

### File Instructions: 

1. Data Clean-up.ipynb: Clean up and data transformation
2. Modeling.ipynb: Machine Learning Analysis and visualization
3. Lending Club_final.pptx: Presentation and summarized finding

### Background and Scope:

The project is to predict whether outcomes of Lending Club loans will become negative (default or late for payment).  The project will help gain insights of negative outcomes, understand key indicators, and be able to predict outcomes with given indicators. 

### Project design

Started with data transformation so data are in appropriate type and can be used for prediction. Model with multiple algorithms to identify best baseline models. Optimize selected baseline models for best performance. Multiple iterations are required with different feature combinations.

### Tools
- Programming: Python and common data science libraries, including pandas, numpy, etc.
- Data Preparation: Scikit-learn
- Modeling: Scikit-learn Algorithms
- Visualization: Matplotlib, Scikit-learn ROC Curve and Confusion Matrix

### Data

The data used for analysis is from Kaggle, covering Lending Club loans from year 2007 to 2015. The data contains 890K records.

### Algorithm(s)

Modeling with multiple algorithms including: Random Forest, Decision Tree, Bagging Classifier, SVC, Logistic Regression, and SGD-loc. Random Forest was identified as best algorithm to predict the outcomes. 

### Result

Negative outcomes are 7.6% of total outcomes. The model has an accuracy of 62.4%, and True Positive Rate of 72%. In another word, the model was able to detect 72% of 7.6% negative outcomes, with 28% undetected. Improvement area: the model tends to over-classify loan outcomes as negative outcomes but actually are not. 

Key indicators of negative outcomes were identified, such as high debt-to-income-ratio, low grade level assigned by lending club, and purposes of loan (consolidation). Lending Club Loans typically demand 8%+ interest rate. Using high interest loan to consolidate debt is not a financial-sound decision and thus regarded as negative indicator.

### Challenges and After Thoughts

Data Transformations

1. Data has highly imbalanced classes. Positive outcomes are ~92% (~80K records) and negative outcomes are ~7% (~66K records). Under-sampling techniques has been used to conquer imbalanced classification

2. Lots of features are categorical in nature. Take a few has an example, purposes of loans, types of homeownerships, etc. The features were transformed to captured the information

3. Transforming features. Many features do not convey loan-level information when read stand-alone. Take asset and debt level as an example. They convey borrower information and can vary considerably from loan to loans. In order to have them provide loan-level information, creative transformation is needed.  We converted them to Debt-to-income ratios, and income-to-payment coverage ratios as a result.

Feature Engineering

The model itself takes 1-2 hours to run for each iteration. Iterated through multiple rounds to select best features combinations. Increased level of feature transformation may not lead to better model performance. Had to revert back to earlier version to optimize model performance. Lesson learned: need to keep records of multiple iterations for more efficient features selection and fine-tuning.
