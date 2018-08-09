# Lending Club Loans Outcomes Prediction

### File Instructions: 

1. Data Clean-up.ipynb: Clean up and data transformaton
2. Modeling.ipynb: Machine Learning Analysis and visualization
3. Lending Club_final.pptx: Presentation and summarized finding

### Background and Scope:

The project is to predict whether outcomes of Lending Club loans will become negative (default or late for payment).  The project will help gain insights of negative outcomes, understand key indicators, and be able to predict outcomes with given indicators. 

### Project design

Iterate through multiple models in

### Tools
- Programming: Python and common data science libraries, including pandas, numpy, etc.
- Data Prepration: Scikit-learn
- Modeling: Scikit-learn Algorithms
- Visualisation: Matplotlib, Scikit-learn ROC Curve and Confusion Matrix

### Data

The data used for analysis is from Kaggle, covering all loans in Lending Club from year 2007 to 2015. The data contains 890K records.

### Algorithm(s)

Modeling with multiple algorithms including: Random Forest, Decision Tree, Bagging Classifier, SVC, Logistic Regression, and SGD-loc. Random Forest was identified as best algorithm to predict the outcomes. 

### Result

Negative outcomes are 7.6% of total outcomes. The model has an accuracy of 62.4%, and True Positive Rate of 72%. In another word, the model was able to decte 72% of 7.6% negative outcomes, with 28% undetected. Improvement area: the model tend to over-classify loans as neative coutcomes but actually are not. 

Key indicators of negative outcomes were identified, such as high debt-to-income-ratio, low grade level assigned by lending club, and purposes of loan (consolidation). Lending Club Loans typically demand 8%+ intrest rate. Using high interest loan to consolidate dabt is not a financial sound decision and thus regarded as negative indicator.

### Challenges and After Thoughts

- Data Transformations

1. Data has highly imblanced classess. Positive outcomes are ~92% (~80K records) and negative outcomes are ~7% (~66K records). Undersampling techniques has been used to counter imblanced classsification

2. Lots of features are categorical in nature. Take a few has an example, purposes of loans, types of homownership, etc. The feautres were transformed to captured the informaiton

3. Transoforming features. Many features do not convey full picutre of specific loan when stand-alone. Take asset and debt level as an example. They can vary considerably from loan to loan, but may not provide full picture of the specifc loan. Transformation to debt-to-income ratios, and income-to-payment coverage ratios need to be applied to extact loan specific information. 

- Feature Engineering

The model itself takes 1-2 hours to run each iterations. Iterate through multiple rounds to select best features combinations. Increased level of feature transformation may not lead to better model performance. Had to revert back to earlier version to optimze model performance. Lesson learned: records of multiple iterations need to kept for more efficient features selectoin .






