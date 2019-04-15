# Lending Club Loans Outcomes Prediction

### File Instructions: 

1. Data Clean-up.ipynb: Data cleanup and transformation
2. Modeling.ipynb: Machine Learning Analysis and visualization
3. Lending Club_final.pptx: Presentation and summarized finding

### Background and Scope:

The project is to predict outcomes of Lending Club loans (default or not).  The project will help gain insights of defaults, understand key features of loans, and predict outcomes with given features. 

### Project design

Started with data cleanup and transformation to prepare data for further analysis. Modeled with multiple algorithms to identify the best baseline models. Optimized selected baseline models for performance. Multiple iterations were performed for hyper-parameters tuning.

### Tools
- Programming: Python and data science libraries, including pandas, numpy, matplotlib, etc.
- Modeling: Scikit-learn Algorithms
- Visualization: Matplotlib, ROC Curve and Confusion Matrix

### Data

The data used for analysis is from Kaggle, covering Lending Club loans from year 2007 to 2015. The data contains 890K records.

### Algorithm(s)

Modeled with multiple algorithms including: Random Forest, Decision Tree, Bagging Classifier, SVC, Logistic Regression, and SGD-loc. 

### Result

Negative outcomes are 7.6% of total outcomes. The model has an Accuracy Rate of 62.4%, and True Positive Rate of 72%. In another word, the model was able to detect 72% out of 7.6% negative outcomes, with 28% undetected. Improvement area: the model tends to over-classify loan outcomes as negative outcomes. 

Key indicators of negative outcomes were identified, such as high debt-to-income-ratio, low grade level assigned by lending club, and purposes of loan (consolidation). Lending Club Loans typically demand 8%+ interest rate. Using high interest loans to consolidate debt is not a financial-sound decision and thus regarded as negative indicator.

### Challenges and After-thoughts

Data Transformations

1. Highly imbalanced classes. Positive outcomes are ~92% (~80K records) and negative outcomes are ~7% (~66K records). Under-sampling techniques has been used to conquer imbalanced classes

2. Many categorical features. Take a few has an example, purposes of loans, types of homeownerships, etc. The features were transformed to dummy variables for further modeling

3. Combining features. A few features need to be combined to provide meaningful information for loans. Take asset and debt level as an example. The actual amounts can vary considerably from loan to loan. In order to provide meaningful loan-related information, we combined and converted them to debt-to-income ratios, and income-to-payment coverage ratios as a result.
