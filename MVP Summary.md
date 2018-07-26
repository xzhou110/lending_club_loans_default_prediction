
### Domain:
The project will study deafault rate of P2P (Peer to Peer Lending) industry. The key objective of the project is to extract insight of a default. Leverage and train machine with historical default data, and then apply to test data. By the end of project, we will able to do following: 
- Understand the key indicators of default
- Able to predict default with given customers attributes 

### Data:
- Data set1: Historical data from Kaggle (2007-2015)
- Data set2 (Possible): My own P2P investments with deafault 

key data will be used: 
- Loan Satatus: Default or not
- Loan Amount: borrowed amount 
- Payment Plan: 36 vs. 60 months
- Inquiry last 6 months: number of credit requests submitted in last 6 month.
- Annual Income: Salary
- Outstanding Balance: debt owed currenlty
- Grade Level: grade asigned by Lending Club
and etc.


### Known unknowns:
- The data is provied in sqlite format. Need to study the tool before data manipulation.
- Features contained in data may not have direct relationship with Default. This may lead to addtionalreserach and data collection
- Assumptions made here: default pattern hasn't changed througout of years. This needs to be confirmed. 
