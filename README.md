# ILLINOIS-DATATHON-2025---THE-MEOWS

This README file contains information and instructions regarding the setup of our code, the data our models used, and the commands used to execute our code. 

The aim of our code is to try to forecast customer spending in the fourth quarter based off features provided through the datasets from Synchrony, as well as make attempt at classifying accounts that may need credit line increases based off the predicted spendings, while also factoring in various risk factors associated with an account. 

# Libraries utilized:
 - pandas
 - numpy
 - matplotlib
 - seaborn
 - statsmodels 
 - sklearn
 - xgboost

# Functions involved:
- ## Pandas
-  pd.read_csv()
- pd.to_datetime()
- DataFrame.groupby()
- DataFrame.sum()
- DataFrame.reset_index()
- DataFrame.rename()
- DataFrame.merge()
- DataFrame.fillna()
- DataFrame.drop()
- DataFrame.isna()
- DataFrame.describe()
- DataFrame.sort_values()
- DataFrame.set_index()
- DataFrame.resample()

- ## Numpy:
- np.sqrt()

- ## Scikit-learn:
- train_test_split()
- mean_squared_error()
- r2_score()
- mean_absolute_error()
- RandomizedSearchCV()

- ## XGBoost:
- XGBRegressor()

- ## SciPy:
- stats.randint()
- stats.uniform()

- ## Matplotlib:
- plt.scatter()
- plt.plot()
- plt.show()

- ## Seaborn:
- sns.histplot()

- ## Statsmodels:
- adfuller()
- AutoReg()
- smf.ols()

## Links to documentation of functions
https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBRegressor
https://medium.com/@rithpansanga/optimizing-xgboost-a-guide-to-hyperparameter-tuning-77b6e48e289d
https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
https://www.geeksforgeeks.org/pandas-series-dt-date/
https://www.statology.org/dickey-fuller-test-python/
https://www.statsmodels.org/dev/generated/statsmodels.tsa.ar_model.AutoReg.html


# Data 
As mentioned before, all the csv datasets utilized in this project was provided by Synchrony through the datathon event. Here are all the datasets given and as titled in a mapping document also provided to do this project: 

account_dim_20250325.csv -> Card Holder Account 
statement_fact_20250325.csv -> Statement Data
transaction_fact_20250325.csv -> Transaction Data
syf_id_20250325.csv -> Customer Identifier
rams_batch_cur_20250325.csv -> Account Level Features
fraud_claim_case_20250325.csv -> Fraudulent Activities reported at customer
fraud_claim_tran_20250325.csv -> Fraudlent Transactions reported on a given account
wrld_stor_tran_fact_20250325.csv -> World Transactions Data

*we also renamed the files locally for our own ease of access*

## Data used and it's utilization
For this project, we made two models to predict customer spending and made one classification model. 
One the prediction models, the AR_model (Auto Regressive model), utilized only one dataset which was the Transaction dataset. 
The other prediction model, the xgb model, utilized the Transaction Data, World Transactions Data, Account Level Feature data.

- For the xgb model, in summary, this model aimed to predict the total amount spent in a quarter, which would represent an anticipated amount of spending in the fourth quarter of 2025. We filtered transactions only by 'payment' type to predict as to simplfy predictions and also because we felt that it made the most sense since we only want to factor in total transactions and not the other types of transactions that involved returns and payments. 

  - The primary variables the model used to make its predictions:
    - total amount spent in a year
    - total amount spent within the last 8 months
    - total spent per month
    - total spent per quarter
    - difference between a customers old and new credit behavior score 
    - Average card utilization within the last 3 months

 - This approach started by creating a dataframe, specifically for this model dubbed 'all_transactions', that would have all the features we want to select. The variables, although scattered around the different dataframed, are all linked to a common attribute which is dubbed 'current_account_nbr' which represents a customers account number. This led us to start by using the 'groupby()' commands from the pandas library to group customer ids with the various attributes which is better illustrated when you read the code. There are a few instances where we needed to do a bit more work such as renaming columns using the 'rename()' function to make accessing columns for intuitive and easier or most importantly, the feature engineering needed to be done to actually make these variables. the 'total' variables are all derivatives from the total amount spent in a year with little calculations to make the new 'total' variables. 

- For the AR_model .... 







## Unlisted YouTube Presentation:
  Presentation Link - 



