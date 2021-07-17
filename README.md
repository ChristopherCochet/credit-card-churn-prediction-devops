# Predict Customer Churn

<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/churn-leaky-bucket.JPG"/> </kbd>

**Project description:** In this project, we implement an end to end solution to identify credit card customers that are most likely to churn (binary classification). Customer information such  as age, salary, marital_status, credit card limit, credit card category etc. are provided. 16.07% of customers in the dataset have churned. The complete project follows PEP8 coding standards and engineering best practices for implementing software and includes a Python package which uses machine learning to predict customer churn. <br>

The main python script is automated to perform the following end to end data science workflow : 

* Loads data from the csv dataset
* Performs EDA
* Performs feature engineerung
* Splits the dataset into training and test sets
* Trains machine learning models using hyper parameter tuning
* Outputs results: classification reports and ROC metric performance
* Exports the models as a pickle files

## 1. Credit card Dataset

The data set contains 10,127 credit card customer information and holds 18 features:

```
* CLIENTNUM : Client number. Unique identifier for the customer holding the account
* Attrition_Flag : Internal event (customer activity) variable - if the account is closed then 1 else 0
* Customer_Age : Demographic variable - Customer's Age in Year
* Gender : Demographic variable - M=Male, F=Female
* Dependent_count : Demographic variable - Number of dependents
* Education_Level : Demographic variable - Educational Qualification of the account holder (example: high school, college graduate, etc.)
* Marital_Status : Demographic variable - Married, Single, Divorced, Unknown
* Income_Category : Demographic variable - Annual Income Category of the account holder
* Card_Category : Product Variable - Type of Card (Blue, Silver, Gold, Platinum)
* Months_on_book : Period of relationship with bank
* Total_Relationship_Count : Total no. of products held by the customer
* Months_Inactive_12_mon : No. of months inactive in the last 12 months
* Contacts_Count_12_mon : No. of Contacts in the last 12 months
* Credit_Limit : Credit Limit on the Credit Card
* Total_Revolving_Bal : Total Revolving Balance on the Credit Card
* Avg_Open_To_Buy : Open to Buy Credit Line (Average of last 12 months)
* Total_Amt_Chng_Q4_Q1 : Change in Transaction Amount (Q4 over Q1)
* Total_Trans_Amt : Total Transaction Amount (Last 12 months)
* Total_Trans_Ct : Total Transaction Count (Last 12 months)
* Total_Ct_Chng_Q4_Q1 :  Change in Transaction Count (Q4 over Q1)
* Avg_Utilization_Ratio : Average Card Utilization Ratio
```

Dataset Kaggle Reference: https://www.kaggle.com/sakshigoyal7/credit-card-customers

## 2. Running the Script - EDA, Data Pipeline and Modelling 

```
    ├───churn_package
    │   │   churn_library.py
    │   │   churn_script_logging_and_tests.py
    │   │   __init__.py
```

In the churn_package directory, the 'churn_library.py' file performs the machine learning pipeline using the following funcation calls flow:

1. import_data() : reads in the customer churn data
2. perform_eda() : performs eda on the churn dataframe
3. encoder_helper() : turn each categorical column into a new column propotion of churn for each category
4. test_perform_feature_engineering() : tests the perform_feature_engineering function
5. train_models() : train, store model results: images + scores and store models
6. classification_report_image() : produces classification report for training and testing results and stores report as image in images folder
7. feature_importance_plot(): creates and stores the feature importances for customer churn

To run the pipeline that trains classifier and saves the outputs excecute the following in the `churn_package` directory <br>
    ```cd churn_package``` <br>
    ```ipython churn_library.py```  

To run a test suite of the pipeline excecute the following in the `churn_package` directory <br>
    ```cd churn_package``` <br>
    ```ipython hurn_script_logging_and_tests.py```


Refer to the following Python file [here](https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/tree/main/images/churn_package\churn_library.py)

## 3. EDA - Key takeaways

The dataset is imbalanced - approx. 16% of customers have churned
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/eda/churn_histogram.png"/> </kbd> 

The distribution of customer age approximately follows a normal distribution:
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/eda/customer_age_histogram.png"/> </kbd> 

Most customers are married : 
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/eda/marital_status_distribution.png"/> </kbd> 

The total transactions per customer distribution seems binomial with peaks at approx. 40 and 75 : 
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/eda/total_trans_ct_histogram.png"/> </kbd>  

* Total_Trans_Ct - Total Transaction Count (Last 12 months) and Total_Amt_Chng_Q4_Q1 - Change in Transaction Amount (Q4 over Q1) have the strongest correlations with churn (though these are weak to moderate and negative)
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/eda/correlation_heatmap.png"/> </kbd> 

## 4. Model Evaluation

A Logistic Regression model and Random Forest model were used for the projects - these are saved in the models directory: 
```
├───models
│       cv_logistic_regression.pkl
│       cv_random_forest.pkl
```

The best model tested was a tuned RandomForest classifier with the following hyperparamaters:
```
'criterion': 'entropy'
'max_depth': 100
'max_features': 'auto'
'n_estimators': 500
```

The model achieved an AUC of 0.99:
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/results/model_performance_roc_curves.png"/> </kbd> 

## 5. Feature Importance

The most iportant three features for predicting customer churn are:
```
1. Total_Trans_Ct : Total Transaction Count (Last 12 months) 
2. Total_Trans_Amt : Total Transaction Amount (Last 12 months)
3. Total_Revolving_Bal : Total Revolving Balance on the Credit Card
```
<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/results/random_forest_feature_importance.png"/> </kbd>

<kbd> <img src="https://github.com/ChristopherCochet/credit-card-churn-prediction-devops/blob/main/images/results/random_forest_feature_shap_values.png"/> </kbd>

## 6. Project Structure

The project structure is shown below:

```
    │   README.md
    │   requirements.txt
    │   __init__.py
    │
    ├───churn_package
    │   │   churn_library.py
    │   │   churn_script_logging_and_tests.py
    │   │   __init__.py
    │
    ├───data
    │       bank_data.csv
    │
    ├───images
    │   │   churn-leaky-bucket.JPG
    │   │   logistic_regression_classification_report.png
    │   │   random_forest_classification_report.png
    │   │
    │   ├───eda
    │   │       churn_histogram.png
    │   │       correlation_heatmap.png
    │   │       customer_age_histogram.png
    │   │       marital_status_distribution.png
    │   │       total_trans_ct_histogram.png
    │   │
    │   └───results
    │           model_performance_roc_curves.png
    │           random_forest_feature_importance.png
    │           random_forest_feature_shap_values.png
    │
    ├───logs
    │       churn_library.log
    │
    ├───models
    │       cv_logistic_regression.pkl
    │       cv_random_forest.pkl
    │
    └───notebooks
        │   churn_notebook.ipynb
        │   Guide.ipynb
```

## Resources Used for This Project

* Udacity Machine Learning DevOps Engineer Nanodegree: [here](https://www.udacity.com/course/machine-learning-dev-ops-engineer-nanodegree--nd0821) <br>
