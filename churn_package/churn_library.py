# library doc string

"""

This module contains functions used t perfom and end to end credit card churn 
prediction model (binary classification) and assess the best results obtained.

This contains the following functions:

* import_data() : returns dataframe for the csv found at pth
* perform_eda() : perform eda on churn_df and save figures to images folder
* encoder_helper() : turn each categorical column into a new column with
                    propotion of churn for each category
* perform_feature_engineering() : tests the perform_feature_engineering function
* feature_importance_plot: creates and stores the feature importances in pth
* classification_report_image : produces classification report for training and testing results 
                                and stores report as image in images folder
* train_models() : train, store model results: images + scores, and store models

"""

# import libraries

import logging
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sns.set()


# default images saved layout
rcParams.update({'figure.autolayout': True})

# set logging configurations
logging.basicConfig(
    filename='../logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

IMG_FOLDER = "../images/"
FOLDER_EDA = "eda/"
FOLDER_RESULTS = "results/"


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            churn_df: pandas dataframe
    '''
    logging.info("import_data() - with param %s called" % pth)

    try:
        churn_df = pd.read_csv(pth)
        print("import_data() - dataframe loaded has shape %s" % (churn_df.shape,))

        churn_df['Churn'] = churn_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)
        return churn_df
    except FileNotFoundError:
        logging.error("File not found %s - check your path and file name" % pth)


def perform_eda(churn_df, img_folder=IMG_FOLDER+FOLDER_EDA):
    '''
    perform eda on churn_df and save figures to images folder
    input:
            churn_df: pandas dataframe

    output:
            None
    '''
    logging.info("perform_eda() - with param churn_df shape %s called" % (churn_df.shape,))

    plt.figure(figsize=(20, 10))
    # Churn histogram
    img_file_path = img_folder + 'churn_histogram.png'
    churn_df['Churn'].hist()
    plt.savefig(img_file_path)
    print("perform_eda(): churn histogram saved to {}".format(img_file_path))

    # Customer Age distribution
    img_file_path = img_folder + 'customer_age_histogram.png'
    churn_df['Customer_Age'].hist()
    plt.savefig(img_file_path)
    print("perform_eda(): customer age histogram saved to {}".format(img_file_path))

    # Marital Status distribution (in percent)
    img_file_path = img_folder + 'marital_status_distribution.png'
    churn_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(img_file_path)
    print("perform_eda(): Marital Status distribution saved to {}".format(img_file_path))

    # Total CT distribution
    img_file_path = img_folder + 'total_trans_ct_histogram.png'
    sns.distplot(churn_df['Total_Trans_Ct'])
    plt.savefig(img_file_path)
    print("perform_eda(): Total CT distribution saved to {}".format(img_file_path))

    # Dataframe Correlation Heatmap
    img_file_path = img_folder + 'correlation_heatmap.png'
    sns.heatmap(churn_df.corr(), cmap='Blues', annot=True, linewidths=2)
    plt.savefig(img_file_path)
    print("perform_eda(): dataframe correlation heatmap saved to {}".format(img_file_path))


def encoder_helper(churn_df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            churn_df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            churn_df: pandas dataframe with new columns for
    '''

    logging.info("encoder_helper() with param churn_df shape %s called with category list %s" \
            % (churn_df.shape, category_lst))

    # create a dictionary of categorical columns and their churn means
    # create new columns for each category and map their churn mean values
    for col in category_lst:
        cat_dict = {}
        cats = churn_df[col].unique()
        for cat in cats:
            cat_dict[cat] = churn_df[churn_df[col] == cat]['Churn'].mean()
        churn_df[col + "_Churn"] = churn_df[col].map(cat_dict)

    print("encoder_helper(): encoded dataframe has shape {}" \
        .format(churn_df.shape))
    return churn_df


def perform_feature_engineering(churn_df, response):
    '''
    input:
              churn_df: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              X_churn_train: X training data
              X_churn_test: X testing data
              y_churn_train: y training data
              y_churn_test: y testing data
    '''
    logging.info("perform_feature_engineering() with param churn_df shape %s called"  \
                % (churn_df.shape,))

    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book',
        'Total_Relationship_Count', 'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]

    y_churn = churn_df['Churn']
    X_churn = pd.DataFrame()
    X_churn[keep_cols] = churn_df[keep_cols]

    # train test split
    X_churn_train, X_churn_test, y_churn_train, y_churn_test = \
        train_test_split(X_churn, y_churn, test_size=0.3, random_state=42)
    print("perform_feature_engineering(): \
          Xtrain shape {}, X_test  shape {}, y_train shape {},y_test shape {}" \
         .format(X_churn_train.shape, X_churn_test.shape, y_churn_train.shape, y_churn_test.shape))

    return(X_churn_train, X_churn_test, y_churn_train, y_churn_test)


def classification_report_image(y_churn_train,
                                y_churn_test,
                                y_churn_train_preds_lr,
                                y_churn_train_preds_rf,
                                y_churn_test_preds_lr,
                                y_churn_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_churn_train: training response values
            y_churn_test:  test response values
            y_churn_train_preds_lr: training predictions from logistic regression
            y_churn_train_preds_rf: training predictions from random forest
            y_churn_test_preds_lr: test predictions from logistic regression
            y_churn_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    logging.info("classification_report_image() with y_churn_train %s, y_churn_test %s, \
                 y_churn_train_preds_lr %s, y_churn_train_preds_rf %s, y_churn_test_preds_lr %s, \
                 y_churn_test_preds_rf %s, called" % \
            (y_churn_train.shape,
            y_churn_test.shape,
            y_churn_train_preds_lr.shape,
            y_churn_train_preds_rf.shape,
            y_churn_test_preds_lr.shape,
            y_churn_test_preds_rf.shape)
        )

    # scores
    img_file_path = IMG_FOLDER + FOLDER_RESULTS + 'random_forest_classification_report.png'
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 1.0, str('Random Forest - Train'),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.8, str(classification_report(y_churn_train, y_churn_train_preds_rf)),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest - Test'),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str(classification_report(y_churn_test, y_churn_test_preds_rf)),
              {'fontsize': 10}, fontproperties='monospace')

    plt.axis('off')
    plt.savefig(img_file_path)
    print("classification_report_image(): classification_report saved to {}" \
        .format(img_file_path))

    img_file_path = IMG_FOLDER + FOLDER_RESULTS + 'logistic_regression_classification_report.png'
    plt.figure(figsize=(10, 10))
    plt.text(0.01, 1.0, str('Logistic Regression Train'),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text( 0.01, 0.8, str(classification_report(y_churn_train, y_churn_train_preds_lr)),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
              {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.4, str(classification_report(y_churn_test, y_churn_test_preds_lr)),
              {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(img_file_path)
    print("classification_report_image(): classification_report saved to {}" \
        .format(img_file_path))


def feature_importance_plot(model, X_churn_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_churn_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    logging.info("feature_importance_plot() with for model Random Forest called")

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importances
    names = [X_churn_data.columns[i] for i in indices]

    # Create feature importance plot
    img_file_path = output_pth + 'random_forest_feature_importance.png'

    plt.figure(figsize=(20, 10))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(X_churn_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_churn_data.shape[1]), names, rotation=90)
    plt.savefig(img_file_path)
    print("feature_importance_plot(): feature importance saved to {}" \
        .format(img_file_path))

    # Create feature importance plot
    img_file_path = output_pth + 'random_forest_feature_shap_values.png'

    plt.figure(figsize=(20, 10))
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_churn_data)
    shap.summary_plot(shap_values, X_churn_data, plot_type="bar", show=False)
    plt.savefig(img_file_path)
    plt.close()
    print("feature_importance_plot(): feature shap values saved to {}" \
        .format(img_file_path))


def train_models(X_churn_train, X_churn_test, y_churn_train, y_churn_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_churn_train: X training data
              X_churn_test: X testing data
              y_churn_train: y training data
              y_churn_test: y testing data
    output:
              None
    '''
    logging.info(
        "train_models() with param X_churn_train %s, X_churn_test %s, y_churn_train %s, \
        y_churn_test %s called" % \
           ( (X_churn_train.shape,),
            (X_churn_test.shape,),
            (y_churn_train.shape,),
            (y_churn_test.shape,))
        )

    # models
    random_state = 42
    rfc = RandomForestClassifier(random_state=random_state)
    lrc = LogisticRegression(
        class_weight='balanced',
        max_iter=10000,
        random_state=random_state
    )

    # grid search hyper parameter space
    rf_param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    lr_param_grid = {
        'C': [100, 10, 1.0, 0.1, 0.01],
        'penalty': ['l2']
    }

    # cross validation
    cv = 2
    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=rf_param_grid,
        cv=cv,
        n_jobs=-1)
    cv_lrc = GridSearchCV(
        estimator=lrc,
        param_grid=lr_param_grid,
        cv=cv,
        n_jobs=-1)

    # model container
    model_dict = {'cv_logistic_regression': cv_lrc, 'cv_random_forest': cv_rfc}
    model_train_preds_dict = {}
    model_test_preds_dict = {}

    # model fitted, best model saved in pkl file, predictions used to generate classification report
    # and  roc curve saved to image charts
    model_path = '../models/'
    img_file_path = IMG_FOLDER + FOLDER_RESULTS + 'model_performance_roc_curves.png'
    plt.figure(figsize=(15, 8))
    ax = plt.gca()

    for model_name, model in model_dict.items():
        print("train_models(): training model {} ...".format(model_name))
        model.fit(X_churn_train, y_churn_train)

        model_pkl_path = model_path + model_name + '.pkl'
        joblib.dump(model.best_estimator_, model_pkl_path)
        print(
            "train_models(): saved model {} to {}".format(
                model_name,
                model_pkl_path))

        logging.info("train_models() - model %s CV results = %s" \
                    % (model_name, model.best_params_))

        model_train_preds_dict[model_name] = model.best_estimator_ \
            .predict(X_churn_train)
        model_test_preds_dict[model_name] = model.best_estimator_ \
            .predict(X_churn_test)

        plot_roc_curve(model.best_estimator_, X_churn_test, y_churn_test, ax=ax, alpha=0.8)

    plt.savefig(img_file_path)
    plt.clf()
    print("train_models(): models roc performance curves saved to {}" \
        .format(img_file_path))

    # generate classification reports and images
    print("train_models(): classification_report_image called")
    classification_report_image(
        y_churn_train,
        y_churn_test,
        model_train_preds_dict['cv_logistic_regression'],
        model_train_preds_dict['cv_random_forest'],
        model_test_preds_dict['cv_logistic_regression'],
        model_test_preds_dict['cv_random_forest'])

    # feature importance plots images
    feature_importance_plot(cv_rfc, X_churn_test, IMG_FOLDER + FOLDER_RESULTS)


if __name__ == "__main__":
    # vars
    DATA_FILE_PATH = "../data/bank_data.csv"

    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    quant_columns = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio'
    ]

    # Script completed
    print("Beginning of Script ! ********")

    # load data
    churn_df = import_data(DATA_FILE_PATH)

    # EDA & save figures
    perform_eda(churn_df)

    # EDA & save figures
    churn_df = encoder_helper(churn_df, cat_columns, None)

    # Keep relevant cols and split churn_df into test and train datasets
    X_train, X_test, y_train, y_test = perform_feature_engineering(churn_df, None)

    # Keep relevant cols and split churn_df into test and train datasets
    train_models(X_train, X_test, y_train, y_test)

    # Script completed
    print("SUCCESS: End of Script ! ********")
