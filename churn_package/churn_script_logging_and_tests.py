# library doc string
"""

This script allows the userto test the different functions used to
perfom and end to end credit card churn prediction model and assess
the best results obtained.

This contains the following functions:

* test_import() : tests the import_data function
* test_eda() : tests the perform_eda function
* test_encoder_helper() : tests the encoder_helper function
* test_perform_feature_engineering() : tests the perform_feature_engineering function
* test_train_models() : test the test_train_models function

"""

# import modules

import os
import logging
import churn_library as cls


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("../data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    df = cls.import_data("../data/bank_data.csv")
    perform_eda(df)

    expected_files= ['churn_histogram.png', 'customer_age_histogram.png',
                    'marital_status_distribution.png', \
                    'total_trans_ct_histogram.png', 'correlation_heatmap.png']

    list_dir = os.listdir("../images/")
    count = 0
    for file in list_dir:
        if file in expected_files:
            count += 1

    try:
        assert count == 5
        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_eda: Invalid number of charts file created {} expected 5" \
                .format(5))
        raise err


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    new_col = ['Card_Category_Churn', 'Education_Level_Churn', 'Income_Category_Churn', \
                'Marital_Status_Churn', 'Gender_Churn']
    df = cls.import_data("../data/bank_data.csv")

    try:
        df = encoder_helper(df, cat_columns, None)
        assert set(new_col).issubset(df.columns)
        logging.info("test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("test_encoder_helper: all expected new columns were not created")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    df = cls.import_data("../data/bank_data.csv")
    df = cls.encoder_helper(df, cat_columns, None)

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, None)
        assert (X_train.shape[0] > 0) & (X_train.shape[1] > 0)
        assert (X_test.shape[1] > 0) & (X_test.shape[1] > 0)
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing test_perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing test_perform_feature_engineering: invalid expected output")
        raise err


def test_train_models(train_models):
    '''
    test train_models
    '''
    cat_columns = ['Gender','Education_Level','Marital_Status','Income_Category','Card_Category']
    df = cls.import_data("../data/bank_data.csv")
    df = cls.encoder_helper(df, cat_columns, None)
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df, None)

    train_models(X_train, X_test, y_train, y_test)

    expected_files= ['random_forest_feature_shap_values.png', \
                    'random_forest_classification_report.png', \
                    'random_forest_feature_importance.png', \
                    'logistic_regression_classification_report.png', \
                    'model_performance_roc_curves.png' ]

    list_dir = os.listdir("../images/")
    count_img = 0
    for file in list_dir:
        if file in expected_files:
            count_img += 1

    list_dir = os.listdir("../models/")
    count_models = 0
    for file in list_dir:
        if file.endswith("pkl"):
            count_models += 1

    try:
        assert count_img == 5
        assert count_models == 2
        logging.info("Testing test_train_models: SUCCESS")

    except AssertionError as err:
        logging.error("Invalid number of charts file created expected 5 (vs. {}) and 2 (vs {})" \
                .format(count_img, count_models))
        raise err



if __name__ == "__main__":

    # test import_data()
    test_import(cls.import_data)

    # test perform_eda()
    test_eda(cls.perform_eda)

    # test encoder_helper()
    test_encoder_helper(cls.encoder_helper)

    # test perform_feature_engineering()
    test_perform_feature_engineering(cls.perform_feature_engineering)

    # test test_train_models()
    test_train_models(cls.train_models)