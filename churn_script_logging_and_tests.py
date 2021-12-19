'''
Author: Leen
Version: 0.1
Date: 12/12/2021
This module tests a set of functions used in churn_library, including:
- test_import
- test_eda
- test_encoder_helper
- test_perform_feature_engineering
- test_train_models
'''
import os
import logging
import warnings
from sklearn.model_selection import train_test_split
import pandas as pd
import churn_library
warnings.filterwarnings("ignore")
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(file_path):
    '''
    test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        df = churn_library.import_data(file_path)
        print(df.head())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError:
        logging.error("Testing import_eda: The file wasn't found")

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")


def test_eda(df):
    '''
    test perform eda function
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        churn_library.perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError:
        logging.error("Testing perform_eda: Empty or Non-existent dataframe")
    except BaseException:
        logging.error("Testing perform_eda: Could not run perform_eda")


def test_encoder_helper(df, category_lst, response):
    '''
    test encoder helper
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(category_lst, list)
        assert isinstance(response, list)
        encoded_df = churn_library.encoder_helper(df, category_lst, response)
        assert encoded_df.shape[0] > 0
        assert encoded_df.shape[1] > 0
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing encoder_helper: The given Dataframe doesn't appear to have rows and columns")
    except BaseException:
        logging.error(
            "Testing encoder_helper: Basic exception")


def test_perform_feature_engineering(df, response):
    '''
    test perform_feature_engineering
    '''

    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(response, str)
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: the given parameters are incorrect types")
        print("in test_perform_feature_engineering: incorrect parameter types")
    try:
        X_train, X_test, y_train, y_test = churn_library.perform_feature_engineering(
            df, response)
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except TypeError as err:
        logging.error(
            'Testing perform_feature_engineering: Type error occurred. Please check parameters.')
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: The output variables \
            dont appear to have rows or columns")


def test_train_models(X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        churn_library.train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing train_models: SUCCESS")
    except AssertionError:
        logging.error(
            "Testing train_models: The given parameters dont appear to have rows or columns")
    except BaseException:
        logging.error("Testing train_models: Basic Exception")


if __name__ == "__main__":

    df = pd.read_csv("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    pth = "./data/bank_data.csv"

    test_import(pth)
    test_import('not_real_path')
    test_eda(df)
    test_eda(pd.DataFrame())

    cat_list = ['Gender', 'Education_Level', 'Marital_Status',
                'Income_Category', 'Card_Category']
    resp_list = [
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    test_encoder_helper(df, cat_list, resp_list)
    test_encoder_helper(df, 'um', 'yes')
    encoded_df = churn_library.encoder_helper(df, cat_list, resp_list)

    y = df['Churn']
    X = pd.DataFrame()
    cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
            'Income_Category_Churn', 'Card_Category_Churn']

    X[cols] = encoded_df[cols]
    test_perform_feature_engineering(encoded_df, 'Churn')

    X_training_set, X_testing_set, y_training_set, y_testing_set = train_test_split(
        X, y, test_size=0.3, random_state=42)
    test_train_models(X_training_set, X_testing_set,
                      y_training_set, y_testing_set)
