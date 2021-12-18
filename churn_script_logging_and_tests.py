import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import logging
import churn_library
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(pth):
    '''
    test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        df = churn_library.import_data(pth)
        print(df.head())
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
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
    except AssertionError as msg:
        logging.error("Testing perform_eda: Empty or Non-existent dataframe")
#         raise err
    except BaseException as msg:
        print("error message in test_eda: ",msg)
        logging.error("Testing perform_eda: Could not run perform_eda")
#         raise err


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
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The given Dataframe doesn't appear to have rows and columns")
    except BaseException as err:
        logging.error(
            "Testing encoder_helper: BaseException: ", err)


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
        X_train, X_test, y_train, y_test = churn_library.perform_feature_engineering(df, response)
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except TypeError as err:
        logging.error('Testing perform_feature_engineering: TypeError: ', err)
        print("error in testing perform feature engineering: ",err)
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The output variables dont appear to have rows or columns")


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
    except AssertionError as err:
        logging.error(
            "Testing train_models: The given parameters dont appear to have rows or columns")
    except BaseException as err:
        logging.error("Testing train_models: Basic Exception")
        print("error msg in test train_models: ", err)
#         raise err


if __name__ == "__main__":
    
    df = pd.read_csv("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    pth = "./data/bank_data.csv"
    
    test_import(pth)
    test_import('not_real_path')
    test_eda(df)
    test_eda(pd.DataFrame())
    
    category_lst = ['Gender','Education_Level','Marital_Status',
            'Income_Category','Card_Category']
    response = ['Gender_Churn','Education_Level_Churn',
            'Marital_Status_Churn','Income_Category_Churn', 'Card_Category_Churn']
    test_encoder_helper(df, category_lst, response)
    test_encoder_helper(df, 'um', 'yes')
    encoded_df = churn_library.encoder_helper(df, category_lst, response)
    
    y = df['Churn']
    X = pd.DataFrame()
    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
             'Income_Category_Churn', 'Card_Category_Churn']
    
    X[keep_cols] = encoded_df[keep_cols]
    test_perform_feature_engineering(encoded_df, 'Churn')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42) 
    test_train_models(X_train, X_test, y_train, y_test)
    
