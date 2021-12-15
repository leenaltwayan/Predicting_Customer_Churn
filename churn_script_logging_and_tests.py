import os
import logging
import churn_library

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
        '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
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
    try:
        df = pd.read_csv("./data/bank_data.csv")
        perform_eda(df)
        logging.info("Testing perform_eda: SUCCESS")
    except:
        logging.error("Testing perform_eda: Could not run perform_eda")
        raise err

def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        category_lst = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 
                       'Card_Category']
        response = ['Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
                   'Income_Category_Churn', 'Card_Category_Churn']
        
        encoded_df = encoder_helper(df, category_lst, response)
        logging.info("Testing encoder_helper: SUCCESS")
        assert encoded_df.shape[0] > 0
        assert encoded_df.shape[1] > 0
    except AssertionError as err:
        logging.error("Testing encoder_helper: The new Dataframe doesn't appear to have rows and columns")
        raise err



def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''
    
    try:
        df = pd.read_csv("./data/bank_data.csv")
        response = 'Churn'
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
        assert X_train.shape[0] > 0
        assert X_train.shape[1] > 0
        assert X_test.shape[0] > 0
        assert X_test.shape[1] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: The output variables dont appear to have rows or columns")
        raise err
        
    
    
def test_train_models(train_models):
    '''
    test train_models
    '''
    try:
        df = pd.read_csv("./data/bank_data.csv")
        response = 'Churn'
        X_train, X_test, y_train, y_test = perform_feature_engineering(df, response)
        train_models(X_train, X_test, y_train, y_test)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except BaseException as err:
        logging.error("Testing perform_feature_engineering: Basic Exception")
        raise err


if __name__ == "__main__":
    pass
    #test_import(churn_library.import_data())
    #test_import(churn_library.import_data('pth'))
    #test_import(churn_library.import_data("./data/bank_data.csv"))
    
    #thesee all result in errors
    #unsure how to test this since the parameters are the functions themselves and not the parameters for the function (by following example of first testing funtion)







