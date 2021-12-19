# library doc string
'''
Author: Leen
Version: 0.1
Date: 12/12/2021
This module provides a set of functions that will predict customer churn,
guaranteeing readability and modularization
'''

# import libraries
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import scikitplot as skplt
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    data = pd.DataFrame()
    try:
        assert isinstance(pth, str)
        data = pd.read_csv(pth)
    except AssertionError:
        print("in import_data: The given path is not a string")

    except FileNotFoundError:
        print("in import_data: The given path could not be found")

    return data


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    try:
        assert isinstance(df, pd.DataFrame)

        plt.figure(figsize=(20, 10))
        df['Churn'].hist()
        plt.savefig('images/eda/churn.png')

        plt.figure(figsize=(20, 10))
        df['Customer_Age'].hist()
        plt.savefig('images/eda/customer_age.png')

        plt.figure(figsize=(20, 10))
        df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig('images/eda/marital_status.png')

        plt.figure(figsize=(20, 10))
        sns.distplot(df['Total_Trans_Ct'])
        plt.savefig('images/eda/total_trans_ct.png')

        plt.figure(figsize=(20, 10))
        sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.savefig('images/eda/heat_map.png')

    except AssertionError as msg:
        print("in perform_eda: The given path is not a string")
    except BaseException as msg:
        print("Exception in perform_eda: ", msg)

def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain
            categorical features
            response: string of response name [optional argument
            that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    try:
        assert isinstance(category_lst, list)
        assert isinstance(response, list)
        assert isinstance(df, pd.DataFrame)
        i = 0
        for cat in category_lst:
            # encoded column
            cat_lst = []
            cat_groups = df.groupby(cat).mean()['Churn']

            for val in df[cat]:
                cat_lst.append(cat_groups.loc[val])

            df[response[i]] = cat_lst
            i = i + 1
        return df

    except AssertionError as msg:
        print('in encoder_helper: the parameters are incorrect types')

    except BaseException as msg:
        print('exception in encoder helper: ', msg)


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    try:
        assert isinstance(df, pd.DataFrame)
        assert isinstance(response, str)
        y = df[response]
        X = df.drop(columns=response)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        return X_train, X_test, y_train, y_test

    except AssertionError as msg:
        print('in perform_feature_engineering: parameter types are incorrect')
        print(msg)
    except BaseException as msg:
        print('Exception in perform_feature_engineering')
        print(msg)


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    try:
        assert isinstance(y_train, list)
        assert isinstance(y_test, list)
        assert isinstance(y_train_preds_lr, list)
        assert isinstance(y_train_preds_rf, list)
        assert isinstance(y_test_preds_lr, list)
        assert isinstance(y_test_preds_rf, list)

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Random Forest Train'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {
                 'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('images/results/random_forest_report.png')

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
                 'fontsize': 10}, fontproperties='monospace')

        plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
                 'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('images/results/logistic_regression_report.png')

    except AssertionError as msg:
        print('one of the parameters is not a list')
        print(msg)


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    try:
        assert isinstance(X_data, pd.DataFrame)
        assert isinstance(output_pth, str)
        # unsure how to check for model instance without specifiying if its logistic regression
        # or Random Forest

        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)

        plt.savefig(output_pth + '/feature_importance.png')

    except AssertionError as msg:
        print('the parameters are incorrect types')
        print(msg)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''

    try:
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)

        # grid search
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=1000)
        param_grid = {
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [4, 5, 100],
            'criterion': ['gini', 'entropy']
        }
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(X_train, y_train)
        lrc.fit(X_train, y_train)
        y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
        y_train_preds_lr = lrc.predict(X_train)
        y_test_preds_lr = lrc.predict(X_test)

        lrc_plot = plot_roc_curve(lrc, X_test, y_test)
        # # Not sure if this will work
        # lrc_plot.savefig('images/results/lr_roc_curve.png')
        print('X test and y test in train model: ')
        print(X_test)
        print(y_test)
        plot_roc_curve(lrc, X_test, y_test)
        plt.savefig('images/results/lr_roc_auc.png')
        feature_importance_plot(cv_rfc, X_train,'images/results/')

        # plots
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(
            cv_rfc.best_estimator_,
            X_test,
            y_test,
            ax=ax,
            alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig('images/results/comparative_roc_auc.png')

        # save best model
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

    except AssertionError as msg:
        print('in train_models: Incorrect parameters; make sure X values are dataframes and Y values are a list')
        print(msg)
    except BaseException as err:
        print('Exception in Training function: ', err)
        raise err


if __name__ == "__main__":
    
    df = pd.read_csv("./data/bank_data.csv")
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    pth = "./data/bank_data.csv"
    
    print(import_data(pth))
    perform_eda(df)
    
    category_lst = ['Gender','Education_Level','Marital_Status',
            'Income_Category','Card_Category']
    response = ['Gender_Churn','Education_Level_Churn',
            'Marital_Status_Churn','Income_Category_Churn', 'Card_Category_Churn']
    encoded_df = encoder_helper(df, category_lst, response)

    print(encoded_df)
    
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
    print('y shape is: ', y.shape)
    print(y.dtype)
    print('X shape is: ', X.shape)
    perform_feature_engineering(encoded_df, 'Churn')
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42) 
    print(y_train.dtype)
    print(y_test.dtype)
    train_models(X_train, X_test, y_train, y_test)
    