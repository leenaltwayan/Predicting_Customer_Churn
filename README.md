# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
In this project i refactored the prototype code from the notebook to create production level scripts in a way that increases modularization and readability while following pep8 guidelines.

## Files In The Repo: 
**data**
   - bank_data.csv
**images**
   * eda
      - churn_distribution.png
      - customer_age_distribution.png
      - heatmap.png
      - marital_status_distribution.png
      - total_transaction_distribution.png
   * results
      - feature_importance.png
      - logistics_results.png
      - rf_results.png
      - roc_curve_result.png
**logs**
   - churn_library.log
**models**
   - logistic_model.pkl
   - rfc_model.pkl
   - churn_library.py
   - churn_notebook.ipynb
   - churn_script_logging_and_tests.py


## Running Files
Here are the steps to running the files:

1- install all needed libraries and packages. 
You can install these packages using the following command:
    **pip install scikit-learn shap pylint autopep8 scikit-plot**

2- Run the file *churn_library.py* using the following command:
    **ipython churn_library.py**

3- Run the testing file *churn_script_logging_and_testing.py* using the following command:
    **ipython churn_script_logging_and_tests.py**
    
4- View the logs by going to the folder '/logs/churn_library.log' 
