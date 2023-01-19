import pandas as pd
import pytest
from main import best_models, main, find_model, scaling

'''
Purpose: Check if all the unique values from salary data and
        clean data match
'''


def test_columns():
    PATH_1 = 'salary_data_cleaned.csv'
    PATH_2 = 'salary_data_cleaned.csv'
    salary_data = pd.read_csv(PATH_1)
    cleaned_data = pd.read_csv(PATH_2)

    sd_unique = []
    for column in salary_data:
        sd_unique.append(salary_data[column].nunique())

    for column in cleaned_data:
        assert cleaned_data[column].nunique() in sd_unique


'''
Purpose: Check each data type of the columns to 
         see if they are all numeric and not
         string
'''


def test_against_string():
    PATH = 'clean_salary_data.csv'
    salary_data = pd.read_csv(PATH)
    for column in salary_data.columns[29:]:
        assert not isinstance(column, str), f"{column} is a string"


'''
Purpose: Check if chosen model has an
         accuracy over 50% or .5
Reason: Need to check the model to make sure that the model is 
        better than random chance. If the model is less than .5
        then the dataset we have doesn't make sense to make a model for
'''


def test_model():
    PATH = 'clean_salary_data.csv'
    salary_data = pd.read_csv(PATH)
    X_train, X_test, y_train, y_test = scaling(salary_data, 'avg_salary')
    all, best_model = find_model(X_train, X_test, y_train, y_test)
    # print(f'Best Model: {best_model[0]} {best_model[1]}')
    assert best_model[1] > .5


test_against_string()
test_model()
test_columns()

pytest.main(["-v", "--tb=line", "-rN", __file__])
