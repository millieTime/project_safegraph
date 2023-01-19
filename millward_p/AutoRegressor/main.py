# Data manipulator
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Scaling data
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Model finder
from time import time
from xgboost import XGBRegressor
from xgboost import plot_tree
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score

'''
Steps:
    1. Grab the data
    2. Drop columns
    3. Clean the data
    4. Scale/Normilize the data
    5. Find model for the data
    6. Make model for data
'''






def best_models(best_model):
    print(f'Model: {best_model[0]} \nScore: {best_model[1]}')

    return best_model


'''
Clean the data into numberic values
so we can read it into our model.
How:   
    1. Find all columns that doesn't have a number
    2. Find how many unique values there are
    3. One hot encode each unique value 
'''


def clean_data(data, target_column, keep= None):
    '''
    Prupose: One Hot Encoding all unique values to a digit form
    How:
        1. Match the unique value with a unique digit
        2. replace the unique vale in main data with unique digit 
    '''
    def update_columns(data, column_name=None):
        base_dict = dict()
        main_list = []
        count = 1

        main_list = data[column_name]
        for _base in main_list:
            if _base not in base_dict:
                base_dict.update({_base: [_base, count]})
                count += 1

        for _base in base_dict:
            data[column_name] = data[column_name].replace(
                [_base], base_dict[_base][1])

        return data
    Y = data[target_column]
    if keep != None:
        kept = pd.DataFrame()
        for i in keep:
            kept[i] = data[i]
            data = data.drop([i],axis = 1)
    data = data.drop([target_column], axis=1)
    for column in data:
        data = update_columns(data, column)

    data[target_column] = Y
    if keep != None:
        for i in keep:
            data[i] = kept[i]
    return data


'''
Scale/Normalize the data so we can
make data closer to each other so
modeling is easier
'''


def scaling(data, test):

    scaler = StandardScaler()
    check = 0

    # Check if we are looking at test data or not
    for column in data:
        if column == test:
            check += 1

    # runs if we are using test data
    if check > 0:
        Y = data[[test]]
        X = data.drop([test], axis=1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=1)

        X_scaler = scaler.fit(X_train)
        X_train_scaled = X_scaler.transform(X_train)
        X_test_scaled = X_scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    # run if we are using predictable data
    else:
        scale_fit = scaler.fit(data)
        return scale_fit.transform(data)


'''
Purpose: Finding easist model to use based on prediction and time
How: 
    1. Grab all models
    2. Grab Through in for loop
    3. Start time
    4. Predict
    5. Look at explained variance score
    6. Look at mean absolute error score
    7. Look at r2 score
    8. Stop time
    9. Save all values 
'''


def find_model(X_train, X_test, y_train, y_test):

    regressors = [
        KNeighborsRegressor(),
        GradientBoostingRegressor(),
        ExtraTreesRegressor(),
        RandomForestRegressor(),
        DecisionTreeRegressor(),
        LinearRegression(),
        Lasso(),
        Ridge(),
        XGBRegressor()
    ]

    head = 10
    all_score = dict()
    for model in regressors[:head]:
        start = time()
        model.fit(X_train, y_train)
        train_time = time() - start
        start = time()
        y_pred = model.predict(X_test)
        predict_time = time()-start
        # save values
        all_score.update({model: [train_time,
                                  predict_time,
                                  explained_variance_score(y_test, y_pred),
                                  mean_absolute_error(y_test, y_pred),
                                  r2_score(y_test, y_pred)]})

        best_model = [0, 0, 0, 0, 0, 0]
        for columns in all_score:
            if all_score[columns][4] > best_model[1]:
                best_model[0] = columns
                best_model[1] = all_score[columns][4]
                best_model[2] = all_score[columns][3]
                best_model[3] = all_score[columns][2]
                best_model[4] = all_score[columns][0]
                best_model[4] = all_score[columns][1]


    return all_score, best_model


def main():
    PATH = 'salary_data_cleaned.csv'
    data = pd.read_csv(PATH)
    # cleaning data
    c_data = clean_data(data, 'avg_salary')

    # export cleaned data to csv
    c_data.to_csv(
        'clean_salary_data.csv', index=False)

    # scaling data
    X_train, X_test, y_train, y_test = scaling(c_data, 'avg_salary')

    # make model

    all_score, best_model = find_model(X_train, X_test, y_train, y_test)

    for columns in all_score:
        print(f"Method {columns}:")
        print(f"Training Time:\t\t {all_score[columns][0]}")
        print(f"Prediction Time:\t {all_score[columns][1]}")
        print(f"Explained Varience:\t {all_score[columns][2]}")
        print(f'Mean Absolute Error:\t{all_score[columns][3]}')
        print(f"R2 (model accuracy):\t{all_score[columns][4]}")
        print('\n\n\n')

    print(f'Model: {best_model[0]} \nR2 Score: {best_model[1]}\nMean Absolute: {best_model[2]}\nExplained Varience: {best_model[3]}\nTraining Time: {best_model[4]}\nPrediction Time: {best_model[5]}')
    print(best_model)
    return best_model


if __name__ == '__main__':
    main()
