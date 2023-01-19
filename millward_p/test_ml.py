#%% Initialization
import pandas as pd
import altair as alt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
#Trying multiple models
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# alt.data_transformers.enable('json')
parquet_folder_path = "C:\\Users\\Preso\\Code\\VSCode\\Winter2023\\BigData\\project_safegraph\\parquet"
poi_df = pd.read_parquet(parquet_folder_path + "\\poi.parquet")
popularity_by_day_df = pd.read_parquet(parquet_folder_path + "\\popularity_by_day.parquet")
related_same_day_brand_df = pd.read_parquet(parquet_folder_path + "\\related_same_day_brand.parquet")

#%%

#Valuable predictors:
# find certain brands that are highly related (same-day, same-month), and see how many of those are in a given area.
# 

#%% Visitors per season
# We only have one month, so aggregate over the whole dataset.
poi_popularity_df = pd.merge(popularity_by_day_df, poi_df, on="placekey")
total_visits_per_building = (poi_popularity_df
    .groupby(
        ['placekey']
    ).agg(
        visits = ('value', np.sum),
    ).reset_index()
)
# Add those calculated numbers to the data that we already have.
#  This number is the thing that we'll be predicting.
# %%
# Create training and testing data
#target:before 1980. giveaway: Year Built
X_pred = dwellings_ml.drop(columns=['Year Built', 'Built Before 1980'])
y_pred = dwellings_ml.filter(['Built Before 1980'])

X_train, X_test, y_train, y_test = train_test_split(
    X_pred,
    y_pred,
    test_size=.34,
    random_state= 76)

