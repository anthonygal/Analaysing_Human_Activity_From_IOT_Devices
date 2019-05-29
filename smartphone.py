import ast
import itertools
import math

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# from pandas.io.json import json_normalize


parse_dates = ['timestamp']

smartphone_df = pd.read_csv(
    "./donnees/smartphone.csv", parse_dates=parse_dates, nrows=200000)

smartphone_df = smartphone_df.rename(columns={'values': 'val'})

# smartwatch_df = pd.read_csv(
#     "./donnees/smartwatch.csv", parse_dates=parse_dates)

parse_dates = ['to', 'from']
report_df = pd.read_csv("./donnees/report.csv", parse_dates=parse_dates)
report_df = report_df['activity_type', 'duration', 'from', 'to']


print(report_df.head())


def valueL(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


for column in smartphone_df:
    smartphone_df[column] = smartphone_df[column].apply(valueL)


smartphone_df = smartphone_df.pivot(
    columns='source',
    values='val',
    index='timestamp')

# -------- WORK ON ACTIVITY --------

df_activity = smartphone_df.activity.dropna()
print(df_activity['2017-06-30 15:24:08.579'][0].split(", "))


def toDict(item):
    if not isinstance(item, list):
        return math.nan
    L = []
    for e in item[0].split(', '):
        L += e.split(': ')
    return dict(itertools.zip_longest(*[iter(L)] * 2, fillvalue=""))


df_activity = df_activity.apply(toDict)
activity_split = df_activity.apply(pd.Series)

for column in activity_split:
    activity_split[column] = pd.to_numeric(
        activity_split[column], errors="coerce")

activity_split = activity_split.resample('min').mean()
# activity_split.replace('NaN', math.nan, inplace=True)

activity_split = activity_split.dropna(how='all')
activity_split = activity_split.fillna(0.0)

# kmeans = KMeans(n_clusters=4).fit(activity_split)
# centroids = kmeans.cluster_centers_

# print(activity_split.describe())  # 40% immobile

activity_split.plot(kind="box")
plt.show()

# bplot = sns.boxplot(
#     y="STILL", data=activity_split, width=0.5, palette='colorblind')


# -------- OTHERS --------

smart_df = smartphone_df[
    ['battery', 'pressure', 'step_counter', 'step_detector']]


def elementL(item):
    if not isinstance(item, list):
        return math.nan
    return item[0]


for column in smart_df:
    smart_df[column] = smart_df[column].apply(elementL)
    smart_df[column] = pd.to_numeric(
        smart_df[column], errors="coerce")

# print(smart_df)

smart_df = smart_df.resample('min').mean()
smart_df['step_detector'].fillna(0.0, inplace=True)
smart_df = smart_df.dropna()

# print(smart_df.describe())

# # ------- LINKED DATAFRAMES -------

smart_link = pd.concat([activity_split, smart_df], axis=1, join='inner')

# print(smart_link.describe())  # ON_FOOT 55%


labels = {1: 'STILL', 2: 'ON_FOOT', 3: 'WALKING', 4: 'IN_VEHICLE',
          5: 'ON_BICYCLE', 6: 'UNKNOWN', 7: 'RUNNING', 8: 'TILTING'}
