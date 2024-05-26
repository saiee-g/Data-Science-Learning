# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 09:38:36 2024

@author: snowfox
"""

"""imorting the pandas library"""
import pandas as pd

"""storing genshin.csv in df1, name of dataset"""
df1 = pd.read_csv('genshin.csv', encoding='latin-1')

"""print first 20 rows of dataset"""
print(df1.head(20))

"""Return the first 5 rows"""
print(df1.head(n=5))

"""Return the last 5 rows"""
print(df1.tail(n=5))

"""Return a tuple representing the dimensionality of the
Dataset."""
print(df1.shape)

"""Return the dtypes in the Dataset."""
print(df1.dtypes)

"""The index (row labels) of the dataset"""
print(df1.index)

"""The column labels of the Dataset."""
print(df1.columns)

"""Return the columns values in the Dataset in array
format"""
print(df1.columns.values)

"""to view some basic statistical details like percentile, mean, std etc. of a data frame or a series of numeric values"""
print(df1.describe(include='all'))

"""Read the Data Column wise"""
print(df1['character_name'])

"""Sort object by labels (along an axis)"""
print(df1.sort_index(axis=1,ascending=False))

"""Sort values by column name"""
print(df1.sort_values(by='atk_1_20'))

"""Purely integer-location based indexing for selection by position"""
print(df1.iloc[5])

"""Selecting via [], which slices the rows"""
print(df1[0:3])

"""Selection by label"""
print(df1.loc[:,['character_name', 'weapon_type']])

"""a subset of the first 3 rows of the original data"""
print(df1.iloc[:3,:])

"""a subset of the first 3 columns of the original data"""
print(df1.iloc[:,:3])

"""a subset of the first 3 rows and the first 4 columns"""
print(df1.iloc[:5,:5])

print(df1.isnull())

print(df1.isna())

print(df1.isnull().any())

print(df1.isnull().sum().sum())

print(df1.isnull().sum(axis=1))

print(df1.dtypes)

