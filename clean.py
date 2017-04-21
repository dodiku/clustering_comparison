import csv
import pandas as pd


'''------------------------------------
LOADING THE DATASET THAT INCLUDES THE FOLLOWING COLUMNS:
    [0] = Neighborhood
    [1] = BldClassif [classes]
    [2] = YearBuilt
    [3] = GrossSqFt
    [4] = GrossIncomeSqFt
    [5] = MarketValueperSqFt [data]
------------------------------------'''

df = pd.read_csv('manhattan-dof.csv',index_col=False,delimiter=';')

'''--------------------
cleaning the data
--------------------'''
def clean():
    df_clean = df
    print ('\ndf_clean before cleaning: ', df_clean.shape, '\n')
    df_clean = df_clean[df.MarketValueperSqFt < 200]
    df_clean = df_clean[df.BldClassif > 0]
    df_clean = df_clean[df.BldClassif < 3]
    df_clean = df_clean[df.GrossSqFt < 200000]
    print ('\ndf_clean before cleaning: ', df_clean.shape)

    X1 = df_clean['GrossSqFt']
    X2 = df_clean['MarketValueperSqFt']
    columns = [X1, X2]
    X = pd.concat(columns, axis=1)
    Y = df_clean['BldClassif']

    return X, Y
