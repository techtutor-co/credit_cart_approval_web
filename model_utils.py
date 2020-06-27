import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')


def label_encode(df, column):
    le = preprocessing.LabelEncoder()
    values = list(df[column].values)
    le.fit(values)
    df[column] = le.transform(values)
    return df


def one_hot_encode(df, column):
    df = df.join(pd.get_dummies(df[column], prefix=column))
    return df.drop([column], axis=1)


def scale_normalize(df, columns):
    df[columns] = MinMaxScaler().fit_transform(df[columns])
    for column in columns:
        df[column] = df[column].apply(lambda x: np.log(x + 1))
    return df


def encode_dataset(df):
    df = df.drop(['ID'], axis=1)
    df['Age'] = df['Age'].apply(lambda value: int(value / 10) * 10)

    df = one_hot_encode(df, 'Securities Account')
    df = one_hot_encode(df, 'CD Account')
    df = one_hot_encode(df, 'Online')
    df = one_hot_encode(df, 'CreditCard')

    df = label_encode(df, 'ZIP Code')
    features = df.drop(['Personal Loan'], axis=1)
    labels = df['Personal Loan']

    features = scale_normalize(features, ['Age', 'Experience', 'Income', 'ZIP Code', 'Family', 'CCAvg', 'Education',
                                          'Mortgage'])
    df = features
    df['Personal Loan'] = labels
    return df
