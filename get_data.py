import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix


def get_data(file, cols=None):
    if cols is None:
        return pd.read_csv(file)
    return pd.read_csv(file, usecols=cols)


# Returns train_Set, test_set, train_label, test_label
def split_data(data):
    train, test = train_test_split(data, test_size=0.2)
    return train.loc[:, train.columns != 'SalePrice'], test.loc[:, test.columns != 'SalePrice'], train['SalePrice'], \
           test['SalePrice']


def correlation_matrix(train):
    return train.corr()


def correlation_with_label(train, label):
    return train.corrwith(label)


def plot_correlation_matrix(train):
    df_cm = correlation_matrix(train)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm)
    plt.show()


def plot_correlation_with_result(train, label):
    df_cm = pd.DataFrame({'corr': correlation_with_label(train, label)})
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm)
    plt.show()


def pick_best_feature(train, label, remove_threshold=0.8):
    label_corr = correlation_with_label(train, label)
    inter_corr = correlation_matrix(train)
    selected_feature = label_corr.idxmax()
    print(f'Selected {selected_feature} with {label_corr[selected_feature]} correlation score')
    to_keep = inter_corr.index[inter_corr[selected_feature] <= remove_threshold].tolist()
    return selected_feature, train[to_keep], label


def pipeline(train, label, number_of_features):
    train_copy = train.copy()
    features = []
    while len(features) < number_of_features:
        feat, train, label = pick_best_feature(train, label)
        features.append(feat)
    return train_copy[features]


if __name__ == '__main__':
    train, test, train_label, test_label = split_data(get_data('data/train.csv'))
    train = pipeline(train, train_label, 10)
