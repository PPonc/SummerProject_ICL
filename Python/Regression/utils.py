import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

def relu(X):
    return np.maximum(0, X)

def prune_non_significant_features(features, df, t):
    N_t = t * df.shape[0]
    new_features = []
    for f in features:
        N = (df[f] > 0).sum()
        if N >= N_t:
            new_features.append(f)
    return new_features

def order_and_select_columns(df, features):
    return df.reindex(features, axis=1, fill_value=0)

def split_dataset_Xy(df, y_name, features):
    y = df[y_name]
    X = order_and_select_columns(df, features)
    return X,y

def prepare_dataset(df):
    df = df.select_dtypes(include=[int, float])
    df = df.div(df['T'], axis=0)
    return df

def load_dataset(filename):
    df = pd.read_csv(filename, sep=";")
    return prepare_dataset(df)

def plot_relative_error(y_true, y_pred, plot=True):
    pct = 100.0 * (y_pred - y_true) / y_true
    print("Relative error: {} ({})".format(np.mean(pct), np.std(pct)))
    print("\t95th percentile [{:.2f}%; {:.2f}%]".format(np.mean(pct) - 1.96 * np.std(pct), np.mean(pct) + 1.96 * np.std(pct)))
    if plot:
        sb.histplot(pct,kde=True)
