import pandas as pd
import streamlit as st
from sklearn.datasets import make_blobs
from numpy import cov
from scipy.stats import pearsonr
from sklearn.datasets import make_classification

def dataframe(source):
    if source=="US Census Data":
        data = pd.read_csv("data/census.csv", na_values=' ?')
        return data
    elif source=='Synthetic Data with Noise with Balanced Class':
        X1,y1 = make_classification(n_samples=10000, n_features=3, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=0.5, random_state=2022)
        final_df = pd.DataFrame(X1,columns=['Feature1','Feature2','Feature3'])
        final_df['target'] = y1
        return final_df
    elif source=='Synthetic Data with Noise with Imbalanced Class':
        X1,y1 = make_classification(n_samples=10000, n_features=3, n_informative=3, n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=0.5, weights=[0.7,0.3], random_state=2022)
        final_df = pd.DataFrame(X1,columns=['Feature1','Feature2','Feature3'])
        final_df['target'] = y1
        return final_df
    elif source=='Synthetic Data without Noise with Imbalanced Class':
        X1,y1 = make_classification(n_samples=10000, n_features=3, n_informative=3,n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=0,weights=[0.7,0.3], random_state=2022)
        final_df = pd.DataFrame(X1,columns=['Feature1','Feature2','Feature3'])
        final_df['target'] = y1
        return final_df
    elif source=="Synthetic Data without Noise with Balanced Class":
        X1,y1 = make_classification(n_samples=10000, n_features=3, n_informative=3,n_redundant=0, n_repeated=0, n_classes=2, n_clusters_per_class=1,class_sep=2,flip_y=0, random_state=2022)
        final_df = pd.DataFrame(X1,columns=['Feature1','Feature2','Feature3'])
        final_df['target'] = y1
        return final_df