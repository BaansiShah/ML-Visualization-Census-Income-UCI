import pandas as pd
import streamlit as st
from sklearn.datasets import make_blobs
from numpy import cov
from scipy.stats import pearsonr

def dataframe(source):
    if source=="US Census Data":
        data = pd.read_csv("data/census.csv", na_values=' ?')
        return data
    else:
        n_samples=5000
        n_features=3
        random_state=1
        return_centers=True
        
        x, y, centers = make_blobs(n_samples=n_samples, 
                                   n_features=n_features, 
                                   random_state=random_state, 
                                   return_centers=return_centers)
        
        covariance = cov(x[:,0], x[:, 1])
        st.write("Correlation of Features vs Target : ", pearsonr(x[:,0], x[:, 1]))
        
        final_df = pd.DataFrame(x,columns=['x1','y1','z1'])
        final_df['target'] = y
        return final_df