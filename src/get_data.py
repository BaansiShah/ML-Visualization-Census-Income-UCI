import pandas as pd
import streamlit as st

def dataframe(source):
    if source=="US Census Data":
        data = pd.read_csv("data/census.csv", na_values=' ?')
        return data
    else:
        return 


def synthetic():
    pass