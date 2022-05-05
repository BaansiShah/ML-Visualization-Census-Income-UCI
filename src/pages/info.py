import streamlit as st

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading ..."):
        st.title(
            """CS-GY 9223 Visualisation for Machine Learning"""
            )
        
        st.markdown("### Track 2: Application (Lime & Shap)\n")
        st.write("### It is increasingly unclear how LIME and SHAP perform with different data, and how the structure of the data affects their results.")
        st.write("##### Team Members: \n")
        st.write("Ansh Desai: asd9717  \n")
        st.write("Bansi Shah: bks7385\n")
        st.write("Vishal Shah: vs2530  \n")