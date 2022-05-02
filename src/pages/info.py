import streamlit as st

# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading ..."):
        st.markdown(
            """## CS-GY 9223 Visualisation for Machine Learning""",unsafe_allow_html=True
            )
        st.markdown("### Track 2: Application\n")
        st.write("##### Team Members: \n")
        st.write("Ansh Desai asd9717  \n")
        st.write("Bansi Shah bks7385\n")
        st.write("Vishal Shah vs2530  \n")
        