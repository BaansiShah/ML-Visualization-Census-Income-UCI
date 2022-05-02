import streamlit as st
# pages view
import src.pages.info as info
import src.pages.explanation as explanation


PAGES = {
    "Title": info,
    "Explanations": explanation,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        page.write()
  
if __name__ == "__main__":
    main()
