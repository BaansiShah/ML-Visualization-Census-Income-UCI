# LIME & SHAP Explanations 
Explainability methods, like LIME and SHAP, are rapidly gaining traction. At the same time, like many methods, they contain a variety of hyperparameters that can lead to varied results. 
Furthermore, it is increasingly less clear how LIME and SHAP perform on different data, and how the structure of the data impacts their outputs. 
In this project, we create an interactive visualization to explore sets of LIME and SHAP explanations. Using your interactive visualization, we then explore LIME and SHAP using carefully constructed synthetic data sets. We construct your synthetic data sets in a way for you to test different hypotheses about the data (e.g., many correlated features, categorical features, etc.) and report where LIME and SHAP are successful, and unsuccessful. We not only explore the effects that the parameters of LIME and SHAP have on explanations, but also the effect of the underlying data. 

#### Course: CS-GY 9223 Visualisation for machine learning 
#### Term: Spring 2022
#### Instructor: Prof. Claudio Silva


## To run the app:
#Task 1:
pip install -r requirements.txt

#Task2:
streamlit run app.py
