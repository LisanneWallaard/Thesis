"""
Assignment
    A stroke prediction application for ML models in python using streamlit
Course
    Thesis
Author
    Lisanne Wallaard 
Date
    April 2023
"""

# Necessary imports
# Installation of these libraries on your device are needed 
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os

# Path to the model
PATH_MODEL = "model/lr_stroke.pkl"
# PATH_MODEL = "model/rfc_stroke.pkl" # not fitted?
# PATH_MODEL = "model/svm_stroke.pkl" # set probability=True in the model


def input_user() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user
    """
    options_gender = ("Female", "Male", "Other")
    gender = st.sidebar.selectbox("What is your gender?", options=options_gender)
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    options_yes_no = ("No", "Yes")
    hypertension = st.sidebar.selectbox("Do you have hypertension?", options=options_yes_no)
    heart_disease = st.sidebar.selectbox("Do you have a heart disease?", options=options_yes_no)  
    options_work = ("Private", "Self employed", "Government job", "Children", "Never worked")
    work_type = st.sidebar.selectbox("What kind of work do you do?",
                                    options=options_work)
    glucose_level = st.sidebar.number_input("What is your average glucose level (mg/dL)?",min_value=0.,step=1.,format="%.2f")
    bmi = st.sidebar.number_input("What is your BMI?",min_value=0.,step=1.,format="%.1f")

    # Dataframe containing the input of the user
    input_df = pd.DataFrame({
        "gender": [gender],
        "age": [age],
        "hypertension": [hypertension],
        "heart_disease": [heart_disease],
        "work_type": [work_type],
        "avg_glucose_level": [glucose_level],
        "bmi": [bmi],
    })

    return input_df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns a DataFrame with the preprocessed input of the user
    """
    df['gender'] = df['gender'].replace({'Male':0,'Female':1,'Other':-1}).astype(np.uint8)
    df['work_type'] = df['work_type'].replace({'Private':0,'Self employed':1,'Government job':2,'Children':-1,'Never worked':-2}).astype(np.uint8)
    df['hypertension'] = df['hypertension'].replace({'No':0,'Yes':1}).astype(np.uint8)
    df['heart_disease'] = df['heart_disease'].replace({'No':0,'Yes':1}).astype(np.uint8)
    
    return df

def output_prediction(prediction: int, prediction_prob: float):
    """
    Prints the prediction of the model for the user data
    """
    if prediction == 0:
        st.markdown(f"**:green[The probability that you'll have"
                    f" a stroke is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" You seem to be healthy!]**")
    else:
        st.markdown(f"**:red[The probability that you will have"
                    f" stroke is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" It sounds like you are not healthy!]**")

def main():
    # Add the title and icon of the web page
    st.set_page_config(
        page_title="Stroke Prediction App",
        page_icon="images/heart-fav.png"
    )

    # Add the title and subtitle of the front page 
    st.title("Stroke Prediction")
    st.subheader("This web app can tell your probability to get a stroke based on machine learning models. ")

    # Add a description of the app
    st.markdown("""    
    This application can use several models. You can see the steps of building the model, 
    evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5/notebook). 
    
    In order to get a prediction about your stroke status, you need to take the following steps:
    1. Fill in the asked input features.
    2. Click on the "Predict" button and the prediction will show in a few seconds.
        
    **Keep in mind that this prediction is not the same as a medical diagnosis. 
    It is based on a machine learning model and has an accuracy far from perfect.
    Thus if you experience any health problems, please consult a human doctor.**
    
    **Author: Lisanne Wallaard**
    
    *Based on this [application](https://github.com/kamilpytlak/heart-condition-checker)*
    """)

    # Add a sidebar with a picture for the input to the front page
    st.sidebar.title("Input Features")

    # Get the input data from the user
    df_input = input_user()
    # Convert categorical variables to dummy variables
    df = preprocess_input(df_input)
    
    # Add a button to the side bar to submit the input data
    submission = st.sidebar.button("Predict", type="secondary", use_container_width=True)

    # Add a button to the side bar to stop the application
    stop = st.sidebar.button(label="Stop", type="primary", use_container_width=True)
    
    # Load the machine learning model
    model_ml = pickle.load(open(PATH_MODEL, "rb"))
    
    # Stop the application
    if stop:
        os._exit(0)

    # Make prediction with the input data of the user
    if submission:
        # Get the class prediction
        prediction = model_ml.predict(df)

        # Get the probability of both classes
        prediction_prob = model_ml.predict_proba(df)

        # Print the prediction
        output_prediction(prediction, prediction_prob)

if __name__ == "__main__":
    main()