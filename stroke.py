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

# Path to the model
# PATH_MODEL = "model/lr_stroke.pkl"
# PATH_MODEL = "model/rfc_stroke.pkl"
# PATH_MODEL = "model/svm_stroke.pkl" # set probability=True in the model


def input_user() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user
    """
    options_gender = ("Female", "Male", "Other")
    gender = st.sidebar.selectbox("Gender", options=options_gender)
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    options_yes_no = ("No", "Yes")
    hypertension = st.sidebar.selectbox("Do you have hypertension?", options=options_yes_no)
    heart_disease = st.sidebar.selectbox("Do you have a heart disease?", options=options_yes_no)  
    options_work = ("Private", "Self employed", "Government job", "Children", "Never worked")
    work_type = st.sidebar.selectbox("What kind of work do you do?",
                                    options=options_work)
    glucose_level = st.sidebar.number_input("What is your average glucose level (mg/dL)?",step=1.,format="%.2f")
    bmi = st.sidebar.number_input("What is your BMI?",step=1.,format="%.1f")

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
        st.markdown(f"**The probability that you'll have"
                    f" a stroke is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" You are healthy!**")
        st.image("images/heart-okay.jpg",
                    caption="You seem to be okay! - Dr. Logistic Regression")
    else:
        st.markdown(f"**The probability that you will have"
                    f" stroke is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" It sounds like you are not healthy.**")
        st.image("images/heart-bad.jpg",
                    caption="I'm not satisfied with the condition of health! - Dr. Logistic Regression")

def main():
    # Add the title and icon of the web page
    st.set_page_config(
        page_title="Stroke Prediction App",
        page_icon="images/heart-fav.png"
    )

    # Add the title and subtitle of the front page 
    st.title("Stroke Prediction")
    st.subheader("Are you wondering about whether you are likely to get a stroke? "
                    "This app will help you!")

    col1, col2 = st.columns([1, 3])

    # Add an image on the front page
    with col1:
        st.image("images/doctor.png",
                    caption="I'll help you with stroke prediction! - Dr. Logistic Regression",
                    width=150)
        submission = st.button("Predict")

    # Add text on the front page
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict your likeliness to get a stroke? In this app, you can
        estimate your chance of a stroke (yes/no) in seconds!
        
        Here, the application is based on several models. You can see the steps of building the model, 
        evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5/notebook). 
        
        To predict your stroke status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        
        **Author: Lisanne Wallaard**
        
        *Based on this [application](https://github.com/kamilpytlak/heart-condition-checker)*
        """)

    # Add a sidebar with a picture for the input to the front page
    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    # Get the input data from the user
    df_input = input_user()
    # Convert categorical variables to dummy variables
    df = preprocess_input(df_input)
    # Load the machine learning model
    model_ml = pickle.load(open(PATH_MODEL, "rb"))

    # Make prediction with the input data of the user
    if submission:
        # Get the class prediction
        prediction = model_ml.predict(df)
        st.write(prediction)
        # Get the probability of both classes
        prediction_prob = model_ml.predict_proba(df)
        st.write(prediction_prob)
        # Print the prediction
        output_prediction(prediction, prediction_prob)

if __name__ == "__main__":
    main()