"""
Assignment
    A heart failure prediction application for cox models in R using streamlit
Course
    Thesis
Author
    Lisanne Wallaard
Date
    May 2023
"""

# Inspirated by
# https://goddoe.github.io/r/machine%20learning/2017/12/17/how-to-use-r-model-in-python.html

# Necessary imports
# Installation of these libraries on your device are needed
# make sure libraries are up to date and do no conflict with each other
import os

# Path to R on your device
# Enter R.home() in R studio for example
PATH_R = "C:/Program Files/R/R-4.3.0"
# Set your R path
os.environ["R_HOME"] = PATH_R

import numpy as np
import pandas as pd
import streamlit as st
import rpy2.robjects as ro
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()

# Path to the model
PATH_MODEL = "model/cox_num.rds"

# Set R
r = ro.r


def input_user():
    """Gives input possibilities for the user and saves their response

    Returns:
        pd.DataFrame: input_df contains the input of the user
        pd.DataFrame: options_df contains the input possibilities for the user
    """
    time = st.sidebar.number_input(
        "Over how many days do you want to know your survival prediction?", min_value=0, step=1
    )
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    ejection_fraction = st.sidebar.number_input("What is your ejection fraction?", min_value=0, step=1)
    serum_creatinine = st.sidebar.number_input(
        "What is your serum creatine?", min_value=0, step=0
    )  # format="%.1f" gaf een error?
    serum_sodium = st.sidebar.number_input("What is your serium sodium?", min_value=0, step=1)

    # Dataframe containing the input of the user
    input_df = pd.DataFrame(
        {
            "time": [time],
            "DEATH_EVENT": [0],
            "age": [age],
            "ejection_fraction": [ejection_fraction],
            "serum_creatinine": [serum_creatinine],
            "serum_sodium": [serum_sodium],
        }
    )

    return input_df


def preprocess_input(df: pd.DataFrame):
    """Preprocesses the input of the user

    Args:
        df (pd.DataFrame): contains not preprocessed input of the user

    Returns:
        pd.DataFrame: df contains preprocessed input of the user
    """
    # One-hot encoder for some categorical variables
    ohe = OneHotEncoder(handle_unknown="ignore")

    # Label encoder for some categorical variables
    le = LabelEncoder()

    # Define some categorical columns for labelling
    label_columns = ["Sex", "FastingBS", "RestingECG.LVH", "ST_Slope"]

    # Label encode some categorical variables
    for col in label_columns:
        df_nan = df[col].dropna()
        df[col] = pd.DataFrame(le.fit_transform(df_nan))

    # Define some categorical columns for one-hot encoding
    hot_columns = ["ChestPainType", "ExerciseAngina"]
    # Define the corresponding column names for one-hot encoding (same index as hot_columns)
    column_names = [
        ["ChestPainType.ASY", "ChestPainType.ATA", "ChestPainType.NAP", "ChestPainType.TA"],
        ["ExerciseAngina.N", "ExerciseAngina.Y"],
    ]

    # One-hot encode some categorical variables
    for col in hot_columns:
        df_nan = df[[col]].dropna()
        encoder_df = pd.DataFrame(
            ohe.fit_transform(df_nan).toarray(), columns=column_names[hot_columns.index(col)]
        )
        df = df.join(encoder_df)
        del df[col]

    # Select only the first row (the user input data)
    df = df[:1]

    return df


# def output_prediction(prediction: int, prediction_prob: float):
#     """Prints the prediction itself and the probability of the prediction

#     Args:
#         prediction (int): the prediction of the model, 0 (healthy) or 1 (not healthy)
#         prediction_prob (float): the probability of the prediction
#     """
#     if prediction == 0:
#         st.markdown(
#             f"**:green[The probability that you will have"
#             f" ... is {round(prediction_prob * 100, 2)}%."
#             f" You seem to be healthy!]**"
#         )
#     else:
#         st.markdown(
#             f"**:red[The probability that you will have"
#             f" ... is {round(prediction_prob * 100, 2)}%."
#             f" It sounds like you are not healthy!]**"
#         )


def main():
    # Add the title and icon of the web page
    st.set_page_config(page_title="Heart Failure Prediction App", page_icon="images/heart-fav.png")

    # Add the title and subtitle of the front page
    st.title("Heart Failure Prediction")
    st.subheader(
        "This web app can tell your probability to get a heart disease based on machine learning models. "
    )

    # Add text on the front page
    st.markdown(
        """
    This application can use several models. You can see the steps of building the model, 
    evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/fangya/machine-learning-survival-for-heart-failure/script).
    
    In order to get a prediction about your heart disease status, you need to take the following steps:
    1. Fill in the asked input features.
    2. Click on the "Predict" button and the prediction will show in a few seconds.
    
    **Keep in mind that this prediction is not the same as a medical diagnosis. 
    It is based on a machine learning model and has an accuracy far from perfect.
    Thus if you experience any health problems, please consult a human doctor.**
    
    **Author: Lisanne Wallaard**
    
    *Based on this [application](https://github.com/kamilpytlak/heart-condition-checker)*
    """
    )

    # Add a sidebar with a picture for the input to the front page
    st.sidebar.title("Input features")

    # Get the input data from the user
    df_input = input_user()

    # Print the input data
    st.dataframe(df_input)

    # Add a button to the side bar to submit the input data
    submission = st.sidebar.button("Predict", type="secondary", use_container_width=True)

    # Add a button to the side bar to stop the application
    stop = st.sidebar.button(label="Stop", type="primary", use_container_width=True)

    # Load the machine learning model
    model_ml = r.readRDS(PATH_MODEL)

    # Load the survival library in R
    survival = importr("survival")

    # Stop the application
    if stop:
        os._exit(0)

    # Print the prediction
    if submission:
        # Get the class prediction

        # https://www.rdocumentation.org/packages/survival/versions/3.5-5/topics/predict.coxph
        # lp = survival.predict_coxph(model_ml, newdata=df_input, type="lp")
        # st.write(lp)
        # risk = survival.predict_coxph(model_ml, newdata=df_input, type="risk")
        # st.write(risk)

        # expected = survival.predict_coxph(model_ml, newdata=df_input, type='expected')
        # st.write(expected)
        # terms = survival.predict_coxph(model_ml, newdata=df_input, type='terms')
        # st.write(terms)
        survival = survival.predict_coxph(model_ml, newdata=df_input, type="survival")
        st.write(survival)


if __name__ == "__main__":
    main()
