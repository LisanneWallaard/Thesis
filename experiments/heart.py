"""
Assignment
    A heart failure prediction application for ML models in python using streamlit
Course
    Thesis
Author
    Lisanne Wallaard
Date
    April 2023
"""

# Necessary imports
# Make sure libraries have the correct version (see requirements.txt)
# Some libraries require additional libraries to be installed
import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Path to the model and shap values
PATH_MODEL = "model/lr_heart.pkl"
plot = "shap"
PATH_SHAP = "explain/shap_val_lr.pkl"
# PATH_EXPL = "explain/expl_lr.pkl"

# PATH_MODEL = "model/dt_heart.pkl"
# plot = "feature_importance"

# PATH_MODEL = "model/knn_heart.pkl"
# plot = 'not_given'

# PATH_MODEL = "model/rf_heart.pkl"
# plot = 'feature_importance'

# PATH_MODEL = "model/svc_heart.pkl" # set probability=True during fitting
# plot = 'shap'
# PATH_SHAP = "explain/shap_val_svc.pkl"
# PATH_EXPL = "explain/expl_svc.pkl"

# Constants for scaling the input
# Statistics of oldpeak
min_oldpeak = -2.6
max_oldpeak = 6.2

# Statistics of age
mean_age = 53.51
std_age = 9.43

# Statistics of cholesterol
mean_cholesterol = 198.80
std_cholesterol = 109.38

# Statistics of MaxHR
mean_MaxHR = 136.81
std_MaxHR = 25.46


def input_user():
    """Gives input possibilities for the user and saves their response

    Returns:
        pd.DataFrame: input_df contains the input of the user
        pd.DataFrame: options_df contains the input possibilities for the user
    """
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    options_sex = ("Female", "Male")
    sex = st.sidebar.selectbox("What is your sex?", options=options_sex)
    options_chest_pain = ("ASY", "ATA", "NAP", "TA")
    chest_pain_type = st.sidebar.selectbox("What kind of chestpain do you have?", options=options_chest_pain)
    cholesterol = st.sidebar.number_input("What is your cholesterol (mg/dL)?", min_value=0, step=1)
    options_yes_no = ("No", "Yes")
    fasting_bs = st.sidebar.selectbox("Do you have fasting blood sugar?", options=options_yes_no)
    max_hr = st.sidebar.number_input("What is your maximum heart rate achieved?", min_value=0, step=1)
    exercise_angina = st.sidebar.selectbox("Do you have exercise induced angina?", options=options_yes_no)
    oldpeak = st.sidebar.number_input("What is your oldpeak (ST)?", step=0.1, format="%.1f")
    options_st_slope = ("Down", "Flat", "Up")
    st_slope = st.sidebar.selectbox("What is your ST slope?", options=options_st_slope)

    # Dataframe containing the input of the user
    input_df = pd.DataFrame(
        {
            "Age": [age],
            "Sex": [sex],
            "ChestPainType": [chest_pain_type],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope],
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
    # Convert categorical variables to dummy variables
    df["Sex"] = df["Sex"].replace({"Male": 1, "Female": 0}).astype(np.uint8)
    df["ChestPainType"] = (
        df["ChestPainType"].replace({"ASY": 0, "ATA": 1, "NAP": 2, "TA": 3}).astype(np.uint8)
    )
    df["ExerciseAngina"] = df["ExerciseAngina"].replace({"No": 0, "Yes": 1}).astype(np.uint8)
    df["ST_Slope"] = df["ST_Slope"].replace({"Down": 0, "Flat": 1, "Up": 2}).astype(np.uint8)
    df["FastingBS"] = df["FastingBS"].replace({"No": 0, "Yes": 1}).astype(np.uint8)

    # Scale numerical variables
    df["Oldpeak"] = df[["Oldpeak"]].apply(
        lambda x: ((x - min_oldpeak) / (max_oldpeak - min_oldpeak)) * (max_oldpeak - min_oldpeak)
        + min_oldpeak
    )
    df["Age"] = df[["Age"]].apply(lambda x: ((x - mean_age) / std_age))
    df["Cholesterol"] = df[["Cholesterol"]].apply(lambda x: ((x - mean_cholesterol) / std_cholesterol))
    df["MaxHR"] = df[["MaxHR"]].apply(lambda x: ((x - mean_MaxHR) / std_MaxHR))

    return df


def output_prediction(prediction: int, prediction_prob: float):
    """Prints the prediction itself and the probability of the prediction

    Args:
        prediction (int): the prediction of the model, 0 (healthy) or 1 (not healthy)
        prediction_prob (float): the probability of the prediction
    """
    if prediction == 0:
        st.markdown(
            f"**:green[The probability that you'll have"
            f" a heart disease is {round(prediction_prob * 100, 2)}%."
            f" You seem to be healthy!]**"
        )
    else:
        st.markdown(
            f"**:red[The probability that you will have"
            f" a heart disease is {round(prediction_prob * 100, 2)}%."
            f" It sounds like you are not healthy!]**"
        )


def plot_feature_importance(feature_importances, feature_names):
    """Plots the feature importances of a model

    Args:
        feature_importance: contains the feature importances of the model
        feature_names: contains the feature names of the model
    """
    # Sort the features on Importance
    index_sorted = feature_importances.argsort()

    # Plot the Importance of the features
    fig, ax = plt.subplots()
    ax.barh(feature_names[index_sorted], feature_importances[index_sorted])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Features sorted by Importance")
    st.pyplot(fig)


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
    evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/tanmay111999/heart-failure-prediction-cv-score-90-5-models).
    
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
    st.sidebar.title("Input Features")

    # Get the input data from the user
    df_input = input_user()
    # Preprocess the input data
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

    # Predict the result
    if submission:
        # Get the class prediction
        prediction = model_ml.predict(df)

        # Get the probability of both classes
        prediction_prob = model_ml.predict_proba(df)

        # Print the prediction
        output_prediction(prediction[0], prediction_prob[0][1])

        # Explain the model if given
        # Plot the feature importances of the model
        if plot == "feature_importance":
            st.markdown(
                """To explain how the prediction of the model is made, 
                        the feature importances of the model is shown below."""
            )
            # Calculate the Importance of the features
            feature_importances = np.zeros(len(df.columns))
            feature_importances = np.add(feature_importances, model_ml.feature_importances_)
            plot_feature_importance(feature_importances, df.columns)

        # Plot the SHAP values of the model
        elif plot == "shap":
            st.markdown(
                """To explain how the prediction of the model is made, 
                        the SHAP values of the model is shown below."""
            )
            # expl = pickle.load(open(PATH_EXPL, "rb")) # geeft error TypeError: code() argument 13 must be str, not int
            # shap_val = expl(df)
            # Load the SHAP values
            shap_val = pickle.load(open(PATH_SHAP, "rb"))
            # Plot the SHAP values as a bar plot
            st_shap(shap.plots.bar(shap_val))
            # st_shap(shap.force_plot(expl.expected_value, shap_val, df))


if __name__ == "__main__":
    main()
