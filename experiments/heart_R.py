"""
Assignment
    A heart failure prediction application for ML models in R using streamlit
Course
    Thesis
Author
    Lisanne Wallaard
Date
    April 2023
"""

# Inspirated by
# https://goddoe.github.io/r/machine%20learning/2017/12/17/how-to-use-r-model-in-python.html

# Necessary imports
# Make sure libraries have the correct version (see requirements.txt)
# Some libraries require additional libraries to be installed
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
import matplotlib.pyplot as plt


# Path to the model
PATH_MODEL = "model/rf_heart.rds"
plot = "feature_importance"  # set importance to impurity

# PATH_MODEL = "model/xgb_heart.rds"
# plot = 'feature_importance' # set importance to impurity

# PATH_MODEL = "model/bag_mars_heart.rds"
# plot = 'not_given'

# PATH_MODEL = "model/mars_heart.rds"
# plot = 'not_given'

# PATH_MODEL = "model/knn_heart.rds"
# plot = 'not_given'

# PATH_MODEL = "model/log_heart.rds"
# plot = 'feature_importance' # set importance to impurity

# PATH_MODEL = "model/nb_heart.rds"
# plot = 'not_given'

# Set R
r = ro.r


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
    resting_bp = st.sidebar.number_input("What is your resting blood pressure (mmHg)?", min_value=0, step=1)
    cholesterol = st.sidebar.number_input("What is your cholesterol (mg/dL)?", min_value=0, step=1)
    options_yes_no = ("No", "Yes")
    fasting_bs = st.sidebar.selectbox("Do you have fasting blood sugar?", options=options_yes_no)
    resting_ecg = st.sidebar.selectbox("Do you have a LVH resting ECG?", options=options_yes_no)
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
            "RestingBP": [resting_bp],
            "Cholesterol": [cholesterol],
            "FastingBS": [fasting_bs],
            "RestingECG.LVH": [resting_ecg],
            "MaxHR": [max_hr],
            "ExerciseAngina": [exercise_angina],
            "Oldpeak": [oldpeak],
            "ST_Slope": [st_slope],
        }
    )

    # A dictionary containing the input possibilities for the user
    # A slightly diferent approach to handle the different lengths given
    options_dict = {
        "Age": list(range(0, 126)),
        "Sex": list(options_sex),
        "ChestPainType": list(options_chest_pain),
        "RestingBP": list(range(0, 201)),
        "Cholesterol": list(range(0, 604)),
        "FastingBS": list(options_yes_no),
        "RestingECG.LVH": list(options_yes_no),
        "MaxHR": list(range(60, 203)),
        "ExerciseAngina": list(options_yes_no),
        "Oldpeak": list(np.arange(-2.6, 6.2, 0.1)),
        "ST_Slope": list(options_st_slope),
    }
    # Convert the dictionary to a DataFrame
    options_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in options_dict.items()]))

    return input_df, options_df


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


def output_prediction(prediction: int, prediction_prob: float):
    """Prints the prediction itself and the probability of the prediction

    Args:
        prediction (int): the prediction of the model, 0 (healthy) or 1 (not healthy)
        prediction_prob (float): the probability of the prediction
    """
    if prediction == 0:
        st.markdown(
            f"**:green[The probability that you will have"
            f" a heart disease is {round(prediction_prob * 100, 2)}%."
            f" You seem to be healthy!]**"
        )
    else:
        st.markdown(
            f"**:red[The probability that you will have"
            f" a heart disease is {round(prediction_prob * 100, 2)}%."
            f" It sounds like you are not healthy!]**"
        )


def plot_feature_importance(feature_importance, feature_names):
    """Plots the feature importances of a model

    Args:
        feature_importance: contains the feature importances of the model
        feature_names: contains the feature names of the model
    """
    fig, ax = plt.subplots()
    index_sorted = feature_importance.argsort()
    ax.barh(feature_names[index_sorted], feature_importance[index_sorted])
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
    st.sidebar.title("Feature Selection")

    # Get the input data from the user
    df_input, df_options = input_user()
    df_merge = pd.concat([df_input, df_options], axis=0)

    # Preprocess the input data
    df = preprocess_input(df_merge)
    # Put the dataframe in the order I prefer
    order = [
        "Age",
        "Sex",
        "ChestPainType.ASY",
        "ChestPainType.ATA",
        "ChestPainType.NAP",
        "ChestPainType.TA",
        "RestingBP",
        "Cholesterol",
        "FastingBS",
        "RestingECG.LVH",
        "MaxHR",
        "ExerciseAngina.N",
        "ExerciseAngina.Y",
        "Oldpeak",
        "ST_Slope",
    ]
    df = df[order]

    # Add a button to the side bar to submit the input data
    submission = st.sidebar.button("Predict", type="secondary", use_container_width=True)

    # Add a button to the side bar to stop the application
    stop = st.sidebar.button(label="Stop", type="primary", use_container_width=True)

    # Load the machine learning model
    model_ml = r.readRDS(PATH_MODEL)

    # Stop the application
    if stop:
        os._exit(0)

    # Print the prediction
    if submission:
        # Get the class prediction
        prediction = r.predict(model_ml, new_data=df, type="class")
        # Get the probability of both classes
        prediction_prob = r.predict(model_ml, new_data=df, type="prob")
        # Print the prediction
        output_prediction(int(prediction.iloc[0, 0]), prediction_prob.iloc[0, 1])

        # Explain the model if given
        # Plot the feature importances of the model
        if plot == "feature_importance":
            # Load the vip library in R
            vip = importr("vip")
            # Get the feature importances of the model as R DataFrame
            feature_importance_R = vip.vip(model_ml)
            feature_importance_R = feature_importance_R.rx2("data")
            st.markdown(
                """To explain how the prediction of the model is made, 
                        the feature importances of the model is shown below."""
            )
            plot_feature_importance(feature_importance_R["Importance"], feature_importance_R["Variable"])


if __name__ == "__main__":

    main()
