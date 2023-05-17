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
# Installation of these libraries on your device are needed 
# make sure libraries are up to date and do no conflict with each other
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import shap # you need to have torch, tensorflow installed
from streamlit_shap import st_shap
import matplotlib.pyplot as plt

# Path to the model and shap values
PATH_MODEL = "model/lr_heart.pkl"
plot = 'shap'
PATH_SHAP = "model/shap_val_lr.pkl"
PATH_EXPL = "model/expl_lr.pkl"
# PATH_MODEL = "model/dt_heart.pkl"
# plot = 'feature_importance'
# PATH_MODEL = "model/knn_heart.pkl"
# plot = 'not_given'
# PATH_MODEL = "model/rf_heart.pkl"
# plot = 'feature_importance'
# PATH_MODEL = "model/svc_heart.pkl" # set probability=True during fitting
# plot = 'shap'
# PATH_SHAP = "model/shap_val_svc.pkl"

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


def input_user() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user
    """
    age = st.sidebar.number_input("What is your age?", min_value=0, step = 1)
    options_sex = ("Female", "Male")
    sex = st.sidebar.selectbox("Sex", options=options_sex)
    options_chest_pain = ("ASY", "ATA", "NAP", "TA")
    chest_pain_type = st.sidebar.selectbox("What kind of chestpain do you have", options=options_chest_pain)
    cholesterol = st.sidebar.number_input("What is your cholesterol (mg/dL)?", min_value=0, step = 1)
    options_yes_no = ("No", "Yes")
    fasting_bs = st.sidebar.selectbox("Do you have fasting blood sugar?", options=options_yes_no)
    max_hr = st.sidebar.number_input("What is your maximum heart rate achieved?", min_value=0, step = 1)
    exercise_angina = st.sidebar.selectbox("Do you have exercise induced angina?", options=options_yes_no)
    oldpeak = st.sidebar.number_input("What is your oldpeak (ST)?",step=0.1,format="%.1f")
    options_st_slope = ("Down", "Flat", "Up")
    st_slope = st.sidebar.selectbox("What is your ST slope?", options=options_st_slope)

    # Dataframe containing the input of the user
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [sex], 
        "ChestPainType": [chest_pain_type],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "MaxHR": [max_hr],	
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })

    return input_df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns a DataFrame with the preprocessed input of the user
    """
    # Convert categorical variables to dummy variables
    df['Sex'] = df['Sex'].replace({'Male':1,'Female':0}).astype(np.uint8)
    df['ChestPainType'] = df['ChestPainType'].replace({'ASY':0,'ATA':1,'NAP':2,'TA':3}).astype(np.uint8)
    df['ExerciseAngina'] = df['ExerciseAngina'].replace({'No':0,'Yes':1}).astype(np.uint8)
    df['ST_Slope'] = df['ST_Slope'].replace({'Down':0,'Flat':1,'Up':2}).astype(np.uint8)
    df['FastingBS'] = df['FastingBS'].replace({'No':0,'Yes':1}).astype(np.uint8)

    # Scale numerical variables
    df['Oldpeak'] = df[['Oldpeak']].apply(lambda x: ((x - min_oldpeak) / (max_oldpeak - min_oldpeak)) * (max_oldpeak - min_oldpeak) + min_oldpeak)
    df['Age'] = df[['Age']].apply(lambda x: ((x - mean_age) / std_age))
    df['Cholesterol'] = df[['Cholesterol']].apply(lambda x: ((x - mean_cholesterol) / std_cholesterol))
    df['MaxHR'] = df[['MaxHR']].apply(lambda x: ((x - mean_MaxHR) / std_MaxHR))
    
    return df

def output_prediction(prediction: int, prediction_prob: float):
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
        
def plot_feature_importance(model_ml, feature_names):
    """
    Plots the feature importance of a model
    """
    # Calculate the Importance of the features
    feature_importance = np.zeros(len(feature_names))
    feature_importance = np.add(feature_importance, model_ml.feature_importances_)

    # Sort the features on Importance
    index_sorted = feature_importance.argsort()

    # Plot the Importance of the features
    fig, ax = plt.subplots()
    ax.barh(feature_names[index_sorted], feature_importance[index_sorted])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Features sorted by Importance")
    st.pyplot(fig) 
    
# def plot_shap():
#     """
#     Plots the shap values of a model
#     """
#     # Load the model
#     model = load_model()
    
#     # Load the data
#     X_train, X_test, y_train, y_test = load_data()
    
#     # Create the explainer
#     explainer = shap.TreeExplainer(model)
    
#     # Calculate the shap values
#     shap_values = explainer.shap_values(X_test)
    
#     # Plot the shap values
#     fig, ax = plt.subplots()
#     shap.summary_plot(shap_values[1], X_test, plot_type="bar")
#     st.pyplot(fig)
        
def main():
    # Add the title and icon of the web page
    st.set_page_config(
        page_title="Heart Failure Prediction App",
        page_icon="images/heart-fav.png"
    )

    # Add the title and subtitle of the front page
    st.title("Heart Failure Prediction")
    st.subheader("Are you wondering about the condition of your heart? "
                 "This app will help you to diagnose it!")

    col1, col2 = st.columns([1, 3])

    # Add an image on the front page
    with col1:
        st.image("images/doctor.png",
                 caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
                 width=150)
        submission = st.button("Predict")
        
    # Add text on the front page
    with col2:
        st.markdown("""
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, the application is based on several models. You can see the steps of building the model, 
        evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/tanmay111999/heart-failure-prediction-cv-score-90-5-models). 
        
        To predict your heart disease status, simply follow the steps bellow:
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
    # Preprocess the input data
    df = preprocess_input(df_input)

    # Load the machine learning model
    model_ml = pickle.load(open(PATH_MODEL, "rb"))

    # predict the result
    if submission:
        # Get the class prediction
        prediction = model_ml.predict(df)
        # Get the probability of both classes
        prediction_prob = model_ml.predict_proba(df)
        # Print the prediction
        output_prediction(prediction, prediction_prob)
        
        # Explain the model
        if plot == 'feature_importance':
            plot_feature_importance(model_ml, df.columns)
        elif plot == 'shap':
            # expl = pickle.load(open(PATH_SHAP, "rb"))
            # shap_val = expl(df)
            shap_val = pickle.load(open(PATH_SHAP, "rb"))
            # expl = pickle.load(open(PATH_EXPL, "rb")) # geeft error TypeError: code() argument 13 must be str, not int
            shap.plots.bar(shap_val)
            st_shap(shap.plots.bar(shap_val))
            # st_shap(shap.force_plot(expl.expected_value, shap_val, df))
        
                


if __name__ == "__main__":
    main()



