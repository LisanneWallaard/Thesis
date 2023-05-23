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
# Path to the model
PATH_MODEL = "model/cox_num.rds"

# Path to R on your device
PATH_R = 'C:/Program Files/R/R-4.3.0'

# Necessary imports
# Installation of these libraries on your device are needed 
import os
# Set your R path
os.environ['R_HOME'] = PATH_R

import numpy as np
import pyreadr
import pandas as pd
import streamlit as st
import rpy2.robjects as ro
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter 
from lifelines import CoxPHFitter

# Activate converters
pandas2ri.activate()
numpy2ri.activate()

# Set R
r = ro.r

def input_user() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user
    """   
    time = st.sidebar.number_input("Over how mandy days do you want to know your survival prediction?", min_value=0, step = 1)
    age = st.sidebar.number_input("What is your age?", min_value=0, step = 1)
    ejection_fraction = st.sidebar.number_input("What is your ejection fraction", min_value=0, step = 1)
    serum_creatinine = st.sidebar.number_input("What is your serum creatine?", min_value=0, step = 1)
    serum_sodium = st.sidebar.number_input("What is your serium sodium?", min_value=0, step = 1)
    
    # Dataframe containing the input of the user
    input_df = pd.DataFrame({
        "time": [time],
        "DEATH_EVENT": [0],
        "age": [age],
        "ejection_fraction": [ejection_fraction], 
        "serum_creatinine": [serum_creatinine],
        "serum_sodium": [serum_sodium],
    })

    return input_df

def preprocess_input(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Returns a DataFrame with the preprocessed input of the user
    """
    
    # One-hot encoder for some categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore')
    
    # Label encoder for some categorical variables
    le = LabelEncoder()
    
    # Define some categorical columns for labelling
    label_columns = ['Sex', 'FastingBS', 'RestingECG.LVH',  'ST_Slope']
    
    # Label encode some categorical variables
    for col in label_columns:
        df_nan = df[col].dropna()
        df[col] = pd.DataFrame(le.fit_transform(df_nan))
    
    # # Label some categorical variables
    # for col in label_columns:
    #     dummy_column = pd.get_dummies(df[col], prefix=col)
    #     df = pd.concat([df, dummy_column], axis=1)
    #     del df[col]
    
    # Define some categorical columns for one-hot encoding
    hot_columns = ["ChestPainType", "ExerciseAngina"]
    # Define the corresponding column names for one-hot encoding (same index as hot_columns)
    column_names = [["ChestPainType.ASY", "ChestPainType.ATA", "ChestPainType.NAP", "ChestPainType.TA"], ["ExerciseAngina.N", "ExerciseAngina.Y"]]
    
    # One-hot encode some categorical variables
    for col in hot_columns:
        df_nan = df[[col]].dropna()
        encoder_df = pd.DataFrame(ohe.fit_transform(df_nan).toarray(), columns=column_names[hot_columns.index(col)])
        df = df.join(encoder_df)
        del df[col]
    
    # Select only the first row (the user input data)  
    df = df[:1]
    
    return df

def output_prediction(prediction: int, prediction_prob: float):
    if int(prediction[0][0]) == 0:
        st.markdown(f"**The probability that you'll not have"
                    f" a stroke is {round(prediction_prob[0][0] * 100, 2)}%."
                    f" You are healthy!**")
        st.image("images/heart-okay.jpg",
                    caption="You seem to be okay! - Dr. Logistic Regression")
    else:
        st.markdown(f"**The probability that you will have"
                    f" stroke is {round(prediction_prob[0][0] * 100, 2)}%."
                    f" It sounds like you are not healthy.**")
        st.image("images/heart-bad.jpg",
                    caption="I'm not satisfied with the condition of health! - Dr. Logistic Regression")
        
    

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
        evaluating it, and cleaning the data itself on [Kaggle](https://www.kaggle.com/code/burakdilber/heart-failure-eda-preprocessing-and-10-models). 
        
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
    st.dataframe(df_input)
    
    # r_df = pandas2ri.py2rpy(df_input)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df= ro.conversion.rpy2py(df_input)
    
    st.dataframe(r_df)    
    # Load the machine learning model
    model_ml = r.readRDS(PATH_MODEL)

    # model_py = pandas2ri.py2rpy(model_ml)
    
    # model_cox = CoxPHFitter()
    survival = importr('survival')

    # Print the prediction
    # https://stat.ethz.ch/R-manual/R-devel/library/survival/html/predict.coxph.html
    if submission:
        # Get the class prediction
        # prediction = model_ml.rx2('predict')(newdata=r_df)
        risk = survival.predict_coxph(model_ml, newdata=r_df, type='lp')
        # probability = np.exp(-np.cumsum(model_ml.rx2['baseline_hazard'] * np.exp(risk)))
        # probability = pandas2ri.ri2py(prediction.rx2('surv'))
        
        # Get the probability of both classes
        # probability = r.predict(model_ml, new_data=r_df, type='prob')
        # Print the prediction
        # output_prediction(prediction, probability)
        
        # prediction = model_ml.predict(r_df)
        # probability = model_ml.predict_proba(r_df)
        
        # st.write(prediction)
        st.write(risk)
        


if __name__ == "__main__":
    main()