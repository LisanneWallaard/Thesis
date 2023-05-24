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

# Path to the model
PATH_MODEL = "model/cox_num.rds"

# Path to R on your device
PATH_R = 'C:/Program Files/R/R-4.3.0'

# Necessary imports
# Installation of these libraries on your device are needed 
# make sure libraries are up to date and do no conflict with each other
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
    time = st.sidebar.number_input("Over how many days do you want to know your survival prediction?", min_value=0, step=1)
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    ejection_fraction = st.sidebar.number_input("What is your ejection fraction?", min_value=0, step=1)
    serum_creatinine = st.sidebar.number_input("What is your serum creatine?", min_value=0.0, step=0.1, format="%.1f")
    serum_sodium = st.sidebar.number_input("What is your serium sodium?", min_value=0, step=1)
    
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

def output_prediction(prediction, prediction_prob):
    if int(prediction[0][0]) == 0:
        st.markdown(f"**The probability that you will have"
                    f" heart failure is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" You seem to be healthy!**")
    else:
        st.markdown(f"**The probability that you will have"
                    f" heart failure is {round(prediction_prob[0][1] * 100, 2)}%."
                    f" It sounds like you are not healthy.**")
        
    

def main():
    # Add the title and icon of the web page
    st.set_page_config(
        page_title="Heart Failure Prediction App",
        page_icon="images/heart-fav.png"
    )

    # Add the title and subtitle of the front page
    st.title("Heart Failure Prediction")
    st.subheader("This web app can tell your probability to get a heart disease based on machine learning models. ")  
        
    # Add text on the front page
    st.markdown("""
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
    """)

    # Add a sidebar with a picture for the input to the front page
    st.sidebar.title("Feature Selection")

    # Get the input data from the user
    df_input = input_user()
    st.dataframe(df_input)
    
    # r_df = pandas2ri.py2rpy(df_input)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df= ro.conversion.rpy2py(df_input)  
    st.dataframe(r_df)    
    
    # Add a button to the side bar to submit the input data
    submission = st.sidebar.button("Predict", type="secondary", use_container_width=True)

    # Add a button to the side bar to stop the application
    stop = st.sidebar.button(label="Stop", type="primary", use_container_width=True)
    
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
        
        lp = survival.predict_coxph(model_ml, newdata=r_df, type='lp')
        st.write(lp)
        risk = survival.predict_coxph(model_ml, newdata=r_df, type='risk')
        st.write(risk)
        # expected = survival.predict_coxph(model_ml, newdata=r_df, type='expected')
        # st.write(expected)
        # terms = survival.predict_coxph(model_ml, newdata=r_df, type='terms')
        # st.write(terms)
        # https://www.rdocumentation.org/packages/survival/versions/3.5-5/topics/predict.coxph
        survival = survival.predict_coxph(model_ml, newdata=r_df, type='survival')
        st.write(survival)
        
        # probability = np.exp(-np.cumsum(model_ml.rx2['baseline_hazard'] * np.exp(risk)))
        # probability = pandas2ri.ri2py(prediction.rx2('surv'))
        
        # Get the probability of both classes
        # probability = r.predict(model_ml, new_data=r_df, type='prob')
        # Print the prediction
        # output_prediction(prediction, probability)
        
        # prediction = model_ml.predict(r_df)
        # probability = model_ml.predict_proba(r_df)
        
        # st.write(prediction)
    
    if stop:
        os._exit(0)    


if __name__ == "__main__":
    main()