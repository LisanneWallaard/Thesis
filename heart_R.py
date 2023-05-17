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

# Path to the model
PATH_MODEL = "model/rf_heart.rds"
# PATH_MODEL = "model/xgb_heart.rds"
# PATH_MODEL = "model/bag_mars_heart.rds"
# PATH_MODEL = "model/mars_heart.rds"
# PATH_MODEL = "model/knn_heart.rds"

# Path to R on your device
PATH_R = 'C:/Program Files/R/R-4.3.0'

# Necessary imports
# Installation of these libraries on your device are needed 
import os
# Set your R path
os.environ['R_HOME'] = PATH_R
import numpy as np
import pandas as pd
import streamlit as st
import rpy2.robjects as ro
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter 

# Activate converters
pandas2ri.activate()
numpy2ri.activate()

# Set R
r = ro.r

def input_user() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user
    """
    
    age = st.sidebar.number_input("What is your age?", min_value=0, step = 1)
    options_sex = ("Female", "Male")
    sex = st.sidebar.selectbox("Sex", options=options_sex)
    options_chest_pain = ("ASY", "ATA", "NAP", "TA")
    chest_pain_type = st.sidebar.selectbox("What kind of chestpain do you have", options=options_chest_pain)
    resting_bp = st.sidebar.number_input("What is your resting blood pressure (mmHg)?", min_value=0, step = 1)
    cholesterol = st.sidebar.number_input("What is your cholesterol (mg/dL)?", min_value=0, step = 1)
    options_yes_no = ("No", "Yes")
    fasting_bs = st.sidebar.selectbox("Do you have fasting blood sugar?", options=options_yes_no)
    resting_ecg = st.sidebar.selectbox("Do you have a LVH resting ECG?", options=options_yes_no)
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
        "RestingBP": [resting_bp],
        "Cholesterol": [cholesterol],
        "FastingBS": [fasting_bs],
        "RestingECG.LVH": [resting_ecg],
        "MaxHR": [max_hr],	
        "ExerciseAngina": [exercise_angina],
        "Oldpeak": [oldpeak],
        "ST_Slope": [st_slope]
    })
    
    # Dataframe containing the input possibilities for the user
    options_dict = {
        "Age": list(range(0,126)),
        "Sex": list(options_sex), 
        "ChestPainType": list(options_chest_pain),
        "RestingBP": list(range(0,201)),
        "Cholesterol": list(range(0,604)),
        "FastingBS": list(options_yes_no),
        "RestingECG.LVH": list(options_yes_no),
        "MaxHR": list(range(60,203)),	
        "ExerciseAngina": list(options_yes_no),
        "Oldpeak": list(np.arange(-2.6, 6.2, 0.1)),
        "ST_Slope": list(options_st_slope)
    }
    options_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in options_dict.items()]))

    return input_df, options_df

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
    df_input, df_options = input_user()
    df_merge = pd.concat([df_input, df_options], axis=0)

    # Preprocess the input data
    df = preprocess_input(df_merge)
    # Put the dataframe in the correct order
    order = ['Age', 'Sex', 'ChestPainType.ASY', 'ChestPainType.ATA', 'ChestPainType.NAP', 'ChestPainType.TA', 'RestingBP',
     'Cholesterol', 'FastingBS', 'RestingECG.LVH', 'MaxHR', 'ExerciseAngina.N', 
     'ExerciseAngina.Y', 'Oldpeak', 'ST_Slope']
    df = df[order]
    
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_df = ro.conversion.rpy2py(df)
  
    # Load the machine learning model
    model_ml = r.readRDS(PATH_MODEL)

    # Print the prediction
    if submission:
        # Get the class prediction
        prediction = r.predict(model_ml, new_data=r_df, type='class')
        # Get the probability of both classes
        probability = r.predict(model_ml, new_data=r_df, type='prob')
        # Print the prediction
        output_prediction(prediction, probability)
        


if __name__ == "__main__":
    main()