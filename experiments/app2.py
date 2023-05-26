"""
Assignment
    A heart disease prediction application for ML models 
    using streamlit without the use of a database
Course
    Thesis
Source
    Kamil Pytlak
Edited by
    Lisanne Wallaard
Date
    April 2023
"""

# Necessary imports
# Installation of these libraries on your device are needed
import streamlit as st
import pandas as pd
import pickle

# Path to the model
LOG_MODEL_PATH = "model/logistic_regression.pkl"


def user_input_features() -> pd.DataFrame:
    """
    Returns a DataFrame with the input of the user and
    a dataframe with the input possibilities for the user
    """
    options_race = ("American Indian/Alaskan Native", "Asian", "Black", "Hispanic", "White", "Other")
    race = st.sidebar.selectbox("Race", options=options_race)
    options_sex = ("Female", "Male")
    sex = st.sidebar.selectbox("Sex", options=options_sex)
    options_age_cat = (
        "18-24",
        "25-29",
        "30-34",
        "35-39",
        "40-44",
        "45-49",
        "50-54",
        "55-59",
        "60-64",
        "65-69",
        "70-74",
        "75-79",
        "80 or older",
    )
    age_cat = st.sidebar.selectbox("Age category", options=options_age_cat)
    options_bmi_cat = (
        "Underweight (BMI < 18.5)",
        "Normal weight (18.5 <= BMI < 25.0)",
        "Overweight (25.0 <= BMI < 30.0)",
        "Obese (30.0 <= BMI < +Inf)",
    )
    bmi_cat = st.sidebar.selectbox("BMI category", options=options_bmi_cat)
    sleep_time = st.sidebar.number_input("How many hours on average do you sleep?", 0, 24, 7)
    options_gen_health = ("Excellent", "Very good", "Good", "Fair", "Poor")
    gen_health = st.sidebar.selectbox("How can you define your general health?", options=options_gen_health)
    phys_health = st.sidebar.number_input(
        "For how many days during the past 30 days was" " your physical health not good?", 0, 30, 0
    )
    ment_health = st.sidebar.number_input(
        "For how many days during the past 30 days was" " your mental health not good?", 0, 30, 0
    )
    options_yes_no = ("No", "Yes")
    phys_act = st.sidebar.selectbox(
        "Have you played any sports (running, biking, etc.)" " in the past month?", options=options_yes_no
    )
    smoking = st.sidebar.selectbox(
        "Have you smoked at least 100 cigarettes in" " your entire life (approx. 5 packs)?)",
        options=options_yes_no,
    )
    alcohol_drink = st.sidebar.selectbox(
        "Do you have more than 14 drinks of alcohol (men)" " or more than 7 (women) in a week?",
        options=options_yes_no,
    )
    stroke = st.sidebar.selectbox("Did you have a stroke?", options=options_yes_no)
    diff_walk = st.sidebar.selectbox(
        "Do you have serious difficulty walking" " or climbing stairs?", options=options_yes_no
    )
    options_diabetic = ("No", "No, borderline diabetes", "Yes", "Yes (during pregnancy)")
    diabetic = st.sidebar.selectbox("Have you ever had diabetes?", options=options_diabetic)
    asthma = st.sidebar.selectbox("Do you have asthma?", options=options_yes_no)
    kid_dis = st.sidebar.selectbox("Do you have kidney disease?", options=options_yes_no)
    skin_canc = st.sidebar.selectbox("Do you have skin cancer?", options=options_yes_no)

    # Dataframe containing the input of the user
    features_df = pd.DataFrame(
        {
            "PhysicalHealth": [phys_health],
            "MentalHealth": [ment_health],
            "SleepTime": [sleep_time],
            "BMICategory": [bmi_cat],
            "Smoking": [smoking],
            "AlcoholDrinking": [alcohol_drink],
            "Stroke": [stroke],
            "DiffWalking": [diff_walk],
            "Sex": [sex],
            "AgeCategory": [age_cat],
            "Race": [race],
            "Diabetic": [diabetic],
            "PhysicalActivity": [phys_act],
            "GenHealth": [gen_health],
            "Asthma": [asthma],
            "KidneyDisease": [kid_dis],
            "SkinCancer": [skin_canc],
        }
    )

    # Dataframe containing the input possibilities for the user
    inputs_dict = {
        "PhysicalHealth": list(range(0, 31)),
        "MentalHealth": list(range(0, 31)),
        "SleepTime": list(range(0, 31)),
        "BMICategory": list(options_bmi_cat),
        "Smoking": list(options_yes_no),
        "AlcoholDrinking": list(options_yes_no),
        "Stroke": list(options_yes_no),
        "DiffWalking": list(options_yes_no),
        "Sex": list(options_sex),
        "AgeCategory": list(options_age_cat),
        "Race": list(options_race),
        "Diabetic": list(options_diabetic),
        "PhysicalActivity": list(options_yes_no),
        "GenHealth": list(options_gen_health),
        "Asthma": list(options_yes_no),
        "KidneyDisease": list(options_yes_no),
        "SkinCancer": list(options_yes_no),
    }
    inputs_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in inputs_dict.items()]))

    return features_df, inputs_df


def main():
    # Add the title and icon of the web page
    st.set_page_config(page_title="Heart Disease Prediction App", page_icon="images/heart-fav.png")

    # Add the title and subtitle of the front page
    st.title("Heart Disease Prediction")
    st.subheader(
        "Are you wondering about the condition of your heart? " "This app will help you to diagnose it!"
    )

    col1, col2 = st.columns([1, 3])

    # Add an image on the front page
    with col1:
        st.image(
            "images/doctor.png",
            caption="I'll help you diagnose your heart health! - Dr. Logistic Regression",
            width=150,
        )
        submit = st.button("Predict")

    # Add text on the front page
    with col2:
        st.markdown(
            """
        Did you know that machine learning models can help you
        predict heart disease pretty accurately? In this app, you can
        estimate your chance of heart disease (yes/no) in seconds!
        
        Here, a logistic regression model using an undersampling technique
        was constructed using survey data of over 300k US residents from the year 2020.
        This application is based on it because it has proven to be better than the random forest
        (it achieves an accuracy of about 80%, which is quite good).
        
        To predict your heart disease status, simply follow the steps bellow:
        1. Enter the parameters that best describe you;
        2. Press the "Predict" button and wait for the result.
            
        **Keep in mind that this results is not equivalent to a medical diagnosis!
        This model would never be adopted by health care facilities because of its less
        than perfect accuracy, so if you have any problems, consult a human doctor.**
        
        **Author: Kamil Pytlak ([GitHub](https://github.com/kamilpytlak/heart-condition-checker))**
        
        **Edited by Lisanne Wallaard**

        You can see the steps of building the model, evaluating it, and cleaning the data itself
        on my GitHub repo [here](https://github.com/kamilpytlak/data-analyses/tree/main/heart-disease-prediction). 
        """
        )

    # Add a sidebar with a picture for the input to the front page
    st.sidebar.title("Feature Selection")
    st.sidebar.image("images/heart-sidebar.png", width=100)

    # Get the input data from the user
    input_df, inputs = user_input_features()
    # Concatenate the input data with the input possibilities
    df = pd.concat([input_df, inputs], axis=0)
    # Define the categorical columns
    cat_cols = [
        "BMICategory",
        "Smoking",
        "AlcoholDrinking",
        "Stroke",
        "DiffWalking",
        "Sex",
        "AgeCategory",
        "Race",
        "Diabetic",
        "PhysicalActivity",
        "GenHealth",
        "Asthma",
        "KidneyDisease",
        "SkinCancer",
    ]
    # Encode the categorical variables
    for cat_col in cat_cols:
        dummy_col = pd.get_dummies(df[cat_col], prefix=cat_col)
        df = pd.concat([df, dummy_col], axis=1)
        del df[cat_col]

    # Select only the first row (the user input data)
    df = df[:1]
    # Fill the missing values with 0
    df.fillna(0, inplace=True)

    # Load the machine learning model
    log_model = pickle.load(open(LOG_MODEL_PATH, "rb"))

    # Make prediction with the input data of the user
    if submit:
        # Get the class prediction
        prediction = log_model.predict(df)
        # Get the probability of both classes
        prediction_prob = log_model.predict_proba(df)
        # Print the prediction
        if prediction == 0:
            st.markdown(
                f"**The probability that you'll have"
                f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                f" You are healthy!**"
            )
            st.image(
                "images/heart-okay.jpg", caption="Your heart seems to be okay! - Dr. Logistic Regression"
            )
        else:
            st.markdown(
                f"**The probability that you will have"
                f" heart disease is {round(prediction_prob[0][1] * 100, 2)}%."
                f" It sounds like you are not healthy.**"
            )
            st.image(
                "images/heart-bad.jpg",
                caption="I'm not satisfied with the condition of your heart! - Dr. Logistic Regression",
            )


if __name__ == "__main__":
    main()
