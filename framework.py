"""
Assignment
    A framework for the implementation of ML models in Python using streamlit
Course
    Thesis
Author
    Lisanne Wallaard
Date
    May 2023
"""

# Imports
# Make sure libraries have the correct version (see requirements.txt)
# Some libraries require additional libraries to be installed
import os

# If an R model is used, the path to R on your device needs to be set
# This path can be found by entering R.home() in Rstudio for example
PATH_R = "C:/Program Files/R/R-4.3.0"
# Set your R path
os.environ["R_HOME"] = PATH_R

import shap
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import matplotlib.pyplot as plt
from streamlit_shap import st_shap
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Enter your path to the model (pickle or RDS file)
PATH_MODEL = "path_to_model"
# Enter the type of plot you want to see
# Options: 'feature_importance', 'shap' and 'not_given'
# In order to get the feature importance of a model in R, you need to set the importance to impurity
# However, not all models have a feature importance
# In Python it worked for a decision tree and random forest model
# In R it worked for logistic regression, extreme gradient boosting and random forest model
plot = "..."
# For a SHAP plot, enter the path to the shap values (pickle file)
PATH_SHAP = "path_to_shap_values"

# Only when R model is used
# Set R
r = ro.r

# Example constants for scaling the numerical input
# Statistics of age
mean_age = 53.51
std_age = 9.43

# Statistics of cholesterol
mean_cholesterol = 198.80
std_cholesterol = 109.38


def input_user():
    """Gives input possibilities for the user and saves their response

    Returns:
        pd.DataFrame: input_df contains the input of the user
        pd.DataFrame: options_df contains the input possibilities for the user
    """
    # Enter the input possibilities of the user needed for the model to predict

    # An example of numerical input (int)
    age = st.sidebar.number_input("What is your age?", min_value=0, step=1)
    # An example of categorical intput (str)
    options_gender = ("Female", "Male", "Other")
    gender = st.sidebar.selectbox("What is your gender?", options=options_gender)
    # An example of numerical input (float)
    glucose_level = st.sidebar.number_input(
        "What is your average glucose level (mg/dL)?", min_value=0.0, step=1.0, format="%.2f"
    )

    # Dataframe containing the input of the user
    input_df = pd.DataFrame({"Age": [age], "Gender": [gender], "avg_glucose_level": [glucose_level]})

    # A dictionary containing the input possibilities for the user
    # A slightly diferent approach to handle the different lengths given
    options_dict = {
        "Age": list(range(0, 126)),
        "Gender": list(options_gender),
        "avg_glucose_level": list(range(0, 604)),
    }
    # Convert the dictionary to a DataFrame
    # This DataFrame is not always needed, but can be convenient for preprocessing
    options_df = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in options_dict.items()]))

    return input_df, options_df


def preprocess_input(df: pd.DataFrame):
    """Preprocesses the input of the user

    Args:
        df (pd.DataFrame): contains not preprocessed input of the user

    Returns:
        pd.DataFrame: df contains preprocessed input of the user
    """
    # Preprocessing of the input data needs to be done in the same way as the data for the model
    # Two examples of label encoding data
    # Manual label encoding
    df["Gender"] = df["Gender"].replace({"Male": 0, "Female": 1, "Other": -1}).astype(np.uint8)
    # Label encoder for categorical variables
    le = LabelEncoder()
    # A list containing the columns for label encoding
    label_columns = ["Gender"]
    # Label encode the columns of variables in label_columns
    for col in label_columns:
        df_nan = df[col].dropna()
        df[col] = pd.DataFrame(le.fit_transform(df_nan))

    # It is also possible to use one hot encoding for categorical variables
    ohe = OneHotEncoder(handle_unknown="ignore")
    # A list containing the columns for one hot encoding
    hot_columns = ["Gender"]
    # A list of lists containing the corresponding column names for one-hot encoding (same index as hot_columns)
    # It is important to know that the one-hot encoder encodes the new columns
    # in the alphabetical order of the input possibilites
    # Your column names should therefore not only be the same as those of the training data,
    # but also match this alphabetical order
    column_names = [["Female", "Male", "Other"]]
    # One-hot encode some categorical variables
    for col in hot_columns:
        df_nan = df[[col]].dropna()
        encoder_df = pd.DataFrame(
            ohe.fit_transform(df_nan).toarray(), columns=column_names[hot_columns.index(col)]
        )
        df = df.join(encoder_df)
        del df[col]

    # An example of scaling numerical variables
    df["Age"] = df[["Age"]].apply(lambda x: ((x - mean_age) / std_age))
    df["avg_glucose_level"] = df[["avg_glucose_level"]].apply(
        lambda x: ((x - mean_cholesterol) / std_cholesterol)
    )

    # Select only the first row (the user input data) if multiple rows are present
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
            f" ... is {round(prediction_prob * 100, 2)}%."
            f" You seem to be healthy!]**"
        )
    else:
        st.markdown(
            f"**:red[The probability that you will have"
            f" ... is {round(prediction_prob * 100, 2)}%."
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
    # The icon can be downloaded from https://github.com/LisanneWallaard/Thesis/blob/main/heart-fav.png
    # It is also possible to add your own image and change the path
    st.set_page_config(page_title="... App", page_icon="heart-fav.png")

    # Add the title and subtitle of the front page
    st.title("... Prediction")
    st.subheader("This web app can tell your probability to get ... based on a machine learning model. ")

    # Add text on the front page
    st.markdown(
        """
    You can see the steps of building the model, evaluating it, 
    and cleaning the data itself on [...](link).
    
    In order to get a prediction about ..., you need to take the following steps:
    1. Fill in the asked input features.
    2. Click on the "Predict" button and the prediction will show in a few seconds.
    
    **Keep in mind that this prediction is not the same as a medical diagnosis. 
    It is based on a machine learning model and has an accuracy far from perfect.
    Thus if you experience any health problems, please consult a human doctor.**
    
    **Author: ...**
    
    *Made this application with the help of a [framework](https://github.com/LisanneWallaard/Thesis)*
    """
    )

    # Add a sidebar for the input to the front page
    st.sidebar.title("Input Features")

    # Get the input data and option possibilities from the user
    # If you don't want to use the options, you can replace df_options with _ or change the input_user function
    # Obviously, you do not need to merge and can just pass df_input into preprocess_input
    df_input, df_options = input_user()
    df_merge = pd.concat([df_input, df_options], axis=0)

    # Preprocess the input data
    df = preprocess_input(df_merge)
    # It is highly recommended to print this DataFrame and check if the input is correctly imported into the DataFrame:
    # st.dataframe(df)

    # You might want to change the order of the columns although it does not matter for the model
    # However, it is important that the column names of the input data are the same as those of the training data
    order = ["Age", "Male", "Female", "Other", "avg_glucose_level"]
    df = df[order]

    # Add a button to the side bar to submit the input data
    submission = st.sidebar.button("Predict", type="secondary", use_container_width=True)

    # Add a button to the side bar to stop the application
    stop = st.sidebar.button(label="Stop", type="primary", use_container_width=True)

    # Load the machine learning model saved as a pickle file
    model_pkl = r.readRDS(PATH_MODEL)
    # Load the machine learning model saved as a RDS file
    model_rds = pickle.load(open(PATH_MODEL, "rb"))

    # Stop the application
    if stop:
        os._exit(0)

    # Predict the result
    if submission:
        # Get the class prediction of a pickle model
        prediction = model_pkl.predict(df)
        # Get the class prediction of a RDS model
        prediction = r.predict(model_rds, new_data=df, type="class")
        # You might want to print the prediction directly to check if it is correct
        # st.write(prediction)

        # Get the probability of both classes of a pickle model
        # For a svc/svm model it is needed to set probability=True during fitting
        prediction_prob = model_pkl.predict_proba(df)
        # Get the probability of both classes of a RDS model
        prediction_prob = r.predict(model_rds, new_data=df, type="prob")
        # You might want to print the whole probability of the prediction directly to check if it is correct
        # st.write(prediction_prob)

        # Print the prediction of a pickle model
        output_prediction(prediction[0], prediction_prob[0][1])

        # Print the prediction of a RDS model
        output_prediction(int(prediction.iloc[0, 0]), prediction_prob.iloc[0, 1])

        # Explain the model if given
        # Plot the feature importances of the model
        if plot == "feature_importance":
            st.markdown(
                """To explain how the prediction of the model is made, 
                        the feature importances of the model is shown below."""
            )
            # The following approach applies for a pickle model
            # Calculate the Importance of the features
            feature_importances_pkl = np.zeros(len(df.columns))
            feature_importances_pkl = np.add(feature_importances_pkl, model_pkl.feature_importances_)
            plot_feature_importance(feature_importances_pkl, df.columns)

            # This approach applies for a RDS model
            # Load the vip library in R
            vip = importr("vip")
            # Get the feature importances of the model as R DataFrame
            feature_importances_R = vip.vip(model_rds)
            feature_importances_R = feature_importances_R.rx2("data")
            plot_feature_importance(feature_importances_R["Importance"], feature_importances_R["Variable"])

        # Plot the SHAP values of the model
        # Has only been tested for SHAP values of a pickle model
        elif plot == "shap":
            st.markdown(
                """To explain how the prediction of the model is made, 
                        the SHAP values of the model is shown below."""
            )
            # Load the SHAP values
            shap_val = pickle.load(open(PATH_SHAP, "rb"))
            # Plot the SHAP values as a bar plot
            st_shap(shap.plots.bar(shap_val))


if __name__ == "__main__":
    main()
