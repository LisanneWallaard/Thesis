# Thesis 
Model Deployment in Healthcare

## Table of Contents
1. [General info](#general-info)
2. [Technologies](#technologies)
3. [Installation](#installation)

## General info
This research aims to reduce the gap between developing and implementing predictive machine learning models. To do so, a framework has been made that simplifies the implementation of these models. The framework results from multiple experiments (see `experiments`) containing applications of machine learning models created in R and Python:
- `app.py` provides code that implements a logistic regression concerning heart disease. The code is from: https://github.com/kamilpytlak/heart-condition-checker.
- `app2.py` is an adjusted version of `app.py` that works without a dataset.
- `heart.py` provides code that implements models developed in `heart_models.ipynb`.
- `stroke.py` provides code that implements models developed in `stroke_models.ipynb`.
- `heart_R.py`provides code that implements models developed in `heart.R`.
- `cox_R.py` contains not working code trying to implement a COX-PH model developed in `cox.R`.
- `model` is a folder containing the pickle and RDS files of the developed models.
- `data` is a folder containing the data for developing medical machine learning models. 
- `explain` is a folder containing pickle files considering the explanation of the model like SHAP values.
- `images` is a folder containing images used in the applications.

As access to medical data is often an issue due to privacy, no access to the data is necessary in this framework. Only a pickle or RDS file of your model and some information about the preprocessing and naming of the data are needed for a web application on your local device. 

## Technologies
The framework is written in `Python 3.11.12`. `streamlit 1.22.0` is used to create a web app. `pandas 1.5.3` and `numpy 1.23.5` are used to handle input data from the user and information of the model. `rpy2 3.4.3` is necessary for managing R models in the Python framework and uses `R-4.3.0`. `scikit-learn 1.2.2` can be used for label and one-hot encoding of input data. `matplotlib 3.7.1` is used for plotting the feature importances of the model. `shap 0.41.0` and `1.0.2` are used to plot SHAP values. 

## Installation
To work with the framework on your machine, the libraries specified in `requirements.txt` should be installed in a virtual environment or globally on your device. To work with the framework and implement your machine learning model, the following steps should be taken: 
1. Clone this repository somewhere on your machine.
2.  Install the required packages using `requirements.txt`:
```
pip install -r requirements.txt
```
3. Adapt the framework to your model.
4. Run your application made with the framework:
```
streamlit run framework.py
```
