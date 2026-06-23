import pandas as pd
import streamlit as st
import joblib

# Load your trained ML model
model = joblib.load('classifier.pkl')

# Create input fields in the sidebar
st.sidebar.header('User Input Parameters')

def user_input_features():
    discharge_mapping = {"Frequent Urination": 0, "Yellow,green discharge": 1, "Painful Urination": 2, "Difficulties in stooling": 3}
    feelings_and_urge_mapping = {"Fever": 0, "Tired": 1, "Hunger": 2, "Urge to urinate": 3}
    pain_and_infection_mapping = {"Frequent Infection": 0, "Swollen genitals": 1, "Headache": 2, "Anal Itching and Pain": 3, "Muscle Aches": 4, "Blurred Vision": 5, "Abdominal Pain": 6}
    physical_condition_mapping = {"Rashes": 0, "NIL": 1, "Bloody diarrhea": 2, "Excessive Urination and Thirst": 3, "Swollen testicles": 4, "Rose spots": 5}
    critical_feelings_mapping = {"Slow heart rate and pulse": 0, "NIL": 1, "Severe pelvic pain": 3, "Seizures": 4, "Disorientation": 5, "Confusion": 6, "Cough and breathing difficulties": 7}
    critical_mapping = {"Critical": 0, "Not Critical": 1}

    discharge = st.sidebar.selectbox('Discharge', options=list(discharge_mapping.keys()))
    feelings_and_urge = st.sidebar.selectbox('Feelings and Urge', options=list(feelings_and_urge_mapping.keys()))
    pain_and_infection = st.sidebar.selectbox('Pain and Infection', options=list(pain_and_infection_mapping.keys()))
    physical_condition = st.sidebar.selectbox('Physical Condition', options=list(physical_condition_mapping.keys()))
    critical_feelings = st.sidebar.selectbox('Critical Feelings', options=list(critical_feelings_mapping.keys()))
    critical = st.sidebar.selectbox('Critical', options=list(critical_mapping.keys()))

    data = {'Discharge': discharge_mapping[discharge],
            'Feelings and Urge': feelings_and_urge_mapping[feelings_and_urge],
            'Pain and Infection': pain_and_infection_mapping[pain_and_infection],
            'Physical Conditions': physical_condition_mapping[physical_condition],
            'Critical Feelings': critical_feelings_mapping[critical_feelings],
            'Critical': critical_mapping[critical]}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Display user input parameters
st.subheader('User Input parameters')
st.write(df)

# Predict and display the output
prediction = model.predict(df)
st.subheader('Prediction')
st.write(prediction)

