import streamlit as st
import numpy as np
from preprocess import load_data
from model import train_and_evaluate
from xai import explain_model

st.set_page_config(page_title="Student Depression Predictor", layout="wide")
st.title("🧠 Student Depression Risk Predictor (Mini Project)")

@st.cache_resource
def get_model():
    X, y, labels, _ = load_data()
    model, X_test, _, _, _ = train_and_evaluate(X, y)
    explain_model(model, X_test, labels)
    return model, labels

model, feature_labels = get_model()

st.sidebar.header("Enter Student Details")

academic = st.sidebar.slider("Academic Pressure (1-5)", 1, 5, 3)
financial = st.sidebar.slider("Financial Stress (1-5)", 1, 5, 2)
satisfaction = st.sidebar.slider("Study Satisfaction (1-5)", 1, 5, 3)
sleep = st.sidebar.selectbox("Sleep Duration", ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"])
diet = st.sidebar.selectbox("Dietary Habits", ["Unhealthy", "Moderate", "Healthy"])
hours = st.sidebar.slider("Study Hours/Day", 0, 12, 6)
cgpa = st.sidebar.slider("CGPA", 0.0, 10.0, 7.0, 0.1)
age = st.sidebar.slider("Age", 18, 40, 22)
degree = st.sidebar.selectbox("Degree Level", ["Class 12", "Undergraduate", "Postgraduate"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
suicidal = st.sidebar.selectbox("Suicidal Thoughts?", ["No", "Yes"])
family = st.sidebar.selectbox("Family History?", ["No", "Yes"])

sleep_val = {"Less than 5 hours":1,"5-6 hours":2,"7-8 hours":3,"More than 8 hours":4}[sleep]
diet_val = {"Unhealthy":0,"Moderate":1,"Healthy":2}[diet]
deg_val = {"Class 12":0,"Undergraduate":1,"Postgraduate":2}[degree]
gender_val = 0 if gender == "Male" else 1

x_input = np.array([academic, financial, satisfaction, sleep_val,
                    1 if suicidal=="Yes" else 0,
                    1 if family=="Yes" else 0,
                    hours, cgpa, age, diet_val, deg_val, gender_val])

prob = model.predict_proba(x_input.reshape(1,-1))[0,1]

st.subheader(f"Depression Risk: **{prob:.1%}**")
if prob > 0.6:
    st.error("🔴 HIGH RISK")
elif prob > 0.4:
    st.warning("🟡 MODERATE RISK")
else:
    st.success("🟢 LOW RISK")

st.caption("Mini Project - Random Forest + SHAP")