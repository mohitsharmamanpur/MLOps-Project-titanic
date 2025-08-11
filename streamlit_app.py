# streamlit_app.py
import streamlit as st
import joblib
import numpy as np

# Load pipeline (model + preprocessing)
model = joblib.load("titanic_pipeline.pkl")

st.title("🚢 Titanic Survival Prediction")

# Collect user input
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=50.0)

# Convert input to DataFrame
input_data = [[pclass, sex, age, sibsp, parch, fare]]

# Predict
if st.button("Predict Survival"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("✅ Survived")
    else:
        st.error("❌ Did Not Survive")
