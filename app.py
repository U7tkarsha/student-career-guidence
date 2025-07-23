import streamlit as st
import joblib
import numpy as np

# Load saved model and encoders
model = joblib.load("model.pkl")
le_prof = joblib.load("le_prof.pkl")
le_stream = joblib.load("le_stream.pkl")
le_family = joblib.load("le_family.pkl")

# Streamlit UI
st.set_page_config(page_title="Career Predictor", layout="centered")
st.title("🎓 Student Career Prediction UI")
st.markdown("This app predicts a student's *career group* after Class 10 based on their interests and background.")

# Input UI
grade_10_marks = st.slider("Grade 10 Marks (%)", 0, 100, 75)
interest_math = st.slider("Interest in Math (0-10)", 0, 10, 5)
interest_bio = st.slider("Interest in Biology (0-10)", 0, 10, 5)
interest_econ = st.slider("Interest in Economics (0-10)", 0, 10, 5)
coding = st.slider("Interest in Coding (0-10)", 0, 10, 5)
drawing = st.slider("Interest in Drawing (0-10)", 0, 10, 5)
sports = st.slider("Interest in Sports (0-10)", 0, 10, 5)

stream = st.selectbox("Chosen Stream", le_stream.classes_)
family_background = st.selectbox("Family Background", le_family.classes_)

# Prediction
if st.button("Predict Career Group"):
    input_vector = np.array([[
        grade_10_marks, interest_math, interest_bio, interest_econ,
        coding, drawing, sports,
        le_stream.transform([stream])[0],
        le_family.transform([family_background])[0]
    ]])

    prediction = model.predict(input_vector)[0]
    profession_group = le_prof.inverse_transform([prediction])[0]

    st.success(f"Predicted Career Group: *{profession_group}* 🎯")