import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("medical_diagnosis_model.pkl")
feature_encoders = joblib.load("feature_encoders.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Page title
st.title("🩺 Medical Diagnosis Prediction")
st.write("Fill in the patient details to predict whether the outcome is Positive or Negative.")

# User input form
def user_input_features():
    disease = st.selectbox("Disease", feature_encoders['Disease'].classes_)
    fever = st.selectbox("Fever", feature_encoders['Fever'].classes_)
    cough = st.selectbox("Cough", feature_encoders['Cough'].classes_)
    fatigue = st.selectbox("Fatigue", feature_encoders['Fatigue'].classes_)
    difficulty_breathing = st.selectbox("Difficulty Breathing", feature_encoders['Difficulty Breathing'].classes_)
    age = st.slider("Age", 0, 100, 25)
    gender = st.selectbox("Gender", feature_encoders['Gender'].classes_)
    bp = st.selectbox("Blood Pressure", feature_encoders['Blood Pressure'].classes_)
    cholesterol = st.selectbox("Cholesterol Level", feature_encoders['Cholesterol Level'].classes_)

    # Collect input into DataFrame
    data = {
        "Disease": disease,
        "Fever": fever,
        "Cough": cough,
        "Fatigue": fatigue,
        "Difficulty Breathing": difficulty_breathing,
        "Age": age,
        "Gender": gender,
        "Blood Pressure": bp,
        "Cholesterol Level": cholesterol
    }
    return pd.DataFrame([data])

# Collect user input
input_df = user_input_features()

# Encode input using the same encoders
for col in input_df.columns:
    if col in feature_encoders:
        le = feature_encoders[col]
        input_df[col] = le.transform(input_df[col])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_df)
    outcome = target_encoder.inverse_transform(prediction)[0]
    st.subheader(f"🧾 Predicted Outcome: **{outcome}**")
