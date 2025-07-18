import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title = "Employee Salary Classification", page_icon=" ", layout = 'centered')
st.title(" Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50k or <=50k based on input features.")

st.sidebar.header("Input Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", [
    "Bachelors", "Masters", "PhD", "HS-grad", "Assoc", "Some-college"
])
occupation = st.sidebar.selectbox("Job Role", [
    "Tech-support", "Craft-repair", "Other-service", "Sales",
    "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct",
    "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv",
    "Protective-serv", "Armed-Forces"
])
hours_per_week = st.sidebar.slider("Hours per week", 1, 80, 40)
experience = st.sidebar.slider("Years of Experience", 0, 40, 5)

# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education': [education],
    'occupation': [occupation],
    'hours-per-week': [hours_per_week],
    'experience': [experience]
})

input_df = pd.get_dummies(input_df)

for col in feature_names:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[feature_names]

st.write("### 🔎 Input Data")
st.write(input_df.loc[:, (input_df != 0).any(axis=0)])
# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    batch_encoded = pd.get_dummies(batch_data)

    # Add missing columns
    for col in feature_names:
        if col not in batch_encoded.columns:
            batch_encoded[col] = 0
    batch_encoded = batch_encoded[feature_names]

    preds = model.predict(batch_encoded)
    batch_data['PredictedClass'] = preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())

    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
