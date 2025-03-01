import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pickle

# Load data (using the file directly, if placed in the same directory)
try:
    data = pd.read_csv("./data/pima-data.csv")
except FileNotFoundError:
    st.error("Error: pima-data.csv not found.  Make sure it's in the same directory as your app.")
    st.stop()

# Data cleaning and preprocessing (handle zeros)
for col in data.columns:
    if col != 'diabetes':
        data[col] = data[col].replace(0, data[col].mean())

# Separate features (X) and target (y)
X = data.drop('diabetes', axis=1)
y = data['diabetes']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training (Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Model Evaluation (optional, but good practice)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Streamlit App
st.title("Diabetes Prediction App")
st.write("Enter patient data to predict diabetes.")


# Input fields
num_preg = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0)
glucose_conc = st.number_input("Glucose Concentration", min_value=0, max_value=200, value=100)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=0, max_value=150, value=70)
thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin (mu U/ml)", min_value=0, max_value=1000, value=50)
bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
diab_pred = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
skin = st.number_input("Skin", min_value=0.0, max_value=5.0, value=1.0)

# Prediction
if st.button("Predict Diabetes"):
    # Prepare input data for prediction
    input_data = np.array([[num_preg, glucose_conc, diastolic_bp, thickness, insulin, bmi, diab_pred, age, skin]])  # Ensure correct order
    input_data_scaled = scaler.transform(input_data)  # Scale the input data

    # Make prediction
    prediction = model.predict(input_data_scaled)[0]

    # Display result
    if prediction == 1:
        st.write("The model predicts a positive diagnosis for diabetes.")
    else:
        st.write("The model predicts a negative diagnosis for diabetes.")

    # Optional: Display accuracy and classification report (from the test set)
    st.subheader("Model Performance:")
    st.write(f"Accuracy on Test Set: {accuracy:.2f}")
    st.write("Classification Report:\n", report)
