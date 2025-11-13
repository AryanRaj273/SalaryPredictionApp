import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Load saved model and preprocessing objects
model = joblib.load("salary_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")
feature_columns = joblib.load("feature_columns.pkl")  # ✅ new addition

st.set_page_config(page_title="Salary Prediction App", layout="wide")
st.title("💼 Salary Prediction App")
st.markdown("### Predict employee salary using ML & Ensemble models")

# Sidebar input section
st.sidebar.header("Employee Details")
user_input = {}

# Handle categorical columns
categorical_cols = list(label_encoders.keys())
for feature, encoder in label_encoders.items():
    options = list(encoder.classes_)
    user_input[feature] = st.sidebar.selectbox(feature, options)

# Handle numeric columns
# Extract which columns are numeric by exclusion
numeric_features = [f for f in feature_columns if f not in categorical_cols]

for f in numeric_features:
    user_input[f] = st.sidebar.number_input(f, min_value=0.0, step=0.1)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical features
for col, encoder in label_encoders.items():
    input_df[col] = encoder.transform(input_df[col])

# Reorder columns to match training
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# Apply scaler if applicable
try:
    input_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)
except Exception as e:
    st.warning(f"Scaler failed ({e}), using raw input.")
    input_scaled = input_df

# Predict Salary
if st.button("Predict Salary"):
    try:
        prediction = model.predict(input_scaled)[0]
        st.success(f"💰 Predicted Salary: ₹{prediction:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
