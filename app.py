import streamlit as st
import joblib
import pandas as pd

st.title("Customer Churn Prediction")
st.image("https://www.bing.com/images/search?q=Customer+Churn+Service+Background&FORM=IRIBEP", width=300)

#  Load model & scaler
model = joblib.load('best_rf_model.pkl')

scaler = joblib.load('scaler.pkl')

#  Load encoders (ONLY categorical)
encoders = {
    'gender': joblib.load('le_gender.pkl'),
    'PhoneService': joblib.load('le_phone_service.pkl'),
    'MultipleLines': joblib.load('le_multiple_lines.pkl'),
    'InternetService': joblib.load('le_internet_service.pkl'),
    'OnlineSecurity': joblib.load('le_online_security.pkl'),
    'OnlineBackup': joblib.load('le_onlinebackup.pkl'),
    'DeviceProtection': joblib.load('le_deviceprotection.pkl'),
    'TechSupport': joblib.load('le_techsupport.pkl'),
    'StreamingTV': joblib.load('le_streaming_tv.pkl'),
    'StreamingMovies': joblib.load('le_streaming_movies.pkl'),
    'Contract': joblib.load('le_contract.pkl'),
    'PaymentMethod': joblib.load('le_payment_method.pkl'),
    'Partner': joblib.load('le_partner.pkl'),
    'Dependents': joblib.load('le_dependents.pkl'),
    'PaperlessBilling': joblib.load('le_paperlessbilling.pkl')
}

# 🔹 User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure")
phone = st.selectbox("Phone Service", ["Yes", "No"])
multiple = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly = st.number_input("Monthly Charges")
total = st.number_input("Total Charges")

# 🔹 Predict Button
if st.button("Predict"):

    input_data = {
        'gender': gender,
        'SeniorCitizen': senior,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone,
        'MultipleLines': multiple,
        'InternetService': internet,
        'OnlineSecurity': security,
        'OnlineBackup': backup,
        'DeviceProtection': device,
        'TechSupport': tech,
        'StreamingTV': tv,
        'StreamingMovies': movies,
        'Contract': contract,
        'PaperlessBilling': paperless,
        'PaymentMethod': payment,
        'MonthlyCharges': monthly,
        'TotalCharges': total
    }

    # Convert to DataFrame
    df_input = pd.DataFrame([input_data])

    # Encode categorical columns
    for col in encoders:
        df_input[col] = encoders[col].transform(df_input[col])

    # Scale
    df_scaled = scaler.transform(df_input)

    # Predict
    prediction = model.predict(df_scaled)[0]

    # Output
    if prediction == 1:
        st.error("Customer will churn ")
    else:
        st.success("Customer will stay ")


