import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------
# Load model & preprocessor
# -----------------------
model = joblib.load("detection_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

st.set_page_config(page_title="Insurance Fraud Detection", layout="centered")

st.title("🚨 Insurance Claim Fraud Detection")
st.write("Enter claim details to predict fraud probability.")

# -----------------------
# Input form
# -----------------------
with st.form("fraud_form"):

    st.subheader("Numerical Features")

    Age = st.number_input("Age", 18, 100, 35)
    WeekOfMonth = st.number_input("Week Of Month", 1, 5, 2)
    WeekOfMonthClaimed = st.number_input("Week Of Month Claimed", 1, 5, 2)
    Deductible = st.number_input("Deductible", 0, 10000, 500)
    DriverRating = st.slider("Driver Rating", 1, 5, 3)
    Year = st.number_input("Vehicle Year", 1990, 2025, 2018)
    Claim_Lag = st.number_input("Claim Lag (days)", 0, 365, 10)

    DayOfWeek_sin = st.number_input("DayOfWeek sin", value=0.0)
    DayOfWeek_cos = st.number_input("DayOfWeek cos", value=1.0)
    DayOfWeekClaimed_sin = st.number_input("Claimed Day sin", value=0.0)
    DayOfWeekClaimed_cos = st.number_input("Claimed Day cos", value=1.0)

    st.subheader("Ordinal Features")

    VehiclePrice = st.selectbox("Vehicle Price", [
        'less than 20000','20000 to 29000','30000 to 39000','40000 to 59000',
        '60000 to 69000','more than 69000'
    ])

    PastNumberOfClaims = st.selectbox("Past Claims", ['none','1','2 to 4','more than 4'])
    AgeOfVehicle = st.selectbox("Age Of Vehicle", ['new','2 years','3 years','4 years','5 years','6 years','7 years','more than 7'])
    AgeOfPolicyHolder = st.selectbox("Age Of Policy Holder", ['16 to 17','18 to 20','21 to 25','26 to 30','31 to 35','36 to 40','41 to 50','51 to 65','over 65'])
    NumberOfSuppliments = st.selectbox("Supplements", ['none','1 to 2','3 to 5','more than 5'])
    AddressChange_Claim = st.selectbox("Address Change", ['no change','under 6 months','1 year','2 to 3 years','4 to 8 years'])
    NumberOfCars = st.selectbox("Number Of Cars", ['1 vehicle','2 vehicles','3 to 4','5 to 8','more than 8'])

    st.subheader("Categorical Features")

    Make = st.selectbox("Make", ['Honda','Toyota','BMW','Ford','Mazda','Nissan','Chevrolet','Other'])
    AccidentArea = st.selectbox("Accident Area", ['Urban','Rural'])
    Sex = st.selectbox("Sex", ['Male','Female'])
    MaritalStatus = st.selectbox("Marital Status", ['Single','Married','Divorced','Widowed'])
    Fault = st.selectbox("Fault", ['Policy Holder','Third Party'])
    PolicyType = st.selectbox("Policy Type", ['Sedan - Liability','Sedan - Collision','Sedan - All Perils','Utility - Liability','Utility - Collision','Utility - All Perils'])
    VehicleCategory = st.selectbox("Vehicle Category", ['Sedan','Utility','Sports'])
    PoliceReportFiled = st.selectbox("Police Report Filed", ['Yes','No'])
    WitnessPresent = st.selectbox("Witness Present", ['Yes','No'])
    AgentType = st.selectbox("Agent Type", ['Internal','External'])
    BasePolicy = st.selectbox("Base Policy", ['Liability','Collision','All Perils'])

    submitted = st.form_submit_button("Predict Fraud")

# -----------------------
# Prediction
# -----------------------
if submitted:

    input_dict = {
        'Age': Age,
        'WeekOfMonth': WeekOfMonth,
        'WeekOfMonthClaimed': WeekOfMonthClaimed,
        'Deductible': Deductible,
        'DriverRating': DriverRating,
        'Year': Year,
        'Claim_Lag': Claim_Lag,
        'DayOfWeek_sin': DayOfWeek_sin,
        'DayOfWeek_cos': DayOfWeek_cos,
        'DayOfWeekClaimed_sin': DayOfWeekClaimed_sin,
        'DayOfWeekClaimed_cos': DayOfWeekClaimed_cos,

        'VehiclePrice': VehiclePrice,
        'PastNumberOfClaims': PastNumberOfClaims,
        'AgeOfVehicle': AgeOfVehicle,
        'AgeOfPolicyHolder': AgeOfPolicyHolder,
        'NumberOfSuppliments': NumberOfSuppliments,
        'AddressChange_Claim': AddressChange_Claim,
        'NumberOfCars': NumberOfCars,

        'Make': Make,
        'AccidentArea': AccidentArea,
        'Sex': Sex,
        'MaritalStatus': MaritalStatus,
        'Fault': Fault,
        'PolicyType': PolicyType,
        'VehicleCategory': VehicleCategory,
        'PoliceReportFiled': PoliceReportFiled,
        'WitnessPresent': WitnessPresent,
        'AgentType': AgentType,
        'BasePolicy': BasePolicy
    }

    input_df = pd.DataFrame([input_dict])

    X_processed = preprocessor.transform(input_df)

    prob = model.predict_proba(X_processed)[0][1]

    prediction = "🚨 FRAUD" if prob >= 0.30 else "✅ NOT FRAUD"

    st.markdown("---")
    st.subheader("Prediction Result")

    st.metric("Fraud Probability", f"{prob:.2%}")

    if prediction.startswith("🚨"):
        st.error(prediction)
    else:
        st.success(prediction)
