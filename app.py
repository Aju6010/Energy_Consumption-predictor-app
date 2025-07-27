
import pandas as pd
import streamlit as st

import joblib

# Load trained model
model = joblib.load('linear_regression_model.pkl')

# Load scaler
scaler = joblib.load('scaler.pkl')


# Define features
feature_columns = ['Temperature', 'Humidity', 'SquareFootage', 'Occupancy',
                   'RenewableEnergy', 'HVACUsage_On', 'LightingUsage_On',
                   'Holiday_Yes', 'HVAC_Temp_Interaction', 'PeoplePerSqFt']

# UI
st.title("🏠 Energy Consumption Predictor")

temp = st.number_input("🌡️ Enter Temperature (°C)", min_value=-20.0, max_value=50.0, step=0.1)
humidity = st.number_input("💧 Enter Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
sqft = st.number_input("🏢 Enter Square Footage", min_value=100.0, max_value=10000.0, step=1.0)
occupancy = st.number_input("👥 Enter Occupancy (people)", min_value=0, step=1)
renewable = st.number_input("🔋 Enter Renewable Energy Usage", min_value=0.0, step=1.0)
hvac_on = st.radio("❄️ Is HVAC ON?", [0, 1])
light_on = st.radio("💡 Is Lighting ON?", [0, 1])
day = st.selectbox("📅 Select Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

# Compute features
holiday = 1 if day.lower() in ['saturday', 'sunday'] else 0
hvac_temp_interaction = temp * hvac_on
people_per_sqft = occupancy / sqft if sqft > 0 else 0

input_dict = {
    'Temperature': [temp],
    'Humidity': [humidity],
    'SquareFootage': [sqft],
    'Occupancy': [occupancy],
    'RenewableEnergy': [renewable],
    'HVACUsage_On': [hvac_on],
    'LightingUsage_On': [light_on],
    'Holiday_Yes': [holiday],
    'HVAC_Temp_Interaction': [hvac_temp_interaction],
    'PeoplePerSqFt': [people_per_sqft]
}

input_df = pd.DataFrame(input_dict)

# Scale the input
input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=feature_columns)

# Predict
if st.button("🔍 Predict Energy Consumption"):
    prediction = model.predict(input_df_scaled)[0]
    st.success(f"⚡ Predicted Energy Consumption: {prediction:.2f} units")
