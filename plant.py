# %%
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Set page title
st.title("🌱 Plant Growth Prediction App")

# Load the trained logistic regression model
try:
    with open('logreg_model.pkl', 'rb') as file:
        logreg = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'logreg_model.pkl' not found. Please ensure it is in the same directory.")
    st.stop()

# Define the label encoder for decoding predictions
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(['Yes', 'No'])  # Adjust if your training used different labels

# Sidebar for user inputs
st.sidebar.header("Enter Plant Details")

# Numerical features
Sunlight_Hours = st.sidebar.slider("Sunlight Hours", min_value=0, max_value=12, value=6)
Temperature = st.sidebar.slider("Temperature (°C)", min_value=0, max_value=50, value=25)
Humidity = st.sidebar.slider("Humidity (1-100)", min_value=1, max_value=100, value=50)

# Categorical features
Soil_Type = st.sidebar.selectbox("Soil Type", options=["Clay", "Sandy", "Loam"])
Water_Frequency = st.sidebar.selectbox("Water Frequency", options=["Daily", "Weekly", "Bi-Weekly"])
Fertilizer_Type = st.sidebar.selectbox("Fertilizer Type", options=["organic", "chemical", "none"])


# Function to preprocess input data
def preprocess_input(sunlight_hours, temperature, humidity, soil_type, water_frequency, fertilizer_type):
    # Create DataFrame with numerical features (names must match training!)
    data = {
        'Sunlight_Hours': sunlight_hours,
        'Temperature': temperature,
        'Humidity': humidity
    }
    df = pd.DataFrame([data])

    # One-hot encoding with exact training column names
    soil_options = ['clay', 'sandy', 'loam']
    for soil in soil_options:
        df[f'Soil_Type_{soil}'] = 1 if soil_type.lower() == soil else 0

    water_options = ['daily', 'weekly', 'bi-weekly']
    for water in water_options:
        df[f'Water_Frequency_{water}'] = 1 if water_frequency.lower() == water else 0

    fertilizer_options = ['organic', 'chemical', 'none']
    for fertilizer in fertilizer_options:
        df[f'Fertilizer_Type_{fertilizer}'] = 1 if fertilizer_type.lower() == fertilizer else 0

    # Ensure all expected columns are present in the correct order
    expected_columns = [
        'Sunlight_Hours', 'Temperature', 'Humidity',
        'Soil_Type_clay', 'Soil_Type_sandy', 'Soil_Type_loam',
        'Water_Frequency_daily', 'Water_Frequency_weekly', 'Water_Frequency_bi-weekly',
        'Fertilizer_Type_organic', 'Fertilizer_Type_chemical', 'Fertilizer_Type_none'
    ]
    df = df.reindex(columns=expected_columns, fill_value=0)

    return df


# Button to make prediction
if st.sidebar.button("Predict"):
    # Preprocess the input (lowercase categorical values before passing)
    input_df = preprocess_input(
        Sunlight_Hours,
        Temperature,
        Humidity,
        Soil_Type.lower(),
        Water_Frequency.lower(),
        Fertilizer_Type.lower()
    )

    # Make prediction
    try:
        prediction = logreg.predict(input_df)
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        # Display result
        st.subheader("Prediction Result")
        st.write(f"🌿 The predicted plant growth is: **{predicted_label}**")

        if predicted_label == "No":
            st.write("🚫 No significant plant growth predicted.")
        else:
            st.write("✅ The plant is likely to grow well!")

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")

# Display instructions
st.write("""
### Instructions
1. Use the sidebar to enter the plant details.
2. Adjust the sliders for numerical features (Sunlight Hours, Temperature, Humidity).
3. Select appropriate options for Soil Type, Water Frequency, and Fertilizer Type.
4. Click the **Predict** button to see the predicted plant growth.
""")
