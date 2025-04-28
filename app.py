# insurance_app.py

import numpy as np
import pickle
import streamlit as st

# Load the trained model
try:
    loaded_model = pickle.load(open('insurance_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: insurance_model.pkl not found. Please train the model first.")
    st.stop()

# Prediction function
def medical_insurance_cost_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_data_as_numpy_array)
    return round(prediction[0], 2)

# Navbar setup
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

# Custom CSS for style and spacing
st.markdown("""
    <style>
        .navbar {
            position: fixed; top: 0; width: 100%; background-color: #004d99;
            padding: 10px; z-index: 1000; display: flex; justify-content: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .navbar button {
            background: none; border: none; color: white; font-size: 18px;
            font-weight: bold; padding: 10px 20px; cursor: pointer;
        }
        .navbar button:hover {
            background-color: #007acc; border-radius: 5px;
        }
        .block-container { padding-top: 80px !important; }
        body { background-color: #f2f6fc !important; }
        .title { text-align: center; font-size: 36px; font-weight: bold; color: #003366; padding: 20px; }
        .prediction-box {
            font-size: 24px; font-weight: bold; color: #006600;
            background-color: #e6ffe6; padding: 15px; border-radius: 10px;
            text-align: center; margin-top: 20px;
        }
        .input-box {
            background-color: #e8f0fe; padding: 15px; border-radius: 10px;
            margin-bottom: 10px;
        }
        .input-box label {
            font-weight: bold; color: #003366;
        }
        .input-box input, .input-box select, .input-box div {
            width: 100%; margin-top: 5px; padding: 8px;
            border: 1px solid #cdd2d8; border-radius: 5px;
        }
        .input-box input[type="number"], .input-box input[type="range"] {
            width: calc(100% - 18px);
        }
        .input-box input[type="radio"] {
            width: auto; margin-right: 5px;
        }
        .predict-button {
            background-color: #007acc; color: white; padding: 10px 20px;
            border: none; border-radius: 5px; cursor: pointer;
            font-weight: bold; font-size: 16px;
        }
        .predict-button:hover {
            background-color: #005f99;
        }
        .error-message {
            color: red; font-weight: bold; margin-top: 10px;
        }
        .input-columns {
          display: flex;
          gap: 20px;
        }
        .input-columns > div {
          flex: 1;
        }
    </style>
""", unsafe_allow_html=True)

# Navbar
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📊 Predict Cost"):
        st.session_state.page = "Predict"
with col3:
    if st.button("ℹ️ About"):
        st.session_state.page = "About"
with col4:
    if st.button("📞 Contact"):
        st.session_state.page = "Contact"

# Main App Logic
def main():
    if st.session_state.page == 'Home':
        st.markdown('<h1 class="title">Medical Insurance Cost Prediction</h1>', unsafe_allow_html=True)
        st.write("Welcome! Use the navigation bar above to explore the app.")

    elif st.session_state.page == 'Predict':
        st.markdown('<h1 class="title">📊 Predict Medical Insurance Cost</h1>', unsafe_allow_html=True)

        st.markdown("### Please fill in the following details:")

        st.markdown('<div class="input-columns">', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input('Age', min_value=18, max_value=100, value=30)
            st.markdown('<div class="input-box"><label>Age</label>', unsafe_allow_html=True)
            bmi = st.number_input('Body Mass Index', min_value=10.0, max_value=50.0, value=25.0)
            st.markdown('<div class="input-box"><label>Body Mass Index</label>', unsafe_allow_html=True)
            children = st.number_input('Number of Children', min_value=0, max_value=10, value=1)
            st.markdown('<div class="input-box"><label>Number of Children</label>', unsafe_allow_html=True)
            exercise_frequency = st.slider('Exercise Frequency (per week)', 0, 7, 3)
            st.markdown('<div class="input-box"><label>Exercise Frequency (per week)</label>', unsafe_allow_html=True)
            sleep_hours = st.slider("Sleep Hours (per night)", 3, 12, 7)
            st.markdown('<div class="input-box"><label>Sleep Hours (per night)</label>', unsafe_allow_html=True)
            stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
            st.markdown('<div class="input-box"><label>Stress Level (1-10)</label>', unsafe_allow_html=True)

        with col2:
            sex = st.radio('Sex', ['Female', 'Male'])
            st.markdown('<div class="input-box"><label>Sex</label>', unsafe_allow_html=True)
            smoker = st.radio('Smoker', ['No', 'Yes'])
            st.markdown('<div class="input-box"><label>Smoker</label>', unsafe_allow_html=True)
            alcohol_consumption = st.radio('Alcohol Consumption', ['No', 'Yes'])
            st.markdown('<div class="input-box"><label>Alcohol Consumption</label>', unsafe_allow_html=True)
            region = st.selectbox('Region', ['NorthEast', 'NorthWest', 'SouthEast', 'SouthWest'])
            st.markdown('<div class="input-box"><label>Region</label>', unsafe_allow_html=True)
            diet_type = st.selectbox('Diet Type', ['Balanced', 'High-Protein', 'Fast-Food', 'Vegan'])
            st.markdown('<div class="input-box"><label>Diet Type</label>', unsafe_allow_html=True)

            # Pre-existing Conditions logic
            pre_existing_conditions_options = ['Diabetes', 'Hypertension', 'Heart Disease']
            pre_existing_conditions = []

            none_selected = st.checkbox("No Pre-existing Conditions")

            if not none_selected:
                pre_existing_conditions = st.multiselect(
                    'Pre-existing Conditions',
                    pre_existing_conditions_options
                )
            else:
                pre_existing_conditions = ['None']


            st.markdown('<div class="input-box"><label>Pre-existing Conditions</label>', unsafe_allow_html=True)

            if 'None' in pre_existing_conditions and len(pre_existing_conditions) > 1:
                st.warning("If 'None' is selected, no other conditions can be selected.")
                pre_existing_conditions = ['None']

            water_intake = st.slider("Water Intake (liters/day)", 0.5, 5.0, 2.0, step=0.1)
            st.markdown('<div class="input-box"><label>Water Intake (liters/day)</label>', unsafe_allow_html=True)
            screen_time = st.slider("Screen Time (hours/day)", 1.0, 16.0, 6.0, step=0.5)
            st.markdown('<div class="input-box"><label>Screen Time (hours/day)</label>', unsafe_allow_html=True)
            health_checkups = st.slider("Health Checkups per Year", 0, 12, 2)
            st.markdown('<div class="input-box"><label>Health Checkups per Year</label>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # Encode categorical values
        sex_encoded = 1 if sex == 'Male' else 0
        smoker_encoded = 1 if smoker == 'Yes' else 0
        region_encoded = ['NorthEast', 'NorthWest', 'SouthEast', 'SouthWest'].index(region)
        diet_encoded = ['Balanced', 'High-Protein', 'Fast-Food', 'Vegan'].index(diet_type)
        alcohol_encoded = 1 if alcohol_consumption == 'Yes' else 0

        condition_flags = {
            "Diabetes": 0,
            "Hypertension": 0,
            "Heart Disease": 0
        }
        for condition in pre_existing_conditions:
            if condition in condition_flags:
                condition_flags[condition] = 1

        input_features = [
            age, sex_encoded, bmi, children, smoker_encoded,
            region_encoded, exercise_frequency, diet_encoded, alcohol_encoded,
            condition_flags["Diabetes"], condition_flags["Heart Disease"], condition_flags["Hypertension"],
            sleep_hours, stress_level, water_intake, screen_time, health_checkups
        ]

        st.markdown("---")
        center_col = st.columns(3)[1]
        with center_col:
            if st.button('Predict Medical Insurance Cost', key="predict_button", on_click=None, use_container_width=True, type="primary"):
                if age > 65:
                    st.markdown('<p class="error-message">Life insurance is not provided for individuals above 65.</p>', unsafe_allow_html=True)
                else:
                    prediction = medical_insurance_cost_prediction(input_features)
                    st.markdown(f'<div class="prediction-box">Predicted Insurance Cost: ₹{prediction}</div>', unsafe_allow_html=True)

    elif st.session_state.page == 'About':
        st.markdown('<h1 class="title">ℹ️ About This App</h1>', unsafe_allow_html=True)
        st.write("""
        This AI-powered tool estimates medical insurance costs using your lifestyle and health data.

        **Includes:**
        - Sleep, stress, hydration, screen habits
        - Multi-condition support
        - Smarter prediction model

        Great for individuals, insurers, and health analysts!
        """)

    elif st.session_state.page == 'Contact':
        st.markdown('<h1 class="title">📞 Contact Us</h1>', unsafe_allow_html=True)
        st.write("""
        **Questions? Reach out:** 
        - 📧 Email: support@insurancepredictor.com 
        - 📞 Phone: +91 9876543210 
        - 🌐 Website https://medicalinsurance.com 
        """)

if __name__ == '__main__':
    main()