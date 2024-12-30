import streamlit as st
import joblib
import numpy as np

# Load the saved model and scaler
model_path = 'svr_model.pkl'
scaler_path = 'scaler.pkl'

svr_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Set up the Streamlit app
st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

# Add custom CSS for background color and output font styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #8B4513;
    }
    .output-text {
        font-weight: bold;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Employee Salary Prediction WebApp")

# Form for user inputs
with st.form("prediction_form"):
    st.header("Enter the data for prediction")

    # User input fields
    department = st.selectbox(
        "Department:", ['IT', 'Finance', 'Engineering', 'Customer Support', 'Marketing', 'HR', 'Operations']
    )

    gender = st.selectbox(
        "Gender:", ['Male', 'Female', 'Other']
    )

    job_title = st.selectbox(
        "Job Title:", ['Analyst', 'Manager', 'Engineer', 'Developer', 'Technician', 'Specialist', 'Consultant']
    )

    performance_score = st.number_input("Performance Score:", min_value=0.0, step=0.1)
    overtime_hours = st.number_input("Overtime Hours:", min_value=0.0, step=0.1)
    team_size = st.number_input("Team Size:", min_value=1, step=1)

    resigned = st.selectbox("Resigned:", ['TRUE', 'FALSE'])

    # Submit button
    submitted = st.form_submit_button("Predict")

# Prediction logic
if submitted:
    # Convert categorical inputs to numerical values
    department_dict = {'IT': 0, 'Finance': 1, 'Engineering': 2, 'Customer Support': 3, 'Marketing': 4, 'HR': 5, 'Operations': 6}
    gender_dict = {'Male': 0, 'Female': 1, 'Other': 2}
    job_title_dict = {'Analyst': 0, 'Manager': 1, 'Engineer': 2, 'Developer': 3, 'Technician': 4, 'Specialist': 5, 'Consultant': 6}

    resigned_value = 1 if resigned == 'TRUE' else 0

    # Prepare the input data in the order the model expects (numerical features)
    input_data = np.array([
        department_dict[department],
        gender_dict[gender],
        job_title_dict[job_title],
        performance_score,
        overtime_hours,
        team_size,
        resigned_value
    ]).reshape(1, -1)

    # Scale the input data using the scaler
    scaled_input_data = scaler.transform(input_data)

    # Make the prediction
    prediction = svr_model.predict(scaled_input_data)[0]

    # Add 3000 to the prediction
    prediction = prediction * 1000 + 3000

    # Display the prediction
    st.markdown(f"<p class='output-text'>Prediction: ₦{prediction:,.2f}</p>", unsafe_allow_html=True)
