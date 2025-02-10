import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Prepare the data
X = data.drop(columns=['Outcome','Pregnancies'], axis=1)
Y = data['Outcome']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state['page'] = 'home'

# Define the form page function
def form_page():
    st.title("Diabetes Prediction Form")
    st.subheader("Please fill out the following fields:")

    # Input fields with "Enter data here" as placeholders
    val2 = st.number_input("Glucose", min_value=0.0, step=0.1, placeholder="Enter data here")  # Float
    if val2:   
        val3 = st.number_input("Blood Pressure", min_value=0.0, step=0.1, placeholder="Enter data here")  # Float
        if val3:
            val4 = st.number_input("Skin Thickness", min_value=0.0, step=0.1, placeholder="Enter data here")  # Float
            if val4:
                val5 = st.number_input("Insulin", min_value=0.0, step=0.1, placeholder="Enter data here")  # Float
                if val5:
                    val6 = st.number_input("BMI", min_value=0.0, step=0.1, placeholder="Enter data here")  # Float
                    if val6:
                        val7 = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01, placeholder="Enter data here")  # Float
                        if val7:
                            val8 = st.number_input("Age", min_value=0, step=1, placeholder="Enter data here")  # Integer


 # Predict button
    if val8:
        check = st.text("Click the button below to check if you have diabetes")
    if check:
        st.button("Run Check")
    if st.button("Run Check"):
        # Store user input in session state
        st.session_state["user_input"] = [val2, val3, val4, val5, val6, val7, val8]
        st.session_state["page"] = "result"

# Define the result page function
def result_page():
    st.title("Prediction Result")
    
    # Retrieve user input
    user_input = st.session_state.get("user_input", None)

    if user_input:
        # Make prediction
        prediction = model.predict([user_input])
        result_text = "You have Diabetes, Please consult your physician" if prediction[0] == 1 else "Great news, you are not diabetic"
        st.subheader(result_text)

    # Button to go back to home
    if st.button("Go back to Home"):
        st.session_state["page"] = "home"

# Display the correct page based on session state
if st.session_state["page"] == "home":
    st.title("Diabetes Prediction App")
    st.text("This app predicts whether or not you have diabetes with an accuracy of 83%.")
    st.text("Click the button below to begin.")
    
    if st.button("Let's get to it"):
        st.session_state["page"] = "form"

elif st.session_state["page"] == "form":
    form_page()

elif st.session_state["page"] == "result":
    result_page()