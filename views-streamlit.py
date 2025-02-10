import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv("diabetes.csv")

# Prepare the data
X = data.drop("Outcome","Preganancies", axis=1)
Y = data['Outcome']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit app
st.title("Diabetes Prediction")

# Input fields
import streamlit as st

st.header("Please enter data into the following")

# Specify the appropriate types for each variable
val2 = st.number_input("Glucose", min_value=0.0, step=0.1)  # Float
val3 = st.number_input("Blood Pressure", min_value=0.0, step=0.1)  # Float
val4 = st.number_input("Skin Thickness", min_value=0.0, step=0.1)  # Float
val5 = st.number_input("Insulin", min_value=0.0, step=0.1)  # Float
val6 = st.number_input("BMI", min_value=0.0, step=0.1)  # Float
val7 = st.number_input("Diabetes Pedigree Function", min_value=0.0, step=0.01)  # Float
val8 = st.number_input("Age", min_value=0, step=1)  # Integer

# Predict button
st.text("Click the button below to check if you have diabetes")
if st.button("Check"):
    paired = model.predict([[val2, val3, val4, val5, val6, val7, val8]])
    result1 = ""
    if paired == [1]:
        result1 = "You have Diabetes, Please consult your physician"
    elif paired == [0]:
        result1 = "Great news, you are not diabetic"
    st.write(result1)
