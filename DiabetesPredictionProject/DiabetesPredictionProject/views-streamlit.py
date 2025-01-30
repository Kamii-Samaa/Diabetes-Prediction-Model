import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
data = pd.read_csv(r"C:\Users\Divine Effiom\OneDrive\Documents\Portfolio Projects\Meri Skill Projects\Diabetes Prediction Project\diabetes.csv")

# Prepare the data
X = data.drop("Outcome", axis=1)
Y = data['Outcome']

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Streamlit app
st.title("Diabetes Prediction")

# Input fields
st.header("Please enter data into the following")
val1 = st.number_input("Number of  Pregnancies")
val2 = st.number_input("Glucose")
val3 = st.number_input("Blood Pressure")
val4 = st.number_input("Skin Thickness")
val5 = st.number_input("Insulin")
val6 = st.number_input("BMI")
val7 = st.number_input("Diabetes Pedigree Function")
val8 = st.number_input("Age")

# Predict button
st.text("Click the button below to check if you have diabetes")
if st.button("Check"):
    paired = model.predict([[val1, val2, val3, val4, val5, val6, val7, val8]])
    result1 = ""
    if paired == [1]:
        result1 = "You just might have Diabetes, Consult your physician"
    elif paired == [0]:
        result1 = "Looks like you don't have diabetes"
    st.write(result1)
