import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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

y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print (f"Your model's accuracy score is: {accuracy}")
