import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your trained model
@st.cache_resource
def load_model():
    # adjust the path if needed
    with open('P576.ipynb', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("My Model Prediction App")
st.write("Fill in the inputs and get a prediction")

# Suppose your model needs 3 inputs: feature1, feature2, feature3
feature1 = st.number_input("Enter feature 1", value=0.0)
feature2 = st.number_input("Enter feature 2", value=0.0)
feature3 = st.number_input("Enter feature 3", value=0.0)

# When the user presses predict
if st.button("Predict"):
    input_array = np.array([[feature1, feature2, feature3]])
    # maybe scaling or preprocessing if done before training – include that
    prediction = model.predict(input_array)
    st.write(f"Prediction: {prediction[0]}")
