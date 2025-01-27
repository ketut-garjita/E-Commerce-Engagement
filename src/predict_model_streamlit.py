import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load tokenizer and model
with open('../models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

model = load_model('../models/e-commerce-engagement-model.keras')

# Set a maximum sequence length (should match what was used during training)
MAX_SEQUENCE_LENGTH = 20

# Streamlit app setup
st.title("E-Commerce Engagement Prediction")
st.markdown("##### Version: Local") 
st.markdown("#### Enter a text to predict the engagement level") 

# Input text field
input_text = st.text_area("Input Text:", "")

# Predict button
if st.button("Predict"):
    if not input_text.strip():
        st.error("No text provided. Please enter some text.")
    else:
        try:
            # Convert text to sequences
            sequences = tokenizer.texts_to_sequences([input_text])
            
            # Pad the sequences to ensure consistent input shape
            padded_input = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

            # Convert to NumPy array (Keras requires this)
            processed_input = np.array(padded_input)

            # Perform prediction
            prediction = model.predict(processed_input)

            # Display prediction
            rounded_prediction = [[round(value, 2) for value in pred] for pred in prediction.tolist()]
            st.success(f"Prediction: {rounded_prediction}")


        except Exception as e:
            st.error(f"An error occurred: {e}")
