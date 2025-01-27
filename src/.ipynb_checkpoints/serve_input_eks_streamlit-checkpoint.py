import streamlit as st
import requests
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Maximum sequence length
MAX_SEQUENCE_LENGTH = 20

# TensorFlow Serving URL
TF_SERVING_URL = "http://a09b1c4049dec41438aaf1b11012942f-1058378139.ap-southeast-3.elb.amazonaws.com:8501/v1/models/saved_model:predict"

def preprocess_input(text):
    # Convert text to sequences and pad
    sequences = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_input.tolist()

# Streamlit app setup
st.title("TensorFlow Serving Prediction App")
st.write("Enter text to predict engagement level using the model deployed on EKS.")

# Input text field
input_text = st.text_area("Input Text", "")

# Predict button
if st.button("Predict"):
    if not input_text.strip():
        st.error("No text provided. Please enter some text.")
    else:
        try:
            # Preprocess input
            processed_input = preprocess_input(input_text)
            payload = {"instances": processed_input}

            # Send request to TensorFlow Serving
            response = requests.post(TF_SERVING_URL, json=payload)

            if response.status_code == 200:
                prediction = response.json()

                # Round prediction values to 2 decimal places
                if "predictions" in prediction:
                    rounded_prediction = [[round(value, 2) for value in pred] for pred in prediction["predictions"]]
                    st.success(f"Prediction: {rounded_prediction}")
                else:
                    st.error("Unexpected response format from TensorFlow Serving.")
            else:
                st.error(f"Failed to get prediction. Response: {response.text}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
