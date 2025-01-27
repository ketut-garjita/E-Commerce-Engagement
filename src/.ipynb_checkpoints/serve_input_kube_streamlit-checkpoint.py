import streamlit as st
import requests
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
# Update path to tokenizer based on Docker container structure
with open('./models/tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Maximum sequence length
MAX_SEQUENCE_LENGTH = 20

# TensorFlow Serving URL (use container name 'tf-serving' from docker-compose.yaml)
TF_SERVING_URL = "http://tf-serving:8501/v1/models/saved_model:predict"

def preprocess_input(text):
    """
    Preprocess input text by converting it to sequences and padding.
    Args:
        text (str): Input text
    Returns:
        list: Preprocessed and padded input
    """
    sequences = tokenizer.texts_to_sequences([text])
    padded_input = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    return padded_input.tolist()

# Streamlit app setup
st.title("E-Commerce Engagement Prediction")
st.markdown("##### Version: Docker Container")
st.markdown("#### Enter a text to predict the engagement level")

# Input text field
input_text = st.text_area("Input Text:", "")

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
