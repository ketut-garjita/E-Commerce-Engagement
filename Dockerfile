FROM python:3.12

ENV MODEL_NAME=saved_model

# Copy saved model to TensorFlow Serving path
COPY ./saved_model /models/saved_model

# Copy requirements.txt and install dependensi:
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r /app/requirements.txt

# Copy Streamlit API and tokenizer
COPY ./src/app.py /app/app.py
COPY ./models/tokenizer.pkl /app/models/tokenizer.pkl

# Expose ports
EXPOSE 8501 8502

# Execute Streamlit
CMD ["streamlit", "run", "/app/app.py",  "--server.port=8502"]
