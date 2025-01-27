##########################################################################
# Program Name : train_model_pandas.py
# Purpose : train a model
# Kaggle Dataset Source : obertvici/indonesia-top-ecommerce-unicorn-tweets
# Location of Dataset Loaded : Local File System
# Data Processsing Tools: pandas
###########################################################################

import subprocess
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def run_command(command):
    """Utility function to run shell commands"""
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode == 0:
        print(f"Success: {command}\nOutput:\n{stdout.decode()}")
    else:
        print(f"Error: {command}\nError Message:\n{stderr.decode()}")


# Create directories
print("Creating directories...")
dirs = [
    "kaggle/datasets",
    "kaggle/splits"
]

for dir in dirs:
    print(f"Creating directory: {dirs}")
    run_command(f"mkdir -p ../{dir}")

# Download dataset from Kaggle
print("Downloading dataset from Kaggle...")
kaggle_dataset_path = "../kaggle/datasets"
dataset_name = "indonesia-top-ecommerce-unicorn-tweets"
run_command(f"kaggle datasets download -d robertvici/{dataset_name} -p {kaggle_dataset_path}")

# Unzip the downloaded dataset
print("Unziping the downloaded dataset...")
zip_file_path = f"{kaggle_dataset_path}/{dataset_name}.zip"
run_command(f"unzip -o {zip_file_path} -d {kaggle_dataset_path}")

# Remove indonesia-top-ecommerce-unicorn-tweets.zip
print("Remowing indonesia-top-ecommerce-unicorn-tweets.zip file...")
run_command(f"rm {zip_file_path}")

# Load datasets with Pandas
print("Loading datasets with Pandas...")
blibli_df = pd.read_json(f'{kaggle_dataset_path}/bliblidotcom.json', lines=True)
bukalapak_df = pd.read_json(f'{kaggle_dataset_path}/bukalapak.json', lines=True)
lazadaID_df = pd.read_json(f'{kaggle_dataset_path}/lazadaID.json', lines=True)
shopeeID_df = pd.read_json(f'{kaggle_dataset_path}/ShopeeID.json', lines=True)
tokopedia_df = pd.read_json(f'{kaggle_dataset_path}/tokopedia.json', lines=True)

# Add a new column to identify the company source
print("Adding a new column to identify the company source...")
blibli_df['source'] = 'blibli'
bukalapak_df['source'] = 'bukalapak'
lazadaID_df['source'] = 'lazadaID'
shopeeID_df['source'] = 'shopeeID'
tokopedia_df['source'] = 'tokopedia'

# Merge datasets using concat (equivalent to union in Spark)
print("Merging datasets using concat (equivalent to union in Spark)...")
merged_df = pd.concat([blibli_df, bukalapak_df, lazadaID_df, shopeeID_df, tokopedia_df], axis=0)

# Clean tweet text
print("Cleaning tweet tect")
def clean_text(text):
    return text.lower().replace("#", "").strip()

merged_df['clean_tweet'] = merged_df['tweet'].apply(clean_text)

# Create new feature for engagement
print("Creating  feature for engagement...")
merged_df['engagement'] = merged_df['replies_count'] + merged_df['retweets_count'] + merged_df['likes_count']

# Select relevant features
print("Selecting relevant features...")
selected_data = merged_df[['clean_tweet', 'replies_count', 'retweets_count', 'likes_count', 'engagement', 'hashtags', 'source']]

# Split dataset into train, validate, and test
print("Splitting dataset into train, validate, and test")
splits_dataset_path = "../kaggle/splits"
train_data = selected_data.sample(frac=0.7, random_state=42)
remaining_data = selected_data.drop(train_data.index)
validate_data = remaining_data.sample(frac=0.5, random_state=42)
test_data = remaining_data.drop(validate_data.index)

# Replace null values with 0
print("Replacing null values with 0...")
merged_df.fillna({"likes_count": 0, "replies_count": 0, "retweets_count": 0}, inplace=True)

# Replace negative values with 0
print("Replacing negative values with 0...")
for col in ["likes_count", "replies_count", "retweets_count"]:
    merged_df[col] = merged_df[col].apply(lambda x: max(0, x))

# Tokenize and vectorize text
print ("Tokenize and vectorize text...")
tokenizer = Tokenizer(num_words=5000)
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data["clean_tweet"].values)

# Convert texts to sequences
print("Converting texts to sequences...")
X_train = tokenizer.texts_to_sequences(train_data["clean_tweet"].values)

# Pad the sequences to ensure uniform length
print("Pad the sequences to ensure uniform length...")
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post')

y_train = train_data["engagement"].values

# Define a simple Neural Network model
print("Define a simple Neural Network model...")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

# Check Mxx index in X-train
print("Max index in X_train:", X_train.max())
print("Shape of X_train:", X_train.shape)

# Filter / set index maximum to 5000
X_train[X_train >= 5000] = 0

# Compile the model
print("Compiling the model...")
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
print("Training the model...")
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model
print("Saving the model...")
model.save("../models/e-commerce-engagement-model.keras")

# Save tokenizer for future use
print("Saving tokenizer for future use...")
with open('../models/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

print("Final model and tokenizer saved successfully!")

# Export to save model
model.export("../saved_model/1")
print("Final Model ==> saved_model/1")

# Remove datasets on local file system
print("Removing local dataset files...")
run_command(f"rm -r ../kaggle")

print("")
print("All model train tasks done successfully!")