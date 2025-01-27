##########################################################################
# Program Name : train_model_pyspark.py
# Purpose : train a model
# Kaggle Dataset Source : obertvici/indonesia-top-ecommerce-unicorn-tweets
# Location of Dataset Loaded : Linux HDFS
# Data Processsing Tools: pyspark
###########################################################################

import subprocess
import os
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, lower, regexp_replace, count
from pyspark.sql.types import IntegerType, StringType
from pyspark.sql.functions import lit
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

# Restart DFS
print("Stopping DFS...")
run_command("stop-dfs.sh")

print("Starting DFS...")
run_command("start-dfs.sh")

# Check running Java services
print("Checking Java processes...")
run_command("jps")

# Check Safe Mode status
print("Checking Safe Mode status...")
run_command("hdfs dfsadmin -safemode get")

# Leave Safe Mode if it's ON
print("Leaving Safe Mode if necessary...")
run_command("hdfs dfsadmin -safemode leave")

# Create HDFS directories
hdfs_dirs = [
    "kaggle/datasets",
    "kaggle/splits"
]

for hdfs_dir in hdfs_dirs:
    print(f"Creating HDFS directory: {hdfs_dir}")
    run_command(f"hdfs dfs -mkdir -p {hdfs_dir}")

# Download dataset from Kaggle
kaggle_dataset_path = "../kaggle/datasets"
dataset_name = "indonesia-top-ecommerce-unicorn-tweets"
print("Downloading dataset from Kaggle...")
run_command(f"kaggle datasets download -d robertvici/{dataset_name} -p {kaggle_dataset_path}")

# Unzip the downloaded dataset
zip_file_path = f"{kaggle_dataset_path}/{dataset_name}.zip"
print("Unzipping dataset...")
run_command(f"unzip -o {zip_file_path} -d {kaggle_dataset_path}")

# Upload files to HDFS
print("Uploading Datasets JSON files to HDFS...")
run_command(f"hdfs dfs -put {kaggle_dataset_path}/*.json kaggle/datasets")

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("E-Commerce Engagement Prediction ML") \
    .getOrCreate()

# Load datasets from HDFS
blibli_df = spark.read.json('hdfs://localhost:9000/user/dataeng/kaggle/datasets/bliblidotcom.json')
bukalapak_df = spark.read.json('hdfs://localhost:9000/user/dataeng/kaggle/datasets/bukalapak.json')
lazadaID_df = spark.read.json('hdfs://localhost:9000/user/dataeng/kaggle/datasets/lazadaID.json')
shopeeID_df = spark.read.json('hdfs://localhost:9000/user/dataeng/kaggle/datasets/ShopeeID.json')
tokopedia_df = spark.read.json('hdfs://localhost:9000/user/dataeng/kaggle/datasets/tokopedia.json')

# Add a new column to identify the company source
blibli_df = blibli_df.withColumn('source', lit('blibli'))
bukalapak_df = bukalapak_df.withColumn('source', lit('bukalapak'))
lazadaID_df = lazadaID_df.withColumn('source', lit('lazadaID'))
shopeeID_df = shopeeID_df.withColumn('source', lit('shopeeID'))
tokopedia_df = tokopedia_df.withColumn('source', lit('tokopedia'))

# Merge datasets using union (axis=0 equivalent in Spark)
merged_df = blibli_df.union(bukalapak_df).union(lazadaID_df).union(shopeeID_df).union(tokopedia_df)

# Clean tweet text
def clean_text(text):
    return text.lower().replace("#", "").strip()

clean_text_udf = udf(clean_text, StringType())

# Apply text cleaning and create new features
data_cleaned = merged_df.withColumn("clean_tweet", clean_text_udf(col("tweet"))) \
                       .withColumn("engagement", col("replies_count") + col("retweets_count") + col("likes_count"))

# Select relevant features
selected_data = data_cleaned.select(
    col("clean_tweet").alias("text"),
    col("replies_count").alias("replies"),
    col("retweets_count").alias("retweets"),
    col("likes_count").alias("likes"),
    col("engagement").alias("target"),
    col("hashtags"),    
    col("source")
)

# Split dataset
print("Splitting dataset into train, validate, and test")
train_data, validate_data, test_data = selected_data.randomSplit([0.7, 0.15, 0.15], seed=42)

# Save splits on HDFS for later use
splits_dataset_path = "../kaggle/splits"
train_data.write.json(f"{splits_dataset_path}/train.json", mode="overwrite")
validate_data.write.json(f"{splits_dataset_path}/validate.json", mode="overwrite")
test_data.write.json(f"{splits_dataset_path}/test.json", mode="overwrite")

# Remove datasets on local file system
print("Removing local dataset files...")
run_command(f"rm -r ../kaggle")

# Change null value with 0 (if any)
merged_df = merged_df.fillna({"likes_count": 0, "replies_count": 0, "retweets_count": 0})

from pyspark.sql import functions as F

# Check negative value
merged_df.filter((F.col("likes_count") < 0) | (F.col("replies_count") < 0) | (F.col("retweets_count") < 0)).show()

# Change negative value with 0 (if any)
for col in ["likes_count", "replies_count", "retweets_count"]:
    merged_df = merged_df.withColumn(col, F.when(F.col(col) < 0, 0).otherwise(F.col(col)))
    
# Matching target engagement definitions in Spark DataFrame
blibli_df = blibli_df.withColumn("engagement", F.col("likes_count") + F.col("replies_count") + F.col("retweets_count"))
bukalapak_df = bukalapak_df.withColumn("engagement", F.col("likes_count") + F.col("replies_count") + F.col("retweets_count"))
lazadaID_df = lazadaID_df.withColumn("engagement",   F.col("likes_count") + F.col("replies_count") + F.col("retweets_count"))
shopeeID_df = shopeeID_df.withColumn("engagement",   F.col("likes_count") + F.col("replies_count") + F.col("retweets_count"))
tokopedia_df = tokopedia_df.withColumn("engagement", F.col("likes_count") + F.col("replies_count") + F.col("retweets_count"))

# Load train data (convert Spark DataFrame to Pandas)
train_df = train_data.toPandas()

# Tokenize and vectorize text (fit on original text, not the padded sequences)
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_df["text"])  # Fit tokenizer on the raw text data

# Convert texts to sequences
X_train = tokenizer.texts_to_sequences(train_df["text"])

# Pad the sequences to ensure uniform length
X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, padding='post')

y_train = np.array(train_df["target"])


# Define a simple Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=5000, output_dim=64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation="linear")
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

vocab_size = 42500  # Customize with your tokenizer
embedding_dim = 128
embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)

# Example: Tokenize and pad the input text
max_vocab_size = 5000
tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(train_df["text"])  # 'train_df["text"]' should be a list of strings

# Example of saving the tokenizer
with open('../models/tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)

X_train_sequences = tokenizer.texts_to_sequences(train_df["text"])
X_train = pad_sequences(X_train_sequences, padding='post')

# Ensure y_train is in the correct format (e.g., a numpy array)
y_train = np.array(train_df["target"])  # Adjust this based on your target column

# Train the model 
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Save the model with a valid file extension in local server
model.save("../models/e-commerce-engagement-model.keras")  # For the native Keras format

# Export to save model
model.export("../saved_model/1")
print("Final Model ==> saved_model/1")

# Stop the Spark session
spark.stop()

print("")
print("All model train tasks done successfully!")