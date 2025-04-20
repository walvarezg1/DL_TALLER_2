import pandas as pd
import tensorflow as tf

def load_data(url: str):
    csv_path = tf.keras.utils.get_file("twitter_sentiment.csv", url)
    df = pd.read_csv(csv_path)

    df = df[["tweet", "label"]]
    return df