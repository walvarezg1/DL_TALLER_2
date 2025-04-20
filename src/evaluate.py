from data_loader import load_data
from utils import evaluate_saved_model
from utils import load_saved_model, tokenizador

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Input, Bidirectional, LSTM, Dense, Layer
from tensorflow.keras.models import Model

class Attention(Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        score = tf.nn.tanh(inputs)
        weights = tf.nn.softmax(score, axis=1)
        context = weights * inputs
        return tf.reduce_sum(context, axis=1)

custom_objects = {'Attention': Attention}

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = load_data(url)

sequences, padded = tokenizador(df)

X = padded
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# lstm_model = load_saved_model('../models/lstm_model.h5')

# bilstm_model = load_saved_model('../models/bilstm_attention_model.h5')

print("EVALUACIÓN DEL MODELO RNN")
metrics = evaluate_saved_model(
    model_path='../models/rnn_model.h5',
    X_test=X_test,  # Asegúrate que tiene el mismo preprocesamiento que los datos de entrenamiento
    y_test=y_test,
    threshold=0.5  # Puedes ajustar este umbral según tus necesidades
)

print("EVALUACIÓN DEL MODELO LSTM")
metrics = evaluate_saved_model(
    model_path='../models/lstm_model.h5',
    X_test=X_test,  # Asegúrate que tiene el mismo preprocesamiento que los datos de entrenamiento
    y_test=y_test,
    threshold=0.5  # Puedes ajustar este umbral según tus necesidades
)

print("EVALUACIÓN DEL MODELO BiLSTM CON ATENCIÓN")
metrics = evaluate_saved_model(
    model_path='../models/bilstm_attention_model.h5',
    X_test=X_test,  # Asegúrate que tiene el mismo preprocesamiento que los datos de entrenamiento
    y_test=y_test,
    threshold=0.5,  # Puedes ajustar este umbral según tus necesidades
    custom_objects=custom_objects
)