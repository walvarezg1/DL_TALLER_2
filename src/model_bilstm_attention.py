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

def build_bilstm_attention(vocab_size=10000, embedding_dim=64, max_length=100):
    inputs = Input(shape=(max_length,))
    x = Embedding(vocab_size, embedding_dim)(inputs)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Attention()(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

def run_bilstm(X_train, y_train, X_test, y_test):
    bilstm_model = build_bilstm_attention()
    bilstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    bilstm_model.summary()

    # Entrenamiento
    history = bilstm_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    return history, bilstm_model
