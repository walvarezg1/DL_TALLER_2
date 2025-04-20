from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

def build_rnn(vocab_size=10000, embedding_dim=64, max_length=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        SimpleRNN(64),
        Dense(1, activation='sigmoid')
    ])
    return model

def run_rnn(X_train, y_train, X_test, y_test):
    rnn_model = build_rnn()
    rnn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    rnn_model.summary()

    # Entrenamiento
    history = rnn_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    return history, rnn_model