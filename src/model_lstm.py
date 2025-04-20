from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, LSTM

def build_lstm(vocab_size=10000, embedding_dim=64, max_length=100):
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(64),
        Dense(1, activation='sigmoid')
    ])
    return model

def run_lstm(X_train, y_train, X_test, y_test):
    lstm_model = build_lstm()
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.summary()

    # Entrenamiento
    history = lstm_model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

    return history, lstm_model