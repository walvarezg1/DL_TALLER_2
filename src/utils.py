from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import save_model


def tokenizador(df, vocab_size = 10000, max_length = 100, oov_token = "<OOV>"):
    # Tokenizador
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(df["tweet"])

    # Transformar textos a secuencias
    sequences = tokenizer.texts_to_sequences(df["tweet"])
    padded = pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")
    return sequences, padded

def save_model(model, filename):
    model.save(filename)
    print(f'Modelo guardado en {filename}')