from data_loader import load_data
from utils import tokenizador, save_model
from model_rnn import run_rnn
from model_lstm import run_lstm
from model_bilstm_attention import run_bilstm

from sklearn.model_selection import train_test_split

url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = load_data(url)

sequences, padded = tokenizador(df)

X = padded
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)

def train_rnn():
    history_rnn, rnn_model = run_rnn(X_train, y_train, X_test, y_test)
    save_model(rnn_model, '../models/rnn_model.h5')
    return history_rnn

def train_lstm():
    history_lstm, lstm_model = run_lstm(X_train, y_train, X_test, y_test)
    save_model(lstm_model, '../models/lstm_model.h5')
    return history_lstm

def train_bilstm():
    history_bilstm, bilstm_model = run_bilstm(X_train, y_train, X_test, y_test)
    save_model(bilstm_model, '../models/bilstm_attention_model.h5')
    return history_bilstm

# Entreno el modelo RNN y lo guardo
train_rnn()
# Entreno el modelo LSTM y lo guardo
train_lstm()
# Entreno el modelo BiLSTM y lo guardo
train_bilstm()