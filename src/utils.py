from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


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

def load_saved_model(filename):
    try:
        model = load_model(filename)
        print(f"Modelo cargado correctamente desde {filename}")
        return model
    except Exception as e:
        print(f"Error al cargar el modelo {filename}: {e}")
        return None

def plot_model_history(history, model_name):
    epochs = range(1, len(history['accuracy']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.suptitle(f"Evaluación del modelo {model_name}", fontsize=16, fontweight='bold')
    
    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['accuracy'], label='Entrenamiento')
    plt.plot(epochs, history['val_accuracy'], label='Validación')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['loss'], label='Entrenamiento')
    plt.plot(epochs, history['val_loss'], label='Validación')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def evaluate_saved_model(model_path, X_test, y_test, threshold=0.5, custom_objects=''):
    try:
        # 1. Cargar el modelo
        if custom_objects == '':
            model = load_model(model_path)
        else:
            model = load_model(model_path, custom_objects=custom_objects)
        print(f"✅ Modelo cargado correctamente desde {model_path}")

        # 2. Predecir probabilidades
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba >= threshold).astype(int)

        # 3. Calcular métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'loss': model.evaluate(X_test, y_test, verbose=0)[0]
        }

        # 4. Mostrar resultados
        print("\n📊 Métricas de evaluación:")
        print(f"- Accuracy: {metrics['accuracy']:.4f}")
        # print(f"- Precision: {metrics['precision']:.4f}")
        # print(f"- Recall: {metrics['recall']:.4f}")
        # print(f"- F1-Score: {metrics['f1']:.4f}")
        print(f"- ROC AUC: {metrics['roc_auc']:.4f}")
        # print(f"- Loss: {metrics['loss']:.4f}")
        print("\n🔢 Matriz de confusión:")
        print(np.array(metrics['confusion_matrix']))

        return metrics

    except Exception as e:
        print(f"❌ Error al evaluar el modelo: {str(e)}")
        return None