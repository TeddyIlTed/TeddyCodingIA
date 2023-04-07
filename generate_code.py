import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_code(text):
    # Tokenizzazione del testo
    tokens = tokenizer.texts_to_sequences([text])

    # Padding del testo per ottenere una sequenza della stessa lunghezza del dataset
    padded_tokens = pad_sequences(tokens, maxlen=max_sequence_length, padding='post')

    # Generazione del codice
    generated_code = model.predict(padded_tokens)
    
    # Decodifica dei token generati
    generated_text = tokenizer.sequences_to_texts(generated_code)[0]

    return generated_text
