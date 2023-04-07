import sqlite3
import pandas as pd
import numpy as np
import torch
import transformers as ppb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# Connessione al database
conn = sqlite3.connect('nome_del_database.db')

# Creazione di un cursore
c = conn.cursor()

# Esecuzione di una query per recuperare i dati da tutte le tabelle del database
c.execute("SELECT * FROM sqlite_master WHERE type='table'")

# Recupero dei nomi delle tabelle
tables = c.fetchall()

# Creazione di una lista per contenere tutti i dati
data = []

# Loop attraverso le tabelle per recuperare i dati
for table_name in tables:
    c.execute(f"SELECT * FROM {table_name[1]}")
    data += c.fetchall()

# Chiusura della connessione
conn.close()

# Creazione del DataFrame
df = pd.DataFrame(data, columns=['text', 'label'])

# Preprocessamento del testo
model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

tokenized = df['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

max_len = 0
for i in tokenized.values:
    if len(i) > max_len:
        max_len = len(i)

padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

input_ids = torch.tensor(padded).long() 
attention_mask = torch.tensor(attention_mask)

# Esegui il modello di embedding
with torch.no_grad():
    last_hidden_states = model(input_ids, attention_mask=attention_mask)

features = last_hidden_states[0][:,0,:].numpy()

# Addestra il modello di classificazione
labels = df['label']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels)
lr_clf = LogisticRegression()
lr_clf.fit(train_features, train_labels)

# Valuta il modello di classificazione
y_pred = lr_clf.predict(test_features)
accuracy = accuracy_score(test_labels, y_pred)
print(f'Accuracy: {accuracy}')
