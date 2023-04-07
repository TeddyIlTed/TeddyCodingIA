import sqlite3
import pandas as pd


# connessione al database
conn = sqlite3.connect('mydatabase.db')
c = conn.cursor()

# query per ottenere tutti i dati dal database
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
table_names = [tbl[0] for tbl in c.fetchall()]


# query per ottenere tutti i dati da tutte le tabelle del database
dfs = []
for table_name in table_names:
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    dfs.append(df)
df = dfs[:1000]

# concatenazione dei dati di tutte le tabelle in un unico DataFrame
data = pd.concat(dfs)

# chiusura della connessione al database
conn.close()

# salvataggio dei dati in un file CSV
data.to_csv('training_data.csv', index=False)
df.drop_duplicates(subset=['Text'], inplace=True)
