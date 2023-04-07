import mysql
import pandas as pd

# connessione al database
conn = mysql.connector.connect(
    host='179.43.182.186',
    user='tedesco.andrea.at@gmail.com',
    password='TeddyGoddy03!?',
    database='dbname'
)

# query per ottenere tutti i dati dal database
c = conn.cursor()
c.execute("SHOW TABLES")
table_names = [tbl[0] for tbl in c.fetchall()]

# query per ottenere tutti i dati da tutte le tabelle del database
dfs = []
for table_name in table_names:
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    dfs.append(df)

# concatenazione dei dati di tutte le tabelle in un unico DataFrame
data = pd.concat(dfs)

# chiusura della connessione al database
conn.close()

# salvataggio dei dati in un file CSV
data.to_csv('training_data.csv', index=False)
