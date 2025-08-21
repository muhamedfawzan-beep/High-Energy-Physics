import sqlite3
import pandas as pd

# Part 2: SQL pipeline
def build_sql_pipeline(sampled, db_name="events.db"):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS events")
    cur.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, theta REAL)")
    cur.executemany("INSERT INTO events(theta) VALUES (?)", [(float(x),) for x in sampled])
    conn.commit()

    # Preprocess with SQL → pandas
    df = pd.read_sql("SELECT * FROM events", conn)
    conn.close()
    return df
