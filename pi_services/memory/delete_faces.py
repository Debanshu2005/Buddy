import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv('DB_HOST'),
    database=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    port=os.getenv('DB_PORT'),
    sslmode='require'
)
cur = conn.cursor()
cur.execute("DELETE FROM faces")
conn.commit()
print(f"Cleared all faces: {cur.rowcount} rows deleted")
cur.close()
conn.close()
