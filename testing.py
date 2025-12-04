import mysql.connector
from mysql.connector import Error

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facesys',
    'autocommit': False,
    'pool_name': 'mypool',
    'pool_size': 5
}

try:
    conn = mysql.connector.connect(**DB_CONFIG)
    if conn.is_connected():
        print("   Connection successful!")
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES;")
        print("Tables:", [row[0] for row in cursor.fetchall()])
        conn.close()
    else:
        print("  Connection failed — no connection object returned.")
except Error as e:
    print("  Connection error:", e)
