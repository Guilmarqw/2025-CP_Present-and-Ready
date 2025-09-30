import mysql.connector

try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",         # default in XAMPP
        password="",         # default is empty in XAMPP
        database="fcee"   # or "fcee"
    )
    if conn.is_connected():
        print("✅ Successfully connected to database!")
    conn.close()
except Exception as e:
    print("❌ Error:", e)
