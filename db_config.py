import mysql.connector
from mysql.connector import Error

def get_connection():
    try:
        connection = mysql.connector.connect(
            host="192.168.1.20",          # your MySQL host
            user="test",               # your MySQL username
            password="test",    # your MySQL password
            database="Edquestdb"          # your database name
        )
        return connection
    except Error as e:
        print("Error while connecting to MySQL:", e)
        return None
