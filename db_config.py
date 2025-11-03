import mysql.connector
from mysql.connector import Error

def get_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",          # your MySQL host
            user="root",               # your MySQL username
            password="Passw0rd",    # your MySQL password
            database="Edquestdb"          # your database name
        )
        return connection
    except Error as e:
        print("Error while connecting to MySQL:", e)
        return None
