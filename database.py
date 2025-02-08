import mysql.connector

def connect_db():
    return mysql.connector.connect(
        host="localhost",
        user="root",  # Change if you set a password
        password="",  # If you have a password, enter it here
        database="facerecognition"
    )
