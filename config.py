import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'a_secret_key'
    MONGO_URI = os.environ.get('MONGO_URI') or 'mongodb+srv://maryam_elmou:MAR%40yam123@bigdata.zdq8pq6.mongodb.net/traffic_database?retryWrites=true&w=majority&appName=BigData'
