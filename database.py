from pymongo import MongoClient
from config import MONGO_URI

client = MongoClient(MONGO_URI)

db = client["DocuMind_DB"]

chat_history = db["chat_sessions"]