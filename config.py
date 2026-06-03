import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

raw_user = os.getenv("MONGO_USER")
raw_pass = os.getenv("MONGO_PASS")
cluster_url = os.getenv("MONGO_CLUSTER")

username = urllib.parse.quote_plus(raw_user) if raw_user else ""
password = urllib.parse.quote_plus(raw_pass) if raw_pass else ""

MONGO_URI = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority"

INDEX_DIR = "indexes"