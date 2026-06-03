import os
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

raw_user = os.getenv("MONGO_USER")
raw_pass = os.getenv("MONGO_PASS")
cluster_url = os.getenv("MONGO_CLUSTER")

username = urllib.parse.quote_plus(raw_user) if raw_user else ""
password = urllib.parse.quote_plus(raw_pass) if raw_pass else ""

MONGO_URI = f"mongodb+srv://{username}:{password}@{cluster_url}/?retryWrites=true&w=majority"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "documind_chunks")