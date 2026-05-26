import os
import dotenv
from llamaindex.embeddings import set_emebed_model
from llamaindex.loading_data import load_documents
from llamaindex.indexing import upsert_data

dotenv.load_dotenv()

db_index_name = os.getenv("DB_INDEX_NAME")

print("Setting up embedding model...")
set_emebed_model()

print("Loading documents from ./book ...")
docs = load_documents("./book")
print(f"Loaded {len(docs)} document chunks.")

print("Uploading to Pinecone...")
upsert_data(db_index_name, docs)
print("Done! Your books are indexed in Pinecone.")
