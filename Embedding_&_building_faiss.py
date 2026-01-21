from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
import os, json, re
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

load_dotenv()

Chunk_File = "artifacts/chunks.jsonl"
Faiss_Dir = "vector_store/faiss_openai"

# Load chunks
docs = []
with open(Chunk_File, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        docs.append(
            Document(
                page_content=obj["text"],
                metadata=obj.get("metadata", {})
            )
        )

print("Loaded chunks:", len(docs))
print("Sample metadata:", docs[0].metadata)
print("Sample text:", docs[0].page_content[:200])

# Create embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Build FAISS index
db = FAISS.from_documents(docs, embeddings)

# Save FAISS index
os.makedirs(os.path.dirname(Faiss_Dir), exist_ok=True)
db.save_local(Faiss_Dir)

print("FAISS index saved at:", Faiss_Dir)