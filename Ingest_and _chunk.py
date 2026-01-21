from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os, json, re
from typing import List
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = "Data"
OUT_FILE = "artifacts/chunks.jsonl"

print("DATA_DIR exists?", os.path.exists(DATA_DIR))

#Text cleaning
def clean_text(t: str) -> str:
    t = t.replace("\u00a0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

#Data Load
loader = DirectoryLoader(
    DATA_DIR,
    glob='**/*.pdf',
    loader_cls=PyPDFLoader
)
docs = loader.load()

# Add simple metadata: domain (folder name) + filename (Better Retriever)
for d in docs:
    source_path = d.metadata.get("source", "")
    parts = source_path.replace("\\", "/").split("/")
    d.metadata["domain"] = parts[1] if len(parts) > 2 else "unknown"
    d.metadata["source_file"] = parts[-1]
    
print('Loaded pages: ', len(docs))
print(docs[0])

# Clean + chunk
for d in docs:
    d.page_content = clean_text(d.page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1100,
    chunk_overlap=180,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(docs)
print(f"Created chunks: {len(chunks)}")

# Save chunks
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
with open(OUT_FILE, "w", encoding="utf-8") as f:
    for c in chunks:
        f.write(json.dumps({"text": c.page_content, "metadata": c.metadata}, ensure_ascii=False) + "\n")
    

print(f"Saved to: {OUT_FILE}")
print("Sample chunk:", chunks[0].page_content if chunks else "No chunks")
