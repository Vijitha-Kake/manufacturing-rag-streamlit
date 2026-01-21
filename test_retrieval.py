from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

db = FAISS.load_local(
    "vector_store/faiss_openai",
    OpenAIEmbeddings(model="text-embedding-3-small"),
    allow_dangerous_deserialization=True
)

query = "Why are welding misalignment defects increasing?"
results = db.similarity_search_with_score(query, k=3)

for doc, score in results:
    print("\n---")
    print("Score:", score)
    print("Source:", doc.metadata.get("source_file"))
    print("Domain:", doc.metadata.get("domain"))
    print(doc.page_content[:300])
