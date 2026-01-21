from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = "vector_store/faiss_openai"

SYSTEM_PROMPT = """You are a manufacturing operations assistant for Toyota Camry body assembly (welding).

Use ONLY the provided context. If the context is insufficient, say what is missing.

Answer in this structure:
1) Likely root cause(s)
2) Evidence from the context (quote short phrases)
3) Recommended immediate actions (next 24 hours)
4) Preventive actions (next 2–4 weeks)
5) Who to escalate to (Maintenance / Quality / Supplier) and why
"""

def format_context(docs) -> str:
    # Keep it readable and cite sources inside the context
    blocks = []
    for i, d in enumerate(docs, 1):
        md = d.metadata
        src = f"{md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')}"
        text = d.page_content.strip()
        blocks.append(f"[{i}] SOURCE: {src}\n{text}")
    return "\n\n".join(blocks)

if __name__ == "__main__":
    # 1) Load vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

    # 2) Retrieve relevant chunks 
    retriever = db.as_retriever(search_kwargs={"k": 6, "fetch_k": 20})

    question = "Welding misalignment defects are increasing. What are common causes and what maintenance actions should we take?"
    docs = retriever.invoke(question)
    
    # 3) Build context + call LLM
    context = format_context(docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    user_prompt = f"""Context:{context}
    Question:{question}
"""

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # 4) Print answer
    print("\nANSWER:\n")
    print(response.content)

    # 5) Print sources list
    print("\nSOURCES USED:\n")
    for d in docs:
        md = d.metadata
        print(f"- {md.get('domain')} | {md.get('source_file')} | page {md.get('page')}")
