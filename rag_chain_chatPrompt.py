from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

FAISS_DIR = "vector_store/faiss_openai"

# 1) Load vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

# 2) Retrieve relevant chunks 
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a manufacturing operations + quality assistant for Toyota Camry body assembly (welding).\n"
     "You MUST follow these rules:\n"
     "1) Use ONLY the provided context.\n"
     "2) If the context is insufficient, say: 'I don't know based on the provided context.'\n"
     "3) Do not invent numbers, thresholds, or procedures not present in the context.\n"
     "4) Prefer citing which SOURCE block supports each claim.\n"),
    ("human",
     "Context:\n{context}\n\n"
     "Question:\n{question}\n\n"
     "Respond in this exact format:\n"
     "A) Problem summary (1–2 sentences)\n"
     "B) Likely root causes (bullet list)\n"
     "C) Evidence (bullet list with SOURCE numbers like [1], [2])\n"
     "D) Immediate actions (next 24 hours) (bullets)\n"
     "E) Preventive actions (next 2–4 weeks) (bullets)\n"
     "F) Escalation (who + why) (bullets)\n")
])


def format_context(docs) -> str:
    # Keep it readable and cite sources inside the context
    blocks = []
    for i, d in enumerate(docs, 1):
        md = d.metadata
        src = f"{md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')}"
        text = d.page_content.strip()
        blocks.append(f"[{i}] SOURCE: {src}\n{text}")
    return "\n\n".join(blocks)

question = "Welding misalignment defects are increasing. What are common causes and what maintenance actions should we take?"
docs = retriever.invoke(question)

# 3) Build context + call LLM
context = format_context(docs)
message = prompt.format_messages(context=context,question=question)
    
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke(message)

if __name__ == "__main__":
    # 4) Print answer
    print("\nANSWER:\n")
    print(response.content)

    # 5) Print sources list
    print("\nSOURCES USED:\n")
    for d in docs:
        md = d.metadata
        print(f"- {md.get('domain')} | {md.get('source_file')} | page {md.get('page')}")
    