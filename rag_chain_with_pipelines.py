from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

FAISS_DIR = "vector_store/faiss_openai"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a manufacturing operations + quality assistant for Toyota Camry body assembly (welding).\n"
     "Rules:\n"
     "1) Use ONLY the provided context.\n"
     "2) If insufficient, say: 'I don't know based on the provided context.'\n"
     "3) Do not invent numbers/procedures.\n"
     "4) Cite SOURCE blocks like [1], [2].\n"),
    ("human",
     "Context:\n{context}\n\nQuestion:\n{question}\n\n"
     "Respond in this exact format:\n"
     "A) Problem summary (1–2 sentences)\n"
     "B) Likely root causes (bullet list)\n"
     "C) Evidence (bullet list with SOURCE numbers like [1], [2])\n"
     "D) Immediate actions (next 24 hours) (bullets)\n"
     "E) Preventive actions (next 2–4 weeks) (bullets)\n"
     "F) Escalation (who + why) (bullets)\n")
])

def format_context(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, 1):
        md = d.metadata
        src = f"{md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')}"
        blocks.append(f"[{i}] SOURCE: {src}\n{d.page_content.strip()}")
    return "\n\n".join(blocks)

parallel_chain = RunnableParallel({
    "question": RunnablePassthrough(),
    "context": retriever | RunnableLambda(format_context),
})

model=ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()
chain= prompt | model | parser

rag_chain = parallel_chain | chain

if __name__ == "__main__":
    q = "Welding misalignment defects are increasing. What are common causes and what maintenance actions should we take?"
    print("\nANSWER:\n")
    print(rag_chain.invoke(q))
    rag_chain.get_graph().print_ascii()
    