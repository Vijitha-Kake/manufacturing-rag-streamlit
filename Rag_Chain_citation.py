from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

FAISS_DIR = "vector_store/faiss_openai"

# Load FAISS
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local(FAISS_DIR, embeddings, allow_dangerous_deserialization=True)

TOP_K = 4

# 1) Retrieve with scores (FAISS returns List[(Document, score)])
def retrieve_with_scores(question: str):
    return db.similarity_search_with_score(question, k=TOP_K)

# 2) Format context blocks with score + metadata
def format_context(doc_score_pairs) -> str:
    blocks = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        src = f"{md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')} | score={score:.4f}"
        blocks.append(f"[{i}] SOURCE: {src}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)

# 3) Build sources list (clean summary)
def build_sources_list(doc_score_pairs) -> str:
    lines = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        lines.append(f"[{i}] {md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')} | score={score:.4f}")
    return "\n".join(lines)

# 4) Parallel: pass question + retrieve doc-score pairs
parallel_chain = RunnableParallel({
    "question": RunnablePassthrough(),
    "doc_scores": RunnableLambda(retrieve_with_scores),
})

# 5) Convert doc-score pairs to prompt inputs
to_prompt_inputs = RunnableLambda(lambda x: {
    "question": x["question"],
    "context": format_context(x["doc_scores"]),
    "sources": build_sources_list(x["doc_scores"]),
})

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a manufacturing operations + quality assistant for Toyota Camry body assembly (welding).\n"
     "Rules:\n"
     "1) Use ONLY the provided context blocks.\n"
     "2) If insufficient, say: 'I don't know based on the provided context.'\n"
     "3) Do not invent numbers/procedures.\n"
     "4) In sections B–F, include citations like [1] or [1][2] for every bullet.\n"
     "5) Citations MUST refer to the numbered context blocks.\n"),
    ("human",
     "Context blocks:\n{context}\n\n"
     "Question:\n{question}\n\n"
     "Respond in this exact format:\n"
     "A) Problem summary (1–2 sentences)\n"
     "B) Likely root causes (bullet list with citations like [1])\n"
     "C) Evidence (bullet list with citations like [1], [2])\n"
     "D) Immediate actions (next 24 hours) (bullets with citations)\n"
     "E) Preventive actions (next 2–4 weeks) (bullets with citations)\n"
     "F) Escalation (who + why) (bullets with citations)\n\n"
     "G) Sources (copy exactly the list below)\n"
     "{sources}\n"
    )
])

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

rag_chain = parallel_chain | to_prompt_inputs | prompt | model | parser

if __name__ == "__main__":
    q = "Welding misalignment defects are increasing. What are common causes and what maintenance actions should we take?"
    print("\nANSWER:\n")
    print(rag_chain.invoke(q))

    print("\nGRAPH:\n")
    rag_chain.get_graph().print_ascii()
