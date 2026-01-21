import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ---------- App Config ----------
st.set_page_config(page_title="Decision Intelligence Platform for Manufacturing Defect Analysis (RAG)", layout="wide")

load_dotenv()

FAISS_DIR = "vector_store/faiss_openai"

# ---------- Helpers (your same logic) ----------
def format_context(doc_score_pairs) -> str:
    blocks = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        src = f"{md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')} | score={score:.4f}"
        blocks.append(f"[{i}] SOURCE: {src}\n{doc.page_content.strip()}")
    return "\n\n".join(blocks)

def build_sources_list(doc_score_pairs) -> str:
    lines = []
    for i, (doc, score) in enumerate(doc_score_pairs, 1):
        md = doc.metadata
        lines.append(f"[{i}] {md.get('source_file')} | domain={md.get('domain')} | page={md.get('page')} | score={score:.4f}")
    return "\n".join(lines)

def build_prompt():
    return ChatPromptTemplate.from_messages([
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

# ---------- Caching ----------
@st.cache_resource
def load_vectorstore():
    """
    Loads embeddings + FAISS one time per Streamlit session.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True  # OK if the FAISS index is created by you and trusted
    )
    return db

def build_rag_chain(db, top_k: int, model_name: str, temperature: float):
    # 1) Retrieve with scores
    def retrieve_with_scores(question: str):
        return db.similarity_search_with_score(question, k=top_k)

    # 2) Parallel: question + retrieved pairs
    parallel_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "doc_scores": RunnableLambda(retrieve_with_scores),
    })

    # 3) Convert to prompt inputs
    to_prompt_inputs = RunnableLambda(lambda x: {
        "question": x["question"],
        "context": format_context(x["doc_scores"]),
        "sources": build_sources_list(x["doc_scores"]),
        # keep raw in case we want to show later (not used by the prompt)
        "_raw_doc_scores": x["doc_scores"],
    })

    prompt = build_prompt()
    model = ChatOpenAI(model=model_name, temperature=temperature)
    parser = StrOutputParser()

    chain = parallel_chain | to_prompt_inputs
    rag_chain = chain | (RunnableLambda(lambda x: {k: x[k] for k in ["question", "context", "sources"]})) | prompt | model | parser
    return chain, rag_chain  # chain returns doc_scores too; rag_chain returns final answer

# ---------- UI ----------
st.title("Decision Intelligence Platform for Manufacturing Defect Analysis (RAG)")

st.caption("LLM-powered root cause analysis using structured and unstructured manufacturing documents")


with st.sidebar:
    st.header("Settings")
    top_k = st.slider("TOP_K (docs to retrieve)", min_value=1, max_value=10, value=4, step=1)
    model_name = st.selectbox("Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-5"], index=0)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
    show_context = st.checkbox("Show retrieved context blocks", value=True)

#default_q = "Welding misalignment defects are increasing. What are common causes and what maintenance actions should we take?"

import streamlit as st

# --- Categorized sample questions (keep yours) ---
SAMPLE_QUESTIONS = {
    "Welding / Quality": [
        "Why are welding misalignment defects increasing?",
        "What are common indicators and root causes of welding misalignment?",
        "How can we reduce panel gap and flush issues in body assembly?"
    ],
    "Maintenance / Equipment": [
        "What maintenance actions reduce alignment drift over time?",
        "What happens if we skip calibration or delay preventive maintenance?",
        "Which robot or fixture issues most often lead to quality defects?"
    ],
    "Supplier / Materials": [
        "How do supplier batch variations contribute to defects?",
        "What material tolerance issues commonly cause misalignment?",
        "What controls can reduce variability from incoming panels?"
    ],
}

# Optional default question (keep if you want a starter prompt)
DEFAULT_Q = (
    "Welding misalignment defects are increasing. "
    "What are common causes and what maintenance actions should we take?"
)

# Keep one state variable for the active question
if "question" not in st.session_state:
    st.session_state.question = DEFAULT_Q

st.markdown("### Try a sample question (optional)")

# 1) Category dropdown
categories = ["— Select a category —"] + list(SAMPLE_QUESTIONS.keys())
selected_cat = st.selectbox("Category", categories, index=0)

# 2) Question dropdown (depends on category)
if selected_cat != categories[0]:
    qs = ["— Select a sample question —"] + SAMPLE_QUESTIONS[selected_cat]
    selected_q = st.selectbox("Sample question", qs, index=0)

    # Copy chosen sample into the main input box
    if selected_q != qs[0]:
        st.session_state.question = selected_q
else:
    st.selectbox("Sample question", ["Select a category first"], index=0, disabled=True)

# 3) Free-form question input (editable)
question = st.text_area("Or ask your own question", key="question", height=110)

#question = st.text_area("Question", value=default_q, height=110)

col1, col2 = st.columns([1, 1])
run_btn = col1.button("Run RAG", type="primary")
clear_btn = col2.button("Clear")

if clear_btn:
    st.session_state.pop("answer", None)
    st.session_state.pop("sources", None)
    st.session_state.pop("context", None)
    st.session_state.pop("doc_scores", None)
    st.rerun()

# Load DB once
try:
    db = load_vectorstore()
except Exception as e:
    st.error(f"Failed to load FAISS index from '{FAISS_DIR}'. Error: {e}")
    st.stop()

# Build chains
debug_chain, rag_chain = build_rag_chain(db, top_k=top_k, model_name=model_name, temperature=temperature)

if run_btn:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving + generating answer..."):
            # 1) Get doc_scores + prompt inputs
            debug_out = debug_chain.invoke(question)
            # debug_out has keys: question, doc_scores
            doc_scores = debug_out["_raw_doc_scores"]
            context_text = format_context(doc_scores)
            sources_text = build_sources_list(doc_scores)

            # 2) Final answer
            answer = rag_chain.invoke(question)

        st.session_state["answer"] = answer
        st.session_state["sources"] = sources_text
        st.session_state["context"] = context_text
        st.session_state["doc_scores"] = doc_scores

# ---------- Output ----------
if "answer" in st.session_state:
    left, right = st.columns([1.3, 1])

    with left:
        st.subheader("Answer")
        st.write(st.session_state["answer"])

        if show_context:
            st.subheader("Retrieved Context Blocks (debug)")
            st.text(st.session_state["context"])

    with right:
        st.subheader("Sources")
        st.text(st.session_state["sources"])

        st.subheader("Top Matches (metadata)")
        for i, (doc, score) in enumerate(st.session_state["doc_scores"], 1):
            md = doc.metadata or {}
            st.markdown(
                f"**[{i}] score={score:.4f}**  \n"
                f"- file: `{md.get('source_file')}`  \n"
                f"- domain: `{md.get('domain')}`  \n"
                f"- page: `{md.get('page')}`"
            )
