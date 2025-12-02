from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

st.set_page_config(page_title="AI Podcast Analyzer", layout="wide")
st.title("🎙️ AI Podcast Analyzer")

# Sidebar for quick podcast input
with st.sidebar:
    st.header("🎧 Podcast Input")
    video_id = st.text_input("YouTube Video ID")
    process = st.button("Process Transcript")

youtube = YouTubeTranscriptApi()

if process and video_id:
    with st.spinner("Processing transcript…"):
        try:
            transcript = youtube.fetch(video_id)
            full_text = " ".join(chunk.text for chunk in transcript)

            splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=100)
            docs = splitter.split_text(full_text)
            documents = [Document(page_content=d) for d in docs]

            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            
            pc = Pinecone(api_key=PINECONE_API_KEY)
            index_name = "youtube-podcast"

            if index_name not in pc.list_indexes():
                pc.create_index(
                    name=index_name,
                    dimension=3072,         
                    metric="dotproduct",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            
            index = pc.Index(index_name)

            bm25 = BM25Encoder.default()
            bm25.fit([d.page_content for d in documents])
            bm25.dump("bm25.json")

            bm25_loader = BM25Encoder.default()
            bm25 = bm25_loader.load(path="bm25.json")

            retriever = PineconeHybridSearchRetriever(
                embeddings=embeddings,
                sparse_encoder=bm25,
                index=index
            )

            # index texts (simple)
            retriever.add_texts([doc.page_content for doc in documents])

            st.session_state["retriever"] = retriever
            st.success("Transcript indexed successfully!")

        except Exception as e:
            st.error(f"Error: {e}")

# Main area: ask questions and produce outputs
question = st.text_input("Ask a question about the podcast:")

# Buttons panel for different output formats
col1, col2, col3, col4 = st.columns([1,1,1,1])
with col1:
    get_answer_btn = st.button("Get Answer")
with col2:
    blog_btn = st.button("Generate A Short Blog ")
with col3:
    newsletter_btn = st.button("Newsletter-ready")
with col4:
    top10_btn = st.button("Top 10 key points")

# Export area (stores last generated text)
if "last_output" not in st.session_state:
    st.session_state["last_output"] = ""

# Minimal LLM helper (uses same LLM across actions)
def run_llm(prompt_text, model_name="gpt-3.5-turbo"):
    llm = ChatOpenAI(model=model_name)
    out = llm.invoke(prompt_text)
    return out.content

# Utility: fetch context from retriever (concise)
def get_context_for_question(q):
    retriever = st.session_state.get("retriever")
    if not retriever:
        return ""
    # use a retrieval call appropriate for retriever
    try:
        docs = retriever.get_relevant_documents(q)
    except Exception:
        # fallback if method name differs
        docs = retriever.invoke(q) if hasattr(retriever, "invoke") else []
    return " ".join(getattr(d, "page_content", str(d)) for d in docs)

# Standard prompt (no timestamp requirement; concise answer)
answer_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. Use the context below to answer the question briefly.\n\n"
        "Context:\n{context}\n\nQuestion: {question}\n\nAnswer (concise, to the point):"
    )
)

# Blog prompt (<=100 tokens)
blog_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Write a short blog-style answer (less than 100 tokens) using the context below. "
        "Make it engaging and shareable.\n\nContext:\n{context}\n\nPrompt: {question}\n\nBlog:"
    )
)

# Newsletter prompt
newsletter_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Create a newsletter-ready paragraph (approx 3-4 sentences) summarizing the answer using the context below.\n\n"
        "Context:\n{context}\n\nPrompt: {question}\n\nNewsletter:"
    )
)

# Top 10 key points prompt
top10_prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "From the context below, extract the Top 10 key points relevant to the question. Return a numbered list.\n\n"
        "Context:\n{context}\n\nPrompt: {question}\n\nTop 10:"
    )
)

# Handling clicks
if get_answer_btn or blog_btn or newsletter_btn or top10_btn:
    if question.strip() == "":
        st.warning("Please enter a question first.")
    else:
        context = get_context_for_question(question)
        if not context:
            st.warning("No indexed transcript found. Please Process Transcript first.")
        else:
            if get_answer_btn:
                final_prompt = answer_prompt_template.format(context=context, question=question)
                with st.spinner("Generating answer…"):
                    out = run_llm(final_prompt)
                st.session_state["last_output"] = out
                st.chat_message("assistant").write(out)

            if blog_btn:
                final_prompt = blog_prompt_template.format(context=context, question=question)
                with st.spinner("Generating short blog…"):
                    out = run_llm(final_prompt)
                st.session_state["last_output"] = out
                st.chat_message("assistant").write(out)

            if newsletter_btn:
                final_prompt = newsletter_prompt_template.format(context=context, question=question)
                with st.spinner("Generating newsletter…"):
                    out = run_llm(final_prompt)
                st.session_state["last_output"] = out
                st.chat_message("assistant").write(out)

            if top10_btn:
                final_prompt = top10_prompt_template.format(context=context, question=question)
                with st.spinner("Extracting Top 10…"):
                    out = run_llm(final_prompt)
                st.session_state["last_output"] = out
                st.chat_message("assistant").write(out)

# Optional Export button: tries to make a simple PDF, falls back to TXT
export_col1, export_col2 = st.columns([1,3])
with export_col1:
    if st.button("Export last output"):
        data = st.session_state.get("last_output", "")
        if not data:
            st.warning("No output to export yet.")
        else:
            # try to create a tiny PDF using fpdf if available
            try:
                from fpdf import FPDF
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.set_font("Arial", size=12)
                for line in data.split("\n"):
                    pdf.multi_cell(0, 8, line)
                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                st.download_button("Download PDF", data=pdf_bytes, file_name="podcast_output.pdf", mime="application/pdf")
            except Exception:
                # fallback: download as txt
                st.download_button("Download TXT", data=data, file_name="podcast_output.txt", mime="text/plain")

# show last output in a small box for quick reference
if st.session_state.get("last_output"):
    st.markdown("**Last generated output:**")
    st.write(st.session_state["last_output"])
