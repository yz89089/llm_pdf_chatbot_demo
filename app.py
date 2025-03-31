import streamlit as st
from pdf_utils import extract_text_chunks_from_pdf
from rag_chain import build_faiss_index, query_index, ask_llm

st.title("📄 LLM PDF Chatbot")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

if pdf_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(pdf_file.read())
    
    st.success("PDF uploaded!")
    st.write("Extracting text and building index...")

    chunks = extract_text_chunks_from_pdf("uploaded.pdf")
    index, _ = build_faiss_index(chunks)

    query = st.text_input("Ask a question based on the document:")
    if query:
        context = query_index(query, chunks, index)
        answer = ask_llm(query, context)
        st.markdown(f"**Answer:** {answer}")
