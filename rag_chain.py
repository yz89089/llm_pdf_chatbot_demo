import os
import openai
import faiss
import numpy as np
from dotenv import load_dotenv

# 加载 .env 中的 API Key
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

def get_text_embedding(text, model="text-embedding-ada-002"):
    return get_embedding(text, model=model)

def build_faiss_index(chunks, model="text-embedding-ada-002"):
    embeddings = [get_text_embedding(c, model=model) for c in chunks]
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def query_index(user_query, chunks, index, model="text-embedding-ada-002"):
    query_vec = get_text_embedding(user_query, model=model)
    D, I = index.search(np.array([query_vec]).astype("float32"), k=3)
    retrieved_chunks = [chunks[i] for i in I[0]]
    return "\n".join(retrieved_chunks)

def ask_llm(query, context):
    prompt = f"Answer the question based on the context:\n\nContext:\n{context}\n\nQuestion: {query}"
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
