import fitz  # PyMuPDF

def extract_text_chunks_from_pdf(pdf_path, chunk_size=500):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    
    # 简单按字符数分块（后期可优化为按段落）
    chunks = [full_text[i:i+chunk_size] for i in range(0, len(full_text), chunk_size)]
    return chunks
