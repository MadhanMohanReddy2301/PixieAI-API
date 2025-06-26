# build_faiss_free.py

import os
import pickle
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss

# 🚀 Config
PDF_FOLDER = "pdfs/"           # folder containing PDF(s)
INDEX_FOLDER = "vectorstore/"  # where to save FAISS index
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def load_and_split_pdfs(pdf_dir):
    """Extract text and chunk into overlapping segments."""
    chunks, metadatas = [], []
    splitter = lambda text: [
        text[i : i + CHUNK_SIZE]
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP)
    ]
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        reader = PdfReader(path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        for chunk in splitter(text):
            chunks.append(chunk.strip())
            metadatas.append({"source": fname})
    return chunks, metadatas

def build_faiss(chunks, metadatas, index_dir):
    """Generate embeddings and build FAISS index."""
    os.makedirs(index_dir, exist_ok=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")  # free, open-source:contentReference[oaicite:1]{index=1}
    embeddings = model.encode(chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(index_dir, "faiss.index"))
    with open(os.path.join(index_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadatas, f)
    print(f"FAISS index built with {len(chunks)} chunks and saved to {index_dir}")

if __name__ == "__main__":
    chunks, metas = load_and_split_pdfs(PDF_FOLDER)
    print(f"Extracted {len(chunks)} text chunks from PDFs.")
    build_faiss(chunks, metas, INDEX_FOLDER)
