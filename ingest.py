"""
ingest.py  —  Load Hadith PDFs → Chunk → Embed → Store in ChromaDB
Run this ONCE (or whenever you add new PDFs to your hadith_pdfs/ folder).
"""

import os
import glob
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

# ─────────────────────────────────────────────
#  CONFIGURATION  (tweak these as needed)
# ─────────────────────────────────────────────
PDF_FOLDER    = r"hadees"        # folder where your Hadith PDFs live
CHROMA_DIR    = "./chroma_db"          # where the vector store will be saved

# Chunking settings
CHUNK_SIZE    = 512     # characters per chunk  (good balance for Hadith text)
CHUNK_OVERLAP = 64      # overlap so context isn't lost at boundaries

# Embedding model (free, runs locally, no API key needed)
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# ─────────────────────────────────────────────


def load_pdfs(folder: str):
    """Load every PDF in the given folder."""
    pdf_paths = glob.glob(os.path.join(folder, "**", "*.pdf"), recursive=True)
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in '{folder}'. "
                                "Make sure your Hadith PDFs are placed there.")
    print(f" Found {len(pdf_paths)} PDF(s):")
    for p in pdf_paths:
        print(f"   • {p}")

    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Tag each page with its source file for later citation
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    print(f"\n Loaded {len(all_docs)} pages total.\n")
    return all_docs


def chunk_documents(docs):
    """
    Split documents into overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on:
      1. Paragraph breaks  (\n\n)
      2. Newlines          (\n)
      3. Spaces
      4. Characters
    …in that order, so Hadith sentences/paragraphs stay intact where possible.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"✂️  Created {len(chunks)} chunks "
          f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).\n")
    return chunks


def build_vectorstore(chunks):
    """Embed chunks and persist to ChromaDB."""
    print(f" Loading embedding model: {EMBED_MODEL} …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},      # change to "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},
    )

    print(" Embedding & storing in ChromaDB — this may take a few minutes …\n")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    print(f" Vector store saved to '{CHROMA_DIR}'.")
    print(f"   Total vectors stored: {vectordb._collection.count()}\n")
    return vectordb


if __name__ == "__main__":
    print("=" * 55)
    print("       HADITH RETRIEVER — INGESTION PIPELINE")
    print("=" * 55 + "\n")

    docs   = load_pdfs(PDF_FOLDER)
    chunks = chunk_documents(docs)
    build_vectorstore(chunks)

    print(" Ingestion complete! Now run  retrieve.py  to ask questions.")
