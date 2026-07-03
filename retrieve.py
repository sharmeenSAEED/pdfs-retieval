"""
retrieve.py  —  Ask questions about your Hadith collection
Run this AFTER ingest.py has built the ChromaDB vector store.
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate




load_dotenv()

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL  = "llama-3.1-8b-instant"   # fast & free on Groq
TOP_K       = 4                         # how many chunks to retrieve per query
# ─────────────────────────────────────────────


# Custom prompt so the LLM answers in Hadith context
HADITH_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a knowledgeable Islamic scholar assistant.
Use ONLY the Hadith excerpts provided below to answer the question.
If the answer is not found in the excerpts, say "I could not find a relevant Hadith in the provided collection."
Do not fabricate or add information.

--- Hadith Context ---
{context}
----------------------

Question: {question}

Answer (cite the source file if possible):"""
)


def load_retriever():
    print(" Loading embedding model …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(" Loading ChromaDB vector store …")
    vectordb = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    count = vectordb._collection.count()
    if count == 0:
        raise RuntimeError("Vector store is empty. Run ingest.py first!")
    print(f"   {count} vectors loaded.\n")

    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def load_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found. Add it to your .env file.")
    return ChatGroq(
        groq_api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=0.1,        # low temperature = more faithful answers
    )


def build_qa_chain(retriever, llm):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",             # "stuff" = all chunks stuffed into one prompt
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": HADITH_PROMPT},
    )


def pretty_print_result(result):
    print("\n" + "─" * 55)
    print("📖  ANSWER:")
    print(result["result"])

    print("\n📚  SOURCE CHUNKS USED:")
    seen = set()
    for doc in result["source_documents"]:
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        key  = (src, page)
        if key not in seen:
            seen.add(key)
            print(f"\n  [{src}  —  page {page}]")
        print(f"  {doc.page_content[:300].strip()} …")
    print("─" * 55 + "\n")


if __name__ == "__main__":
    print("=" * 55)
    print("       HADITH RETRIEVER — Q&A INTERFACE")
    print("=" * 55 + "\n")

    retriever = load_retriever()
    llm       = load_llm()
    qa_chain  = build_qa_chain(retriever, llm)

    print(" Ready! Type your question or 'exit' to quit.\n")

    while True:
        query = input("❓ Your question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print(" Goodbye!")
            break
        try:
            result = qa_chain.invoke({"query": query})
            pretty_print_result(result)
        except Exception as e:
            print(f"  Error: {e}\n")
