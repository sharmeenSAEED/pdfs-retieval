# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# import os

# print("\nStep 1: Loading PDF...\n")

# # 🔴 FIX THIS PATH TO YOUR ACTUAL FILE LOCATION
# loader = PyPDFLoader(r"hadees.pdf")
# documents = loader.load()

# print(f"Pages loaded: {len(documents)}")

# print("\nStep 2: Splitting text...\n")

# splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# chunks = splitter.split_documents(documents)

# print(f"Chunks created: {len(chunks)}")

# print("\nStep 3: Creating embeddings...\n")

# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )

# db_path = "./chroma_db"

# print("\nStep 4: Saving / Loading ChromaDB...\n")

# if os.path.exists(db_path) and os.listdir(db_path):
#     vectordb = Chroma(
#         persist_directory=db_path,
#         embedding_function=embeddings
#     )
# else:
#     vectordb = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=db_path
#     )

# retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# print("\n✅ READY! You can now search the PDF.\n")

# # 🔥 SEARCH LOOP (NO AI)
# while True:
#     query = input("\nAsk something (or type exit): ")

#     if query.lower() == "exit":
#         break

#     results = retriever.invoke(query)

#     print("\n📄 Relevant PDF chunks:\n")

#     for i, doc in enumerate(results, 1):
#         print(f"--- Chunk {i} ---")
#         print(doc.page_content[:500])
#         print()