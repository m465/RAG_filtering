import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Configuration
load_dotenv()
SOURCE_DIRECTORY = "all_docs"
DB_PERSIST_DIRECTORY = "chroma_db_store"

def process_and_store_vectors():
    print(f"--- STARTING RAG INGESTION FROM '{SOURCE_DIRECTORY}' ---")
    
    all_documents = []

    # 1. Walk through the directory to load files and TAG METADATA
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"Error: Directory '{SOURCE_DIRECTORY}' not found.")
        return

    for root, dirs, files in os.walk(SOURCE_DIRECTORY):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                
                # Extract Category from the folder name
                # Example: root = "all_docs/HR_Manual" -> category_name = "HR_Manual"
                category_name = os.path.basename(root)
                
                # Load the PDF
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                
                # INJECT METADATA: Add the category key to every page/doc loaded
                for doc in docs:
                    doc.metadata["category"] = category_name
                    doc.metadata["filename"] = file 
                
                all_documents.extend(docs)
                print(f"Loaded: {file} | Category: {category_name}")

    if not all_documents:
        print("No PDF documents found.")
        return

    print(f"\nTotal raw pages loaded: {len(all_documents)}")

    # 2. Chunk the documents (Size = 300)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,  # Slight overlap to keep context
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunked_docs)}")

    # Debug: Check the first chunk to ensure metadata is there
    if chunked_docs:
        print("\n[DEBUG] Sample Chunk Metadata:")
        print(chunked_docs[0].metadata)

    # 3. Create Embeddings & Store in ChromaDB
    print("\n--- GENERATING EMBEDDINGS & STORING ---")
    
    # Use a local HuggingFace model (free, runs on CPU)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Clear old database if it exists to prevent duplicate data during testing
    if os.path.exists(DB_PERSIST_DIRECTORY):
        shutil.rmtree(DB_PERSIST_DIRECTORY)

    # Store vectors
    vector_db = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embedding_model,
        persist_directory=DB_PERSIST_DIRECTORY
    )
    
    print(f"Success! Vector Store saved to '{DB_PERSIST_DIRECTORY}'")

if __name__ == "__main__":
    process_and_store_vectors()