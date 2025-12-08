import os
import json
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# Configuration
load_dotenv()
SOURCE_DIRECTORY = "all_docs"
DB_PERSIST_DIRECTORY = "chroma_db_store"
MANIFEST_FILE = os.path.join(DB_PERSIST_DIRECTORY, "processed_manifest.json")

def load_manifest():
    """Load the manifest of previously processed files."""
    if os.path.exists(MANIFEST_FILE):
        try:
            with open(MANIFEST_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load manifest: {e}")
            return {}
    return {}

def save_manifest(manifest):
    """Save the manifest of processed files."""
    try:
        os.makedirs(DB_PERSIST_DIRECTORY, exist_ok=True)
        with open(MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save manifest: {e}")

def get_file_info(file_path):
    """Get file modification time and size."""
    stat = os.stat(file_path)
    return {
        "mtime": stat.st_mtime,
        "size": stat.st_size
    }

def process_and_store_vectors(force_regenerate=False):
    print(f"--- STARTING RAG INGESTION FROM '{SOURCE_DIRECTORY}' ---")
    
    # If force regenerate, rename existing database instead of deleting
    if force_regenerate:
        print("\n🔄 Force regeneration enabled - archiving existing database...")
        if os.path.exists(DB_PERSIST_DIRECTORY):
            # Rename instead of delete to avoid file locking issues
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{DB_PERSIST_DIRECTORY}_backup_{timestamp}"
            
            try:
                os.rename(DB_PERSIST_DIRECTORY, backup_dir)
                print(f"✓ Archived existing database to: {backup_dir}")
                print("  (You can delete backup folders manually later)")
            except Exception as e:
                print(f"⚠️ Could not archive database: {e}")
                print("  Proceeding with fresh database anyway...")
    
    # 1. Check if source directory exists
    if not os.path.exists(SOURCE_DIRECTORY):
        print(f"Error: Directory '{SOURCE_DIRECTORY}' not found.")
        return

    # 2. Load manifest of previously processed files
    manifest = load_manifest()
    
    # 3. Scan for files and determine which need processing
    files_to_process = []
    current_files = {}
    
    for root, dirs, files in os.walk(SOURCE_DIRECTORY):
        for file in files:
            if file.endswith(".pdf"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, SOURCE_DIRECTORY)
                
                # Get current file info
                file_info = get_file_info(file_path)
                current_files[relative_path] = file_info
                
                # Check if file needs processing
                if relative_path not in manifest:
                    # New file
                    files_to_process.append(file_path)
                    print(f"[NEW] {file}")
                elif (manifest[relative_path]["mtime"] != file_info["mtime"] or 
                      manifest[relative_path]["size"] != file_info["size"]):
                    # Modified file
                    files_to_process.append(file_path)
                    print(f"[MODIFIED] {file}")
                else:
                    # Already processed, skip
                    print(f"[SKIP] {file} (already processed)")
    
    # 4. If no files need processing, exit early
    if not files_to_process:
        print("\n✓ All documents are already up-to-date in the vector store.")
        print(f"Total files tracked: {len(current_files)}")
        return
    
    print(f"\n→ Processing {len(files_to_process)} new/modified file(s)...")
    
    # 5. Load and process only new/modified files
    all_documents = []
    
    for file_path in files_to_process:
        # Extract Category from the folder name
        root = os.path.dirname(file_path)
        category_name = os.path.basename(root)
        file = os.path.basename(file_path)
        
        # Load the PDF
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # INJECT METADATA: Add the category key to every page/doc loaded
        for doc in docs:
            doc.metadata["category"] = category_name
            doc.metadata["filename"] = file
            doc.metadata["file_path"] = os.path.relpath(file_path, SOURCE_DIRECTORY)
        
        all_documents.extend(docs)
        print(f"Loaded: {file} | Category: {category_name} | Pages: {len(docs)}")

    if not all_documents:
        print("No documents to process.")
        return

    print(f"\nTotal raw pages loaded: {len(all_documents)}")

    # 6. Chunk the documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunked_docs = text_splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunked_docs)}")

    # Debug: Check the first chunk to ensure metadata is there
    if chunked_docs:
        print("\n[DEBUG] Sample Chunk Metadata:")
        print(chunked_docs[0].metadata)

    # 7. Create Embeddings & Store in ChromaDB
    print("\n--- GENERATING EMBEDDINGS & STORING ---")
    
    # Use a local HuggingFace model (free, runs on CPU)
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # Check if database exists (only matters if NOT force regenerating)
    db_exists = os.path.exists(DB_PERSIST_DIRECTORY) and not force_regenerate
    
    if db_exists:
        # Add to existing vector store (incremental update)
        print("→ Adding new vectors to existing database...")
        vector_db = Chroma(
            persist_directory=DB_PERSIST_DIRECTORY,
            embedding_function=embedding_model
        )
        vector_db.add_documents(chunked_docs)
    else:
        # Create new vector store (fresh database)
        print("→ Creating new vector database...")
        vector_db = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embedding_model,
            persist_directory=DB_PERSIST_DIRECTORY
        )
    
    # 8. Update manifest with all current files
    save_manifest(current_files)
    
    print(f"\n✓ Success! Vector Store saved to '{DB_PERSIST_DIRECTORY}'")
    print(f"✓ Processed {len(files_to_process)} file(s)")
    print(f"✓ Total files tracked: {len(current_files)}")

if __name__ == "__main__":
    process_and_store_vectors()