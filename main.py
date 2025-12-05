import os
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from vector_store import process_and_store_vectors

# 1. Setup Configuration
load_dotenv()
DB_DIR = "chroma_db_store"

# Define your strict categories (Must match folder names from Phase 2)
VALID_CATEGORIES = [
    "SOPs", 
    "HR_Manual", 
    "Technical_Specifications", 
    "Finance_Policy", 
    "Legal_Contracts"
]

# Initialize OpenAI Client
client = OpenAI()

# =======================================================
# CALL 1: THE CLASSIFIER (ROUTER)
# =======================================================
def classify_query(user_query):
    """
    Uses OpenAI to determine which category the query belongs to.
    """
    system_prompt = f"""
    You are an intelligent query router.
    You must classify the user's query into EXACTLY one of the following categories:
    {VALID_CATEGORIES}
    
    If the query is ambiguous, choose the most likely technical fit.
    Reply ONLY with the exact category name. Do not add punctuation or explanation.
    """

    response = client.chat.completions.create(
        model="gpt-4o", # or gpt-3.5-turbo
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        temperature=0 # Temperature 0 ensures deterministic/strict output
    )
    
    # Clean the output to ensure no whitespace/periods
    category = response.choices[0].message.content.strip()
    
    if category not in VALID_CATEGORIES:
        # Fallback if LLM hallucinates a new category
        print(f"[Warning] LLM predicted '{category}', which is invalid. Defaulting to SOPs.")
        return "SOPs"
        
    return category

# =======================================================
# CALL 2: RAG GENERATION (WITH FILTER)
# =======================================================
def retrieval_augmented_generation(user_query, category):
    """
    1. Retrieves docs ONLY from the specific category.
    2. Sends context + query to OpenAI for the final answer.
    """
    
    print(f"   > Searching Vector DB with Filter: {{'category': '{category}'}}")
    
    # Initialize Vector DB Connection (LAZY INITIALIZATION)
    # This prevents file locking issues by only opening the DB when needed
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
    
    # 1. RETRIEVAL (Not an API call, this is local Vector Search)
    # We apply the metadata filter here!
    results = vector_db.similarity_search(
        user_query,
        k=3,
        filter={"category": category} # <--- THIS IS THE MAGIC PART
    )
    
    if not results:
        return "No relevant documents found in this category."

    # Combine retrieved chunks into a context block
    context_text = "\n\n".join([doc.page_content for doc in results])
    
    # 2. GENERATION (The 2nd OpenAI Call)
    system_prompt = f"""
    You are a helpful assistant for Acme Corp.
    Answer the user's question using ONLY the provided context from the {category} documents.
    If the answer is not in the context, say "I don't know."
    
    Context:
    {context_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content

# =======================================================
# MAIN EXECUTION FLOW
# =======================================================
def main():
    # Get query from user input
    print("=" * 50)
    print("ACME CORP - INTELLIGENT DOCUMENT QUERY SYSTEM")
    print("=" * 50)
    query = input("\nEnter your query: ").strip()
    
    if not query:
        print("Error: Query cannot be empty.")
        return
    
    print(f"\nUser Query: {query}")
    print("-" * 50)

    # --- STEP 1: CLASSIFY ---
    print("1. Routing Query...")
    detected_category = classify_query(query)
    print(f"   > Detected Category: {detected_category}")
    
    # --- STEP 2: RAG WITH FILTER ---
    print("2. Performing Targeted RAG...")
    final_answer = retrieval_augmented_generation(query, detected_category)
    
    print("-" * 50)
    print("Final Answer:")
    print(final_answer)

if __name__ == "__main__":
    process_and_store_vectors() 
    main()