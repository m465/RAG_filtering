import os
import string
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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
# THE RAG CHATBOT CLASS
# =======================================================
class RAGChatBot:
    def __init__(self):
        self.chat_history = []
    
    def rephrase_query(self, user_query):
        """
        If history exists, rephrase the new query to include context from previous turns.
        """
        if not self.chat_history:
            return user_query # No history, no need to rephrase

        # Format history into a string
        history_str = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in self.chat_history])

        system_prompt = """
        Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question.
        The standalone question should be complete and specific, including necessary context from the conversation history.
        
        Example:
        History: 
        User: How much is dinner reimbursement?
        AI: It is capped at $100.
        Follow-up: What about for lunch?
        
        Standalone Question: What is the reimbursement limit for lunch?
        
        Do NOT answer the question. Just rephrase it.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Chat History:\n{history_str}\n\nFollow-up input: {user_query}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()

    def classify_query(self, standalone_query):
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
            model="gpt-4o-mini", # or gpt-4.1-nano
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": standalone_query}
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
    # RAG GENERATION (WITH FILTER)
    # =======================================================
    def retrieval_augmented_generation(self, user_query):
        """
        1. Retrieves docs ONLY from the specific category.
        2. Sends context + query to OpenAI for the final answer.
        Returns: (final_answer, category)
        """
        
         # 1. Rephrase
        standalone_query = self.rephrase_query(user_query)
        if standalone_query != user_query:
            print(f"   [Rephraser] Updated to: '{standalone_query}'")
        else:
            print(f"   [Rephraser] Kept original.")

        # 2. Route
        category = self.classify_query(standalone_query)
        print(f"   > Searching Vector DB with Filter: {{'category': '{category}'}}")
        
        # Initialize Vector DB Connection (LAZY INITIALIZATION)
        # This prevents file locking issues by only opening the DB when needed
        # IMPORTANT: Must use same embedding model as vector_store.py (HuggingFace)
        embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        
        # 1. RETRIEVAL (Not an API call, this is local Vector Search)
        # We apply the metadata filter here!
        results = vector_db.similarity_search(
            standalone_query,
            k=3,
            filter={"category": category} # <--- THIS IS THE MAGIC PART
        )
        
        if not results:
            return "No relevant documents found in this category.", category

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
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": standalone_query}
            ]
        )

        final_answer = response.choices[0].message.content
        
        # Save to history
        self.chat_history.append((user_query, final_answer))
        
        return final_answer, category

# =======================================================
# CLI EXECUTION (Optional - for testing)
# =======================================================
def main():
    """Command-line interface for testing the RAG chatbot."""
    from vector_store import process_and_store_vectors
    
    bot = RAGChatBot()
    # Get query from user input
    print("=" * 50)
    print("ACME CORP - INTELLIGENT DOCUMENT QUERY SYSTEM")
    print("=" * 50)
    
    # Define exact commands that trigger an exit
    EXIT_COMMANDS = {"exit", "quit", "stop", "terminate", "bye"}

    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if not query:
                print("Error: Query cannot be empty.")
                return
            
            print(f"\nUser Query: {query}")
            print("-" * 50)
            
            # 1. Check for Exit Conditions
            if query.lower().strip(string.punctuation) in EXIT_COMMANDS:
                print("\n[System] Terminating session. Goodbye!")
                break

            # 2. Perform RAG with Filter
            print("2. Performing Targeted RAG...")
            final_answer, category = bot.retrieval_augmented_generation(query)
            
            print("-" * 50)
            print("Final Answer:")
            print(final_answer)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n\n[System] Force exit detected. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error] An unexpected error occurred: {e}")


if __name__ == "__main__":
    from vector_store import process_and_store_vectors
    process_and_store_vectors() 
    main()
