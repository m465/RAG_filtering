import os
import string
from collections import deque
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
    def __init__(self, memory_type="top_k"):
        """
        Initialize RAG ChatBot with memory management.
        
        Args:
            memory_type (str): Either "top_k" (only last 5 messages) or "summary" (summary + last 5 messages)
        """
        self.chat_history = []  # Stores tuples: (User, AI) - Only keeps last 5
        self.summary = ""       # Stores the summary of everything before the last 5
        self.max_history_len = 5
        self.memory_type = memory_type  # "top_k" or "summary"

    def manage_history(self):
        """
        Checks if history exceeds 5 turns. 
        If so, pops the oldest turn and merges it into the running summary.
        """
        if len(self.chat_history) > self.max_history_len:
            # 1. Pop the oldest interaction
            oldest_interaction = self.chat_history.pop(0) 
            user_text, ai_text = oldest_interaction
            
            # 2. Update the Summary using OpenAI
            print("   [Memory] Compressing old history into summary...")
            
            prompt = f"""
            You are a memory manager. 
            Current Summary of conversation: "{self.summary}"
            
            Newest Old Interaction to merge:
            User: {user_text}
            AI: {ai_text}
            
            Task: Update the Current Summary to include the key information from the Newest Old Interaction. 
            Keep the summary concise. Do not lose important details like names, numbers, or specific machinery discussed.
            """
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": "You are a helpful summarizer."},
                          {"role": "user", "content": prompt}]
            )
            
            # 3. Save new summary
            self.summary = response.choices[0].message.content.strip()
            print(f"   [Memory] Summary Updated. (History Len: {len(self.chat_history)})")

    
    def rephrase_query(self, user_query):
        # 1. OPTIMIZATION: Check if there is any context at all.
        # If both history and summary are empty, the user's query MUST be treated as standalone.
        if not self.chat_history and not self.summary:
            print("   [Rephraser] First query of session. Skipping rephrase step.")
            return user_query

        # 2. Format the recent history
        # We don't need the "No recent conversation" else block anymore 
        # because the 'if' check above handles the empty case.
        recent_history_str = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in self.chat_history])

        # 3. Define System Instructions
        system_instruction = """
        You are an intelligent query clarifier. 
        Your job is to rewrite the 'Follow-up input' into a STANDALONE QUESTION.
        
        Rules:
        1. Use the 'Context Summary' and 'Recent Chat History' to resolve pronouns (it, that, he, she).
        2. If the user input is already clear, return it as is.
        3. Do NOT answer the question. Just rewrite it.
        """

        # 4. Construct User Content
        user_content = f"""
        --- CONTEXT SUMMARY (Older Conversations) ---
        {self.summary if self.summary else "No summary available."}

        --- RECENT CHAT HISTORY (Last 5 Messages) ---
        {recent_history_str}

        --- FOLLOW-UP INPUT ---
        {user_query}

        Standalone Question:
        """

        # 5. Make the Call
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

    def classify_query(self, standalone_query):
        """
        Uses OpenAI to determine which category the query belongs to.
        Enhanced with category definitions for better routing accuracy.
        """
        category_definitions = """
        1. SOPs: Physical machinery (Hydraulic Press), Safety procedures, Emergency stops, Daily operational workflows, Floor management.
        2. HR_Manual: Employee conduct, Holidays, Leave policy, Benefits, Dress code.
        3. Technical_Specifications: IT Systems, Software Architecture, Servers, Kubernetes, APIs, Databases (PostgreSQL), Cloud infrastructure.
        4. Finance_Policy: Reimbursements, Expenses, Concur, Vendor payments, Procurement.
        5. Legal_Contracts: NDAs, Terms of Service, Liability, Lawsuits.
        """
        
        system_prompt = f"""
        You are a strict query router. 
        Your goal is to map the user's question to the correct document repository based on the definitions below.
        
        {category_definitions}
        
        VALID CATEGORIES: {VALID_CATEGORIES}
        
        Rules:
        - If the query is about PHYSICAL MACHINERY (like a 'Press' or 'Button'), it belongs to 'SOPs', NOT Technical_Specifications.
        - 'Technical_Specifications' is ONLY for Software/IT topics.
        
        Return ONLY the category name.
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
        if standalone_query.lower() != user_query.lower():
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
        # Build system prompt based on memory_type
        if self.memory_type == "summary":
            # Include summary + last 5 messages
            recent_history_str = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in self.chat_history])
            system_prompt = f"""
            You are a helpful assistant for Acme Corp.
            
            Previous Context Summary: {self.summary if self.summary else "No previous summary."}
            
            Recent Conversation (Last 5 messages):
            {recent_history_str if recent_history_str else "No recent conversation."}
            
            Answer the question using ONLY the retrieved context below.
            Context:
            {context_text}
            """
        else:  # memory_type == "top_k"
            # Only include last 5 messages (no summary)
            recent_history_str = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in self.chat_history])
            system_prompt = f"""
            You are a helpful assistant for Acme Corp.
            
            Recent Conversation (Last 5 messages):
            {recent_history_str if recent_history_str else "No recent conversation."}
            
            Answer the question using ONLY the retrieved context below.
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
        
        # Save to history (deque automatically keeps only last 5)
        self.chat_history.append((user_query, final_answer))
        self.manage_history()
        
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
