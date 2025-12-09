import os
import string
from collections import deque
from openai import OpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# 1. Setup Configuration
load_dotenv()
DB_DIR = "chroma_db_store"
SOURCE_DIR = "all_docs"

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
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

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

        self.vector_db = Chroma(persist_directory=DB_DIR, embedding_function=embedding_function)
        # Initialize BM25 (In-Memory Keyword Search)
        print("--- INITIALIZING HYBRID RETRIEVER ---")
        self.bm25_retriever = self._build_bm25_index()

    def _build_bm25_index(self):
        """
        Loads all PDFs to build the Keyword Index (BM25).
        In production, you would save this index to disk, but for this demo, 
        building it at startup is fine.
        """
        print("   [System] Loading documents for Keyword Index (BM25)...")
        all_docs = []
        
        if not os.path.exists(SOURCE_DIR):
            print("   [Error] Source directory not found. Run generation script first.")
            return None

        for root, _, files in os.walk(SOURCE_DIR):
            for file in files:
                if file.endswith(".pdf"):
                    file_path = os.path.join(root, file)
                    category = os.path.basename(root)
                    
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["category"] = category
                        doc.metadata["filename"] = file
                    
                    all_docs.extend(docs)
        
        # Chunking (Must match the logic used for Vector DB)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunked_docs = text_splitter.split_documents(all_docs)
        
        print(f"   [System] Built BM25 Index with {len(chunked_docs)} chunks.")
        return BM25Retriever.from_documents(chunked_docs)
    

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
        2. HR_Manual: Employee conduct, Holidays, Leave policy, Benefits, Dress code, Standard Compensation, Bonuses, Legacy clauses, Grandfathered provisions, Retention schemes.
        3. Technical_Specifications: IT Systems, Software Architecture, Servers, Kubernetes, APIs, Databases (PostgreSQL), Cloud infrastructure, System Logs, Error Codes.
        4. Finance_Policy: Reimbursements, Expenses, Concur, Vendor payments, Procurement.
        5. Legal_Contracts: External NDAs, Terms of Service, Liability, Lawsuits, Vendor Contracts. (NOTE: Internal employee policy clauses belong to HR_Manual, not here).
        """
        
        system_prompt = f"""
        You are a strict query router. 
        Your goal is to map the user's question to the correct document repository based on the definitions below.
        
        {category_definitions}
        
        VALID CATEGORIES: {VALID_CATEGORIES}
        
        Rules:
        - If the query mentions a 'Clause' related to benefits, bonuses, or internal policy, it is 'HR_Manual'.
        - 'Legal_Contracts' is primarily for EXTERNAL agreements (NDAs, Terms of Service).
        - If the query is a specific code (like 'CLAUSE-882' or 'ERR-7719'), infer the category based on the format:
            - 'ERR-' or 'SYS-' usually implies Technical_Specifications.
            - 'CLAUSE-' usually implies HR_Manual (if internal) or Legal (if external). Default to HR_Manual if ambiguous.
        
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
    # HYBRID RETRIEVAL LOGIC
    # =======================================================
    def hybrid_search(self, query, category, k=5):
        print(f"   [Hybrid Search] Query: '{query}' | Category: '{category}'")
        
        # 1. VECTOR SEARCH (Semantic) - Native Filtering
        vector_docs = self.vector_db.similarity_search(
            query, k=k, filter={"category": category}
        )
        print(f"     -> Vector found {len(vector_docs)} results.")

        # 2. KEYWORD SEARCH (BM25) - Manual Filtering
        # BM25 returns top results from *all* docs, so we fetch more (k*2) and filter manually
        if self.bm25_retriever is None:
            print("     -> BM25 not available, using vector search only.")
            bm25_docs_filtered = []
        else:
            bm25_docs_raw = self.bm25_retriever.invoke(query)
            
            # Filter BM25 results to match the requested category
            bm25_docs_filtered = [
                doc for doc in bm25_docs_raw 
                if doc.metadata.get("category") == category
            ]
            print(f"     -> BM25 found {len(bm25_docs_filtered)} valid results (after filtering).")

        # 3. ENSEMBLE (Reciprocal Rank Fusion - RRF)
        # RRF gives better ranking than simple interleaving by considering both retrievers' scores
        print(f"     -> Applying Reciprocal Rank Fusion...")
        
        # Build document map for quick lookup
        doc_map = {}
        for doc in vector_docs + bm25_docs_filtered:
            doc_map[doc.page_content] = doc
        
        # Calculate RRF scores
        rrf_scores = {}
        k_constant = 60  # Standard RRF constant
        
        # Score vector search results
        for rank, doc in enumerate(vector_docs):
            content = doc.page_content
            rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (k_constant + rank + 1)
        
        # Score BM25 results
        for rank, doc in enumerate(bm25_docs_filtered):
            content = doc.page_content
            rrf_scores[content] = rrf_scores.get(content, 0) + 1 / (k_constant + rank + 1)
        
        # Sort by RRF score (highest first)
        sorted_contents = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Reconstruct document list in score order
        combined_docs = [doc_map[content] for content, score in sorted_contents]
        
        print(f"     -> RRF combined {len(combined_docs)} unique documents.")
        
        # Return top k
        return combined_docs[:k]

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
        
        
        # 1. RETRIEVAL (Not an API call, this is local Vector Search)
        # We apply the metadata filter here!
        results = self.hybrid_search(standalone_query, category, k=5)
        
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
