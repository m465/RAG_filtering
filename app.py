import streamlit as st
import os
from dotenv import load_dotenv
from main import RAGChatBot, VALID_CATEGORIES, DB_DIR
from vector_store import process_and_store_vectors

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="ACME Corp - Document Query System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat container */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* User message */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .stChatMessage[data-testid="user-message"] * {
        color: purple !important;
    }
    
    /* Assistant message */
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-left: 4px solid purple !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] * {
        color: purple !important;
    }
    
    /* Target the markdown content specifically */
    .stChatMessage .stMarkdown {
        color: purple !important;
    }
    
    .stChatMessage .stMarkdown * {
        color: purple !important;
    }
    
    /* Spinner text */
    .stSpinner > div {
        color: purple !important;
    }
    
    .stSpinner * {
        color: purple !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.95);
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-weight: 700;
    }
    
    /* Input box */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid purple;
        padding: 10px 20px;
    }
    
    /* Buttons */
    .stButton > button {
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Category badges */
    .category-badge {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 0.85em;
        font-weight: 600;
        margin: 5px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: green;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid green;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory_type" not in st.session_state:
    st.session_state.memory_type = "top_k"

if "bot" not in st.session_state:
    st.session_state.bot = RAGChatBot(memory_type=st.session_state.memory_type)

if "documents_processed" not in st.session_state:
    st.session_state.documents_processed = False

# Main UI
st.title("🤖 ACME Corp - Intelligent Document Query System")

# Sidebar
with st.sidebar:
    st.header("📚 Document Categories")
    st.markdown("The system can answer questions from:")
    for category in VALID_CATEGORIES:
        st.markdown(f'<div class="category-badge">{category.replace("_", " ")}</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.header("🧠 Memory Settings")
    
    # Memory type selector
    memory_type = st.selectbox(
        "Memory Type",
        options=["top_k", "summary"],
        index=0 if st.session_state.memory_type == "top_k" else 1,
        help="**top_k**: Only last 5 messages sent to API\n\n**summary**: Summary + last 5 messages sent to API"
    )
    
    # Update memory type if changed
    if memory_type != st.session_state.memory_type:
        st.session_state.memory_type = memory_type
        # Reinitialize bot with new memory type
        st.session_state.bot = RAGChatBot(memory_type=memory_type)
        st.success(f"✅ Memory type updated to: {memory_type}")
    
    st.markdown("---")
    
    st.header("⚙️ System Controls")
    
    # Process documents button
    if st.button("🔄 Process Documents", use_container_width=True):
        with st.spinner("Processing documents... This may take a moment."):
            try:
                # Clear any cached database connections
                if "bot" in st.session_state:
                    st.session_state.bot = None
                
                # Force garbage collection to close database connections
                import gc
                gc.collect()
                
                # Force regenerate - delete old database and create new one
                process_and_store_vectors(force_regenerate=True)
                
                # Reinitialize the bot with fresh database and current memory type
                st.session_state.bot = RAGChatBot(memory_type=st.session_state.memory_type)
                st.session_state.documents_processed = True
                st.success("✅ Documents processed successfully!")
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.bot = RAGChatBot(memory_type=st.session_state.memory_type)
        st.rerun()
    
    st.markdown("---")
    
    # Status indicator
    st.header("📊 System Status")
    if os.path.exists(DB_DIR):
        st.success("✅ Vector Database: Ready")
    else:
        st.warning("⚠️ Vector Database: Not initialized")
        st.info("Click 'Process Documents' to initialize the system.")

# Main chat area
st.markdown('<div class="info-box">💡 <strong>Tip:</strong> Ask questions about your documents and I\'ll search the relevant category automatically!</div>', unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "category" in message:
            st.caption(f"📁 Category: {message['category'].replace('_', ' ')}")

# Chat input
if prompt := st.chat_input("Ask me anything about your documents..."):
    # Check if documents are processed
    if not os.path.exists(DB_DIR):
        st.error("⚠️ Please process documents first using the sidebar button!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating response..."):
                try:
                    response, category = st.session_state.bot.retrieval_augmented_generation(prompt)
                    st.markdown(response)
                    st.caption(f"📁 Category: {category.replace('_', ' ')}")
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "category": category
                    })
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })

# Footer
st.markdown("---")
st.markdown(
    '<div style="text-align: center; color: white; padding: 20px;">'
    '🚀 Powered by OpenAI GPT-4 & ChromaDB | Built with Streamlit'
    '</div>', 
    unsafe_allow_html=True
)
