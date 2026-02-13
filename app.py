import streamlit as st
import time
from rag_engine import RAGSystem

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Policy RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# HEADER
st.title("🤖 Policy RAG Assistant")
st.markdown("""
This AI assistant answers questions based on your internal policy documents.
*Powered by **Hybrid RAG** (Local Embeddings + Gemini 2.5) with Self-Correction.*
""")

# SIDEBAR
with st.sidebar:
    st.header("⚙️ Configuration")
    
    prompt_mode = st.radio(
        "Prompt Strategy:",
        ["Advanced (Structured)", "Basic (Simple)"],
        index=0,
        help="Switch between a sophisticated Persona-based prompt and a simple instruction."
    )
    
    version = "advanced" if "Advanced" in prompt_mode else "basic"
    
    st.divider()
    st.info("💡 **Tip:** 'Advanced' mode uses negative constraints and strict formatting.")

# INITIALIZE SYSTEM
# I use @st.cache_resource so the Vector DB isn't reloaded on every click.
@st.cache_resource
def load_rag_system():
    return RAGSystem()

try:
    with st.spinner("Initializing RAG Engine & Vector DB..."):
        rag = load_rag_system()
    st.success("System Ready!", icon="✅")
except Exception as e:
    st.error(f"Failed to load RAG System: {e}")
    st.stop()

# CHAT INTERFACE
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about the policy..."):
    # 1. Display User Message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing documents..."):
            # Call engine
            start_time = time.time()
            response = rag.query(prompt, version=version)
            latency = time.time() - start_time
            
            # Extract Data from JSON
            answer = response.get('answer', "Error generating response.")
            confidence = response.get('confidence', "Unknown")
            context_used = response.get('context_used', False)
            
            # Display Answer
            message_placeholder.markdown(answer)
            
            # 3. Show "Under the Hood" details
            with st.expander("🔍 See Retrieval Details"):
                st.markdown(f"**Latency:** {latency:.2f} seconds")
                st.markdown(f"**Confidence Score:** `{confidence}`")
                st.markdown(f"**Prompt Mode:** `{version}`")
                
                # Display the actual text chunks
                sources = response.get('source_documents', [])
                
                if context_used and sources:
                    st.markdown("### 📄 Retrieved Context Sources")
                    for i, source_text in enumerate(sources):
                        # Use a little box for each source
                        st.info(f"**Chunk {i+1}:**\n\n{source_text[:300]}...", icon="📎")
                
                elif context_used:
                     st.info("Context was used, but source text is not available.")
                else:
                    st.warning("No relevant context found in the database.")

    # 4. Save to History
    st.session_state.messages.append({"role": "assistant", "content": answer})