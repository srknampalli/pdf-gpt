import os
import tempfile
import streamlit as st
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import MarkdownElementNodeParser
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def create_new_index():
    embedding_dim = 1536
    faiss_index = faiss.IndexFlatL2(embedding_dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex([], storage_context=storage_context)
    return index, storage_context

def process_uploaded_files(uploaded_files, parser, llm, index, storage_context):
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tf:
            tf.write(uploaded_file.getbuffer())
            temp_pdf_path = tf.name
        
        try:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                documents = parser.load_data(temp_pdf_path)
                node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
                nodes = node_parser.get_nodes_from_documents(documents)
                index.insert_nodes(nodes)
            st.success(f"Successfully processed {uploaded_file.name}")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
        finally:
            os.unlink(temp_pdf_path)
    
    return index

def display_chat_history():
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    st.title("PDF Chatbot")
    st.write("Upload PDFs and ask questions about their content.")

    # Get API keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    llamaparse_api_key = os.getenv("LLAMAPARSE_API_KEY")

    # Initialize LlamaIndex components
    llm = OpenAI(model="gpt-3.5-turbo", api_key=openai_api_key)
    embed_model = OpenAIEmbedding(model="text-embedding-ada-002", api_key=openai_api_key)
    
    # Create a new index
    if 'index' not in st.session_state:
        index, storage_context = create_new_index()
        st.session_state.index = index
        st.session_state.storage_context = storage_context
    else:
        index = st.session_state.index
        storage_context = st.session_state.storage_context

    # File uploader
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    # Initialize LlamaParse
    parser = LlamaParse(
        result_type="markdown",
        api_key=llamaparse_api_key,
        verbose=True,
        language="en",
        num_workers=2,
    )

    # Process uploaded files
    if uploaded_files:
        index = process_uploaded_files(uploaded_files, parser, llm, index, storage_context)
        st.session_state.index = index

    # Initialize chat engine
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        llm=llm,
        embed_model=embed_model,
        system_prompt="You are a helpful chatbot, able to have normal interactions and talk about the documents uploaded by the user. Always provide accurate information based on the documents.",
    )

    # Create containers for chat history and user input
    chat_container = st.container()
    input_container = st.container()

    with chat_container:
        display_chat_history()

    with input_container:
        # Chat interface
        if user_input := st.chat_input("Ask me anything about the uploaded documents..."):
            if user_input.lower() == 'exit':
                st.write("Exiting chat.")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_input)

                # Get bot response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = chat_engine.chat(user_input)
                        st.write(str(response))
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": str(response)})

    # Add a button to clear chat history
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.experimental_rerun()

    # Add a button to clear uploaded files and index
    if st.sidebar.button("Clear Uploaded Files"):
        st.session_state.index = None
        st.session_state.storage_context = None
        st.experimental_rerun()

if __name__ == "__main__":
    main()