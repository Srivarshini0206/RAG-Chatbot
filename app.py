import streamlit as pd_rag_st
import os
import tempfile
from dotenv import load_dotenv

# Ensure safe and clean loading of environment variables
load_dotenv()

# LangChain and Google GenAI imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# Configure Page Styling & Theme
pd_rag_st.set_page_config(
    page_title="PDF RAG Chatbot - Google Gemini",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Elegant Styling using CSS injections
pd_rag_st.markdown("""
<style>
    /* Styling for Headers */
    .title-text {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1E293B;
    }
    .accent-text {
        color: #4F46E5;
    }
    /* Chat bubbles styling */
    .user-bubble {
        background-color: #F1F5F9;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #94A3B8;
        font-family: 'Inter', sans-serif;
    }
    .assistant-bubble {
        background-color: #EEF2FF;
        padding: 15px;
        border-radius: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #4F46E5;
        font-family: 'Inter', sans-serif;
    }
    /* Source chunk expanders styling */
    .source-header {
        font-size: 0.85em;
        font-weight: 600;
        color: #4F46E5;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Define Core Layout
# --- SIDEBAR ---
with pd_rag_st.sidebar:
    pd_rag_st.markdown('<h2 class="title-text"><span class="accent-text">📄 PDF RAG</span> Chatbot</h2>', unsafe_allow_html=True)
    pd_rag_st.write("---")
    
    # RAG Overview Section
    pd_rag_st.markdown("### 🔎 RAG Architecture Overview")
    pd_rag_st.write(
        "Retrieval-Augmented Generation (RAG) optimizes the output of a "
        "Large Language Model. It retrieves relevant content chunks from an "
        "external vector database containing your loaded document before passing "
        "the query to the LLM. This prevents hallucinations and restricts answers "
        "to the uploaded context."
    )
    
    # Technologies Section
    pd_rag_st.markdown("### 🛠️ Technologies Used")
    pd_rag_st.markdown("""
    - **Google Gemini 2.5 Flash** (LLM)
    - **GoogleGenerativeAIEmbeddings**
    - **LangChain** (Framework)
    - **FAISS** (Vector Database)
    - **PyPDF** (PDF Parser)
    - **Streamlit** (User Interface)
    """)
    
    pd_rag_st.write("---")
    
    # Developer Information Placeholder
    pd_rag_st.markdown("### 👤 Developer Info")
    pd_rag_st.markdown("""
    Developed for seamless PDF interaction. Ready to be pushed to GitHub 
    and deployed on **Streamlit Community Cloud** or **Render**.
    """)

# --- MAIN CONTENT ---
pd_rag_st.markdown('<h1 class="title-text">📄 PDF RAG Chatbot using Google Gemini</h1>', unsafe_allow_html=True)
pd_rag_st.markdown('<p style="font-size: 1.15em; color: #64748B;">Upload a PDF document and ask questions based solely on its contents.</p>', unsafe_allow_html=True)
pd_rag_st.write("---")

# Session State Initialization to ensure persistence of index and history
if "vector_db" not in pd_rag_st.session_state:
    pd_rag_st.session_state.vector_db = None
if "chat_history" not in pd_rag_st.session_state:
    pd_rag_st.session_state.chat_history = []
if "uploaded_file_name" not in pd_rag_st.session_state:
    pd_rag_st.session_state.uploaded_file_name = None

# PDF Upload Component
uploaded_file = pd_rag_st.file_uploader(
    "Choose a PDF document", 
    type=["pdf"], 
    accept_multiple_files=False,
    help="Upload a single PDF file (support for large PDFs is built-in)."
)

# Check if Gemini API Key is available
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not api_key:
    pd_rag_st.error("🔑 Missing Google API Key! Please configure the GOOGLE_API_KEY environment variable in your .env file.")
else:
    # Build vector database index if a new file is uploaded
    if uploaded_file is not None and pd_rag_st.session_state.uploaded_file_name != uploaded_file.name:
        with pd_rag_st.status("⚙️ Processing and Indexing PDF...", expanded=True) as status:
            try:
                # 1. Save uploaded file to a temporary file
                status.write("Saving uploaded PDF...")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    tmp_path = tmp_file.name
                
                # 2. Extract Text using PyPDFLoader
                status.write("Extracting text contents from pages...")
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                # Clean up the temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
                
                if not documents or len(documents) == 0:
                    raise ValueError("The uploaded PDF is empty or could not be parsed.")
                
                # 3. Split Text into manageable chunks cleanly
                status.write("Splitting document into text chunks...")
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800,
                    chunk_overlap=100,
                    length_function=len
                )
                chunked_docs = text_splitter.split_documents(documents)
                status.write(f"Generated {len(chunked_docs)} chunks from {len(documents)} pages.")
                
                # 4. Generate Embeddings using Google Gemini Embeddings
                status.write("Generating vector embeddings using Google Gemini Embeddings...")
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": False},
    
                )
                
                # 5. Index into FAISS Vector Database
                status.write("Saving embeddings into local FAISS Vector Storage...")
                vector_db = FAISS.from_documents(chunked_docs, embeddings)
                
                # Update session state with completed index
                pd_rag_st.session_state.vector_db = vector_db
                pd_rag_st.session_state.uploaded_file_name = uploaded_file.name
                pd_rag_st.session_state.chat_history = []  # Clear history for a new file
                
                status.update(label="✅ Indexing Completed Successfully!", state="complete", expanded=False)
                pd_rag_st.success(f"Successfully processed and indexed document: **{uploaded_file.name}**")
                
            except Exception as ex:
                status.update(label="❌ Indexing Failed!", state="error")
                pd_rag_st.error(f"Failed to process document: {str(ex)}")

    # Prompt user if no PDF has been uploaded yet
    if pd_rag_st.session_state.vector_db is None:
        pd_rag_st.info("💡 Please upload a PDF document above to begin.")
    else:
        # Document loaded and indexed. Allow asking questions!
        pd_rag_st.write("---")
        pd_rag_st.markdown("### 💬 Ask a Question")
        
        # We wrap input and submission button in a clean structure
        user_question = pd_rag_st.text_input(
            "State your question based on document contents:",
            placeholder="What is the main topic discussing on page 3?...",
            key="input_question"
        )
        
        button_clicked = pd_rag_st.button("Ask Question", type="primary")
        
        if button_clicked and user_question.strip():
            with pd_rag_st.spinner("🔍 Retrieving context and generating answer..."):
                try:
                    # 1. Retrieve the top 3 most relevant chunks
                    retriever = pd_rag_st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.invoke(user_question)
                    
                    # 2. Extract and join text context
                    retrieved_context = "\n\n".join([f"[Chunk {i+1} - Page {doc.metadata.get('page', 0) + 1}]:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
                    
                    # 3. Define professional prompt instructing the LLM precisely
                    prompt_template = """You are a professional RAG assistant answering user questions based solely on the provided context retrieved from an uploaded PDF.

Context:
{context}

Question:
{question}

Instructions:
1. Answer the question using ONLY the provided context. Do not use outside knowledge, general training assumptions, or hallucinate.
2. Be objective, precise, and direct.
3. If the answer cannot be found in the provided context, you MUST reply EXACTLY:
"The uploaded PDF does not contain enough information to answer this question."
Do not attempt to explain further, suggest guesses, or formulate a speculative answer.

Answer:"""
                    formatted_prompt = prompt_template.format(
                        context=retrieved_context, 
                        question=user_question
                    )
                    
                    # 4. Initialize Gemini 2.5 Flash LLM with the API key
                    # Note: We configure temperature=0.0 to prevent creative liberties and restrict to facts
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-2.5-flash",
                        google_api_key=api_key,
                        temperature=0.0
                    )
                    
                    response = llm.invoke(formatted_prompt)
                    ans_text = response.content
                    
                    # 5. Append query, context and answer to local session history
                    pd_rag_st.session_state.chat_history.append({
                        "question": user_question,
                        "answer": ans_text,
                        "sources": [
                            {
                                "page": doc.metadata.get("page", 0) + 1,
                                "content": doc.page_content
                            }
                            for doc in relevant_docs
                        ]
                    })
                    
                except Exception as e:
                    pd_rag_st.error(f"An error occurred while generating the response: {str(e)}")
        
        # Display Conversation Chat in an elegant, reversed or chronological list
        if pd_rag_st.session_state.chat_history:
            pd_rag_st.write("---")
            pd_rag_st.markdown("### 💬 Conversation")
            
            # Display entries in reverse chronological order so the newest questions are visible at the top
            for idx, chat in enumerate(reversed(pd_rag_st.session_state.chat_history)):
                # Distinguish and format chat bubbles cleanly
                pd_rag_st.markdown(f'<div class="user_q_label" style="font-weight: 600; color: #475569; margin-top: 15px;">Question:</div>', unsafe_allow_html=True)
                pd_rag_st.markdown(f'<div class="user-bubble">{chat["question"]}</div>', unsafe_allow_html=True)
                
                pd_rag_st.markdown(f'<div class="assistant_ans_label" style="font-weight: 600; color: #4F46E5;">Answer:</div>', unsafe_allow_html=True)
                pd_rag_st.markdown(f'<div class="assistant-bubble">{chat["answer"]}</div>', unsafe_allow_html=True)
                
                # Expandable retrieved chunks for advanced developer telemetry / reference
                if chat["sources"]:
                    with pd_rag_st.expander("🔍 View Retrieved Document Context Sources"):
                        for i, source in enumerate(chat["sources"]):
                            pd_rag_st.markdown(f'<div class="source-header">Source Chunk {i+1} (Page {source["page"]}):</div>', unsafe_allow_html=True)
                            pd_rag_st.info(source["content"])
                pd_rag_st.markdown("<hr style='border: 0.5px solid #E2E8F0; margin-top: 20px; margin-bottom: 20px;' />", unsafe_allow_html=True)
