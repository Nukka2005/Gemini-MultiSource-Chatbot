import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, UnstructuredURLLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import tempfile, asyncio, sys
from dotenv import load_dotenv

load_dotenv()

# ✅ Fix asyncio issue on Streamlit + Windows
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 🎨 Page setup
st.set_page_config(page_title="Chat with Multiple Sources", page_icon="📚", layout="wide")

# 🌟 Custom CSS for extravagant chat-like styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    h1, h2, h3 {
        color: #FFD700 !important;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.8);
    }
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 15px;
        margin: 8px 0;
        max-width: 80%;
    }
    .user-bubble {
        background-color: #FFD700;
        color: black;
        margin-left: auto;
        text-align: right;
    }
    .ai-bubble {
        background-color: rgba(255,255,255,0.1);
        border: 1px solid #FFD700;
        color: white;
        margin-right: auto;
        text-align: left;
    }
    .stTextInput input {
        border: 2px solid #FFD700;
        border-radius: 10px;
        background-color: #1E1E2F;
        color: #FFD700;
    }
    .stButton button {
        background: linear-gradient(to right, #FFD700, #FFA500);
        color: black;
        font-weight: bold;
        border-radius: 12px;
        padding: 10px 20px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background: linear-gradient(to right, #FFA500, #FF4500);
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# 🎉 Title Section
st.markdown("<h1 style='text-align: center;'>📚 Gemini RAG Assistant 🚀</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Upload PDFs, DOCX, PPTX, or URLs and chat with them in real-time</h3>", unsafe_allow_html=True)

# 📂 Sidebar for inputs
with st.sidebar:
    st.markdown("## ⚡ Upload & Connect Sources")
    uploaded_files = st.file_uploader("📂 Upload files", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
    url_input = st.text_input("🌍 Enter a webpage URL")

docs = []

# 📥 File ingestion
if uploaded_files:
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        else:
            loader = UnstructuredFileLoader(tmp_path)
        docs.extend(loader.load())

# 📥 URL ingestion
if url_input:
    url_loader = UnstructuredURLLoader(urls=[url_input])
    docs.extend(url_loader.load())

# 🧠 Build vector DB
if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    if sys.platform.startswith("win"):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", transport="grpc")
    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 8})

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, max_tokens=500)

    system_prompt = """You are a helpful assistant for question-answering tasks.
    Use the retrieved context to answer the user's question.
    If the answer is not in the documents, say you don't know. {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    # 📝 Chat history stored in session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.chat_input("💡 Ask something about your documents or URLs:")

    if query:
        response = rag_chain.invoke({"input": query})
        st.session_state.chat_history.append({"user": query, "ai": response["answer"]})

    # 💬 Display conversation in chat format
    for chat in st.session_state.chat_history:
        st.markdown(f"<div class='chat-bubble user-bubble'>🧑 {chat['user']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble ai-bubble'>🤖 {chat['ai']}</div>", unsafe_allow_html=True)
else:
    st.info("👆 Upload some documents or enter a URL to start chatting.")
