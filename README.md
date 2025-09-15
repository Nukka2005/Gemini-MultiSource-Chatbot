# 🤖 Gemini MultiSource Chat

A powerful, elegant, and interactive Retrieval-Augmented Generation (RAG) application built with Streamlit and Google's Gemini AI. Chat with your documents (PDF, DOCX, PPTX) and webpages in real-time through a beautiful, chat-like interface.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-FFD700?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-4285F4?style=for-the-badge&logo=google&logoColor=white)
![Chroma](https://img.shields.io/badge/Chroma-DB-FF6B6B?style=for-the-badge)

## ✨ Features

- **📂 Multi-Source Ingestion**: Upload and process multiple `PDF`, `DOCX`, and `PPTX` files simultaneously.
- **🌐 Webpage Content Extraction**: Provide a URL to chat with the content of any webpage.
- **💬 Conversational AI**: Interact with your documents using Google's state-of-the-art `gemini-2.0-flash` LLM.
- **🎨 Beautiful UI**: A sleek, modern, and chat-like interface with a luxurious dark theme and golden accents.
- **🔍 Semantic Search**: Powered by Chroma vector DB and Gemini embeddings for highly relevant context retrieval.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- A Google AI Studio API Key. Get it free [here](https://aistudio.google.com/).

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/Gemini-MultiSource-Chat.git
    cd Gemini-MultiSource-Chat
    ```

2.  **Create a virtual environment (recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variable**
    - Create a `.env` file in the root directory.
    - Add your Google Gemini API key to it:
      ```bash
      GOOGLE_API_KEY=your_actual_api_key_here
      ```

### Usage

1.  **Run the Streamlit app**
    ```bash
    streamlit run main.py
    ```

2.  **Open your browser** and go to the local URL provided (usually `http://localhost:8501`).

3.  **In the sidebar**:
    - Upload your documents.
    - Or/And enter a valid webpage URL.
    - Click on the process button if necessary (the app may process automatically).

4.  **Start chatting!** Type your question in the chat input at the bottom and press Enter.

## 📁 Project Structure
# Gemini-MultiSource-Chat
* main.py # Main Streamlit application code
* requirements.txt # Python dependencies
* .env # Environment variables 
* README.md

## 🛠️ How It Works

1.  **Loading**: Documents are loaded using LangChain's `PyPDFLoader` and `UnstructuredFileLoader`. Webpages are loaded with `UnstructuredURLLoader`.
2.  **Splitting**: The text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding & Storage**: Text chunks are converted into vectors using `GoogleGenerativeAIEmbeddings` and stored in a `Chroma` vector database.
4.  **Retrieval & Generation**: When you ask a question, the app retrieves the most relevant text chunks and passes them as context to the Gemini LLM to generate an accurate, context-aware answer.

## 🔧 Configuration

- The model can be changed in `main.py` (`gemini-2.0-flash`).
- Adjust the `chunk_size` and `chunk_overlap` in the `RecursiveCharacterTextSplitter` for different document types.
- The number of retrieved context chunks (`k`) can be modified in the `search_kwargs` parameter.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/Gemini-MultiSource-Chat/issues).

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/).
- Powered by [Google Gemini](https://deepmind.google/technologies/gemini/).
- Vector storage by [Chroma](https://www.trychroma.com/).
