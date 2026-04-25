# LangGraph Chatbot

A multi-utility AI chatbot built with **LangGraph**, **Streamlit**, and **Groq**, featuring:

- multi-chat thread support
- persistent chat history
- PDF upload + RAG
- source citations
- export chat
- thread rename/search
- per-chat document isolation
- live tools for search, stock, currency, date/time, calculator, and text utilities

## Features

### Chat Features
- Multiple chat threads
- Persistent chat names and ordering
- Rename chats
- Search/filter past conversations
- Latest active chat moves to the top
- Export current chat as TXT or JSON
- Header actions for `New Chat` and `Export`

### PDF / RAG Features
- Upload a PDF to a specific chat
- Ask questions about the uploaded document
- Source citations with page references
- Document persists for that chat across refresh/restart
- Documents are isolated per chat
- Remove document from a specific chat

### Tool Features
- Live web search
- Stock price lookup
- Currency exchange rate
- Calculator for mathematical expressions
- Date/time utilities
- Percentage calculations
- URL extraction
- Slug generation
- String transformations
- Temperature and distance conversion

### Persistence
- Chat history stored in SQLite
- Chat metadata stored in SQLite
- Uploaded document metadata stored in SQLite
- Uploaded PDF files stored on disk
- Built for persistent storage in deployment

## Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangGraph
- **LLM:** Groq
- **Embeddings:** HuggingFace
- **Vector Store:** FAISS
- **Persistence:** SQLite
- **Tools:** Tavily, Alpha Vantage, MCP tools

## Project Structure

```text
LangGraph_Chatbot/
├── langgraph_rag_backend.py
├── streamlit_rag_frontend.py
├── mcp_server/
│   └── mcp_server.py
├── requirements.txt
├── render.yaml
├── .env.example
├── .gitignore
└── thread_uploads/

**## Environment Variables**

```text
Create a .env file locally using .env.example.

**## Required variables:**

GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
APP_DATA_DIR=./data

**## Notes**

APP_DATA_DIR controls where the app stores:
chatbot.db
uploaded PDFs
In deployment, this should point to persistent storage.

**## Local Setup**

**### 1. Clone the repository**

```bash
git clone <your-repo-url>
cd LangGraph_Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv myenv
```

### 3. Activate the virtual environment

```bash
.\myenv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Add environment variables
Create a .env file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
```

### 6. Run the app
```bash
streamlit run streamlit_rag_frontend.py
```
Or on Windows without activation:
```bash
.\myenv\Scripts\streamlit.exe run streamlit_rag_frontend.py
```

## Deployment
This project is prepared for Render deployment.

Why Render?
The app needs persistent storage for:
SQLite database
uploaded PDFs
Included deployment support
render.yaml
persistent disk mount path
environment variable configuration
deployment-safe data path with APP_DATA_DIR

## Render Requirements
Add these environment variables in Render:

- GROQ_API_KEY
- TAVILY_API_KEY
- ALPHAVANTAGE_API_KEY
- APP_DATA_DIR is already defined in render.yaml:

## How It Works

### Chat Persistence
Chats are stored using LangGraph + SQLite, so thread history survives refreshes and restarts.

### Thread Metadata
Chat names and activity order are stored separately in SQLite, so titles do not reset to New chat after refresh.

### Document Persistence
Uploaded PDFs are saved to disk and linked to their chat thread. Retrievers are rebuilt lazily when needed.

## Future Improvements
- User authentication
- Global document library
- Multi-document RAG
- OCR for scanned PDFs
- Regenerate response
- Pinned chats
- Workspaces/folders

