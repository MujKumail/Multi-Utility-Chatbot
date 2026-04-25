# 🤖 LangGraph Chatbot

A multi-utility AI chatbot built with **LangGraph**, **Streamlit**, and **Groq** — featuring persistent multi-thread chat, PDF-based RAG with citations, and a suite of live utility tools.

---

## ✨ Features

### 💬 Chat
- Multiple independent chat threads
- Persistent chat names, ordering, and history
- Rename chats inline
- Search and filter past conversations
- Most recently active chat surfaces to the top
- Export current chat as **TXT** or **JSON**
- Header-level actions: `New Chat` and `Export`

### 📄 PDF / RAG
- Upload a PDF to any specific chat thread
- Ask questions grounded in the document's content
- Source citations with page-level references
- Document persists across refreshes and restarts
- Full per-chat document isolation
- Remove a document from any chat at any time

### 🛠️ Live Tools
| Category | Tools |
|---|---|
| **Search** | Live web search |
| **Finance** | Stock price lookup, currency exchange rates |
| **Math** | Calculator, percentage calculations |
| **Utilities** | Date/time, URL extraction, slug generation |
| **Text** | String transformations, temperature & distance conversion |

### 💾 Persistence
- Chat history stored in **SQLite** via LangGraph checkpointing
- Chat metadata (names, order) stored separately in SQLite
- Uploaded document metadata stored in SQLite
- PDF files persisted to disk and linked to their chat thread
- Designed for zero-state-loss on restart or redeployment

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Orchestration | LangGraph |
| LLM | Groq |
| Embeddings | HuggingFace |
| Vector Store | FAISS |
| Persistence | SQLite |
| Tools | Tavily, Alpha Vantage, MCP |

---

## 📁 Project Structure

```
LangGraph_Chatbot/
├── langgraph_rag_backend.py       # LangGraph graph, tools, RAG logic
├── streamlit_rag_frontend.py      # Streamlit UI
├── mcp_server/
│   └── mcp_server.py              # MCP tool server
├── thread_uploads/                # Uploaded PDF storage
├── requirements.txt
├── render.yaml
├── .env.example
└── .gitignore
```

---

## ⚙️ Environment Variables

Copy `.env.example` to `.env` and fill in your keys:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
ALPHAVANTAGE_API_KEY=your_alphavantage_api_key
APP_DATA_DIR=./data
```

> **Note:** `APP_DATA_DIR` controls where the app stores `chatbot.db` and uploaded PDFs. In deployment, point this to a persistent volume.

---

## 🚀 Local Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd LangGraph_Chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv myenv

# Windows (PowerShell)
.\myenv\Scripts\Activate.ps1

# macOS / Linux
source myenv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add environment variables

Create a `.env` file in the project root (see [Environment Variables](#️-environment-variables) above).

### 5. Run the app

```bash
streamlit run streamlit_rag_frontend.py
```

> On Windows without activating the environment:
> ```bash
> .\myenv\Scripts\streamlit.exe run streamlit_rag_frontend.py
> ```

---

## ☁️ Deployment (Render)

This project includes first-class support for [Render](https://render.com) deployment.

### Why Render?

The app requires persistent storage for the SQLite database and uploaded PDFs — both of which Render supports via mounted persistent disks.

### What's Included

- `render.yaml` with service and disk configuration
- `APP_DATA_DIR` wired to the persistent disk mount path
- Deployment-safe data path handling throughout the app

### Required Environment Variables on Render

| Variable | Notes |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `TAVILY_API_KEY` | Your Tavily search key |
| `ALPHAVANTAGE_API_KEY` | Your Alpha Vantage key |
| `APP_DATA_DIR` | Already defined in `render.yaml` — do not override |

---

## 🔍 How It Works

### Chat Persistence
Chat history is stored via LangGraph's SQLite checkpointer, so conversation state survives restarts and redeploys without any extra setup.

### Thread Metadata
Chat names and activity order are maintained in a separate SQLite table. This means thread titles never reset to *"New Chat"* after a refresh — they stay exactly as you named them.

### Document Persistence
Uploaded PDFs are saved to disk and linked to their specific thread ID. Vector retrievers are built lazily on first use and rebuilt automatically when the app restarts, ensuring RAG continues to work correctly across sessions.

---

## 🔮 Future Improvements

- [ ] User authentication
- [ ] Global shared document library
- [ ] Multi-document RAG per thread
- [ ] OCR support for scanned PDFs
- [ ] Regenerate / retry last response
- [ ] Pinned chats
- [ ] Workspaces and folder organization
