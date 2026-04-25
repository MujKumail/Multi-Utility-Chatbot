from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
import threading
import time
from datetime import datetime
from typing import Annotated, Any, Dict, Optional, TypedDict

import aiosqlite
import requests
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("APP_DATA_DIR", BASE_DIR)
DB_PATH = os.path.join(DATA_DIR, "chatbot.db")
UPLOADS_DIR = os.path.join(DATA_DIR, "thread_uploads")
TIME_SENSITIVE_KEYWORDS = {
    "latest",
    "current",
    "today",
    "recent",
    "newest",
    "price",
    "pricing",
    "release",
    "released",
    "version",
    "update",
    "updated",
    "stock",
    "lineup",
    "model",
    "models",
}
THREAD_METADATA_TABLE = "thread_metadata"
THREAD_DOCUMENTS_TABLE = "thread_documents"

load_dotenv()
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# -------------------
# Async event loop
# -------------------
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    return _submit_async(coro)


def is_time_sensitive_query(text: str) -> bool:
    """Return True when a user query likely needs live data."""
    words = {word.strip(".,?!:;()[]{}\"'").lower() for word in text.split()}
    return any(keyword in words for keyword in TIME_SENSITIVE_KEYWORDS)


# -------------------
# 1. LLM
# -------------------
llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7,
)


# -------------------
# 2. Embeddings
# -------------------
_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily load embeddings so the app can boot before any PDF is uploaded."""
    global _EMBEDDINGS
    if _EMBEDDINGS is None:
        _EMBEDDINGS = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _EMBEDDINGS


# -------------------
# 3. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _safe_filename(filename: Optional[str]) -> str:
    base_name = os.path.basename(filename or "uploaded.pdf").strip()
    return base_name or "uploaded.pdf"


def _thread_upload_path(thread_id: str, filename: Optional[str]) -> str:
    safe_name = _safe_filename(filename)
    thread_dir = os.path.join(UPLOADS_DIR, str(thread_id))
    os.makedirs(thread_dir, exist_ok=True)
    return os.path.join(thread_dir, safe_name)


def _build_retriever_from_pdf(file_path: str):
    """Load a PDF from disk and build a retriever plus summary metadata."""
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    vector_store = FAISS.from_documents(chunks, get_embeddings())
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    return retriever, {
        "filename": os.path.basename(file_path),
        "documents": len(docs),
        "chunks": len(chunks),
        "file_path": file_path,
    }


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if not thread_id:
        return None
    tid = str(thread_id)
    if tid in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[tid]

    metadata = run_async(_load_thread_document(tid))
    file_path = metadata.get("file_path") if metadata else None
    if file_path and os.path.exists(file_path):
        retriever, loaded_meta = _build_retriever_from_pdf(file_path)
        _THREAD_RETRIEVERS[tid] = retriever
        cached_meta = dict(metadata)
        cached_meta.update(loaded_meta)
        _THREAD_METADATA[tid] = cached_meta
        return retriever
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    Returns a summary dict that can be surfaced in the UI.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        retriever, summary = _build_retriever_from_pdf(temp_path)
        upload_time = datetime.now().isoformat(timespec="seconds")
        saved_path = _thread_upload_path(str(thread_id), filename)

        thread_dir = os.path.dirname(saved_path)
        for existing_name in os.listdir(thread_dir):
            existing_path = os.path.join(thread_dir, existing_name)
            if os.path.isfile(existing_path) and existing_path != saved_path:
                os.remove(existing_path)

        shutil.copyfile(temp_path, saved_path)

        persisted_summary = {
            "filename": _safe_filename(filename),
            "documents": summary["documents"],
            "chunks": summary["chunks"],
            "uploaded_at": upload_time,
            "file_path": saved_path,
        }

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = persisted_summary
        run_async(_persist_thread_document(str(thread_id), persisted_summary))

        return {
            "filename": persisted_summary["filename"],
            "documents": persisted_summary["documents"],
            "chunks": persisted_summary["chunks"],
            "uploaded_at": persisted_summary["uploaded_at"],
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


def get_thread_metadata(thread_id: str) -> dict:
    """Return PDF metadata for the given thread, or empty dict if none."""
    metadata = run_async(_load_thread_document(str(thread_id))) or {}
    if not metadata:
        return {}
    return {
        "filename": metadata.get("filename"),
        "documents": metadata.get("documents"),
        "chunks": metadata.get("chunks"),
        "uploaded_at": metadata.get("uploaded_at"),
    }


async def _delete_thread_document(thread_id: str) -> None:
    tid = str(thread_id)
    metadata = await _load_thread_document(tid)
    file_path = metadata.get("file_path") if metadata else None

    _THREAD_METADATA.pop(tid, None)
    _THREAD_RETRIEVERS.pop(tid, None)

    if _db_conn is not None:
        await _db_conn.execute(
            f"DELETE FROM {THREAD_DOCUMENTS_TABLE} WHERE thread_id = ?",
            (tid,),
        )
        await _db_conn.commit()

    if file_path and os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError:
            pass

        parent_dir = os.path.dirname(file_path)
        if os.path.isdir(parent_dir):
            try:
                if not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except OSError:
                pass


def remove_thread_document(thread_id: str) -> None:
    """Remove the persisted PDF and retriever for a single chat thread."""
    run_async(_delete_thread_document(str(thread_id)))


# -------------------
# 4. Thread title store
# -------------------
_THREAD_TITLES: Dict[str, Dict[str, Any]] = {}
_METADATA_SERDE = JsonPlusSerializer()


def _normalize_thread_title(raw_title: str) -> str:
    """Trim and constrain stored thread titles."""
    title = raw_title.strip() or "New chat"
    if len(title) > 50:
        title = title[:47] + "..."
    return title


def _cache_thread_record(thread_id: str, title: str, created_at: float, last_updated_at: float) -> dict:
    record = {
        "title": _normalize_thread_title(title),
        "created_at": created_at,
        "last_updated_at": last_updated_at,
    }
    _THREAD_TITLES[str(thread_id)] = record
    return record


def _parse_checkpoint_timestamp(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


async def _ensure_thread_metadata_table() -> None:
    if _db_conn is None:
        return
    await _db_conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {THREAD_METADATA_TABLE} (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at REAL NOT NULL,
            last_updated_at REAL NOT NULL
        )
        """
    )
    await _db_conn.commit()


async def _ensure_thread_documents_table() -> None:
    if _db_conn is None:
        return
    await _db_conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {THREAD_DOCUMENTS_TABLE} (
            thread_id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT NOT NULL,
            documents INTEGER NOT NULL,
            chunks INTEGER NOT NULL,
            uploaded_at TEXT NOT NULL
        )
        """
    )
    await _db_conn.commit()


async def _fetch_thread_record(thread_id: str) -> dict | None:
    if _db_conn is None:
        return None
    async with _db_conn.execute(
        f"""
        SELECT title, created_at, last_updated_at
        FROM {THREAD_METADATA_TABLE}
        WHERE thread_id = ?
        """,
        (str(thread_id),),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return _cache_thread_record(str(thread_id), row[0], float(row[1]), float(row[2]))


async def _persist_thread_record(thread_id: str, title: str, created_at: float, last_updated_at: float) -> None:
    if _db_conn is None:
        return
    await _db_conn.execute(
        f"""
        INSERT INTO {THREAD_METADATA_TABLE} (thread_id, title, created_at, last_updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            title = excluded.title,
            created_at = excluded.created_at,
            last_updated_at = excluded.last_updated_at
        """,
        (str(thread_id), _normalize_thread_title(title), created_at, last_updated_at),
    )
    await _db_conn.commit()


async def _load_thread_document(thread_id: str) -> dict:
    tid = str(thread_id)
    if tid in _THREAD_METADATA:
        return _THREAD_METADATA[tid]
    if _db_conn is None:
        return {}

    async with _db_conn.execute(
        f"""
        SELECT filename, file_path, documents, chunks, uploaded_at
        FROM {THREAD_DOCUMENTS_TABLE}
        WHERE thread_id = ?
        """,
        (tid,),
    ) as cursor:
        row = await cursor.fetchone()

    if row is None:
        return {}

    metadata = {
        "filename": row[0],
        "file_path": row[1],
        "documents": int(row[2]),
        "chunks": int(row[3]),
        "uploaded_at": row[4],
    }

    if not os.path.exists(metadata["file_path"]):
        await _db_conn.execute(
            f"DELETE FROM {THREAD_DOCUMENTS_TABLE} WHERE thread_id = ?",
            (tid,),
        )
        await _db_conn.commit()
        return {}

    _THREAD_METADATA[tid] = metadata
    return metadata


async def _persist_thread_document(thread_id: str, metadata: dict) -> None:
    if _db_conn is None:
        return
    await _db_conn.execute(
        f"""
        INSERT INTO {THREAD_DOCUMENTS_TABLE}
            (thread_id, filename, file_path, documents, chunks, uploaded_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(thread_id) DO UPDATE SET
            filename = excluded.filename,
            file_path = excluded.file_path,
            documents = excluded.documents,
            chunks = excluded.chunks,
            uploaded_at = excluded.uploaded_at
        """,
        (
            str(thread_id),
            metadata["filename"],
            metadata["file_path"],
            int(metadata["documents"]),
            int(metadata["chunks"]),
            metadata["uploaded_at"],
        ),
    )
    await _db_conn.commit()


async def _derive_thread_record_from_checkpoints(thread_id: str) -> dict | None:
    if _db_conn is None:
        return None

    async with _db_conn.execute(
        """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY rowid DESC
        LIMIT 1
        """,
        (str(thread_id),),
    ) as cursor:
        latest_row = await cursor.fetchone()

    if latest_row is None:
        return None

    async with _db_conn.execute(
        """
        SELECT type, checkpoint
        FROM checkpoints
        WHERE thread_id = ?
        ORDER BY rowid ASC
        LIMIT 1
        """,
        (str(thread_id),),
    ) as cursor:
        first_row = await cursor.fetchone()

    latest_checkpoint = _METADATA_SERDE.loads_typed((latest_row[0], latest_row[1]))
    first_checkpoint = (
        _METADATA_SERDE.loads_typed((first_row[0], first_row[1]))
        if first_row is not None
        else latest_checkpoint
    )

    messages = latest_checkpoint.get("channel_values", {}).get("messages", [])
    first_human = next(
        (message.content for message in messages if isinstance(message, HumanMessage)),
        None,
    )

    created_at = _parse_checkpoint_timestamp(first_checkpoint.get("ts")) or time.time()
    last_updated_at = _parse_checkpoint_timestamp(latest_checkpoint.get("ts")) or created_at
    title = _normalize_thread_title(first_human or "New chat")

    record = _cache_thread_record(str(thread_id), title, created_at, last_updated_at)
    await _persist_thread_record(
        thread_id,
        record["title"],
        record["created_at"],
        record["last_updated_at"],
    )
    return record


async def _load_thread_record(thread_id: str) -> dict | None:
    tid = str(thread_id)
    if tid in _THREAD_TITLES:
        return _THREAD_TITLES[tid]

    record = await _fetch_thread_record(tid)
    if record is not None:
        return record

    return await _derive_thread_record_from_checkpoints(tid)


async def _ensure_thread_records(thread_ids: list[str]) -> None:
    for thread_id in thread_ids:
        await _load_thread_record(thread_id)


def set_thread_title(thread_id: str, first_message: str) -> None:
    """
    Save the title for a thread using the first user message.
    Only sets once so the initial message remains the default title.
    """
    tid = str(thread_id)
    existing = run_async(_load_thread_record(tid))
    if existing is None:
        now = time.time()
        record = _cache_thread_record(
            tid,
            _normalize_thread_title(first_message),
            now,
            now,
        )
        run_async(
            _persist_thread_record(
                tid,
                record["title"],
                record["created_at"],
                record["last_updated_at"],
            )
        )


def rename_thread(thread_id: str, new_title: str) -> str:
    """Rename a thread while preserving its creation timestamp."""
    tid = str(thread_id)
    existing = run_async(_load_thread_record(tid)) or {}
    title = _normalize_thread_title(new_title)
    record = _cache_thread_record(
        tid,
        title,
        float(existing.get("created_at", time.time())),
        float(existing.get("last_updated_at", existing.get("created_at", time.time()))),
    )
    run_async(
        _persist_thread_record(
            tid,
            record["title"],
            record["created_at"],
            record["last_updated_at"],
        )
    )
    return title


def get_thread_title(thread_id: str) -> str:
    """Return the stored title for a thread, or a short fallback."""
    entry = run_async(_load_thread_record(str(thread_id)))
    return entry["title"] if entry else "New chat"


def get_thread_created_at(thread_id: str) -> float:
    """Return the unix creation timestamp for a thread (0.0 if unknown)."""
    entry = run_async(_load_thread_record(str(thread_id)))
    return entry["created_at"] if entry else 0.0


def touch_thread(thread_id: str) -> None:
    """Mark a thread as recently active without changing its title."""
    tid = str(thread_id)
    entry = run_async(_load_thread_record(tid))
    now = time.time()
    if entry is None:
        record = _cache_thread_record(tid, "New chat", now, now)
        run_async(
            _persist_thread_record(
                tid,
                record["title"],
                record["created_at"],
                record["last_updated_at"],
            )
        )
        return
    entry["last_updated_at"] = now
    run_async(
        _persist_thread_record(
            tid,
            entry["title"],
            float(entry["created_at"]),
            float(entry["last_updated_at"]),
        )
    )


def get_thread_last_updated_at(thread_id: str) -> float:
    """Return the unix last-activity timestamp for a thread."""
    entry = run_async(_load_thread_record(str(thread_id)))
    if not entry:
        return 0.0
    return entry.get("last_updated_at", entry.get("created_at", 0.0))


# -------------------
# 5. Built-in tools
# -------------------
search_tool = TavilySearch(max_results=2)


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch latest stock price for a given symbol e.g. AAPL, TSLA."""
    api_key = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
    if not api_key:
        return {
            "error": (
                "ALPHAVANTAGE_API_KEY is not configured. "
                "Set it in the environment before using stock lookups."
            ),
            "symbol": symbol,
        }
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}"
        f"&apikey={api_key}"
    )
    return requests.get(url, timeout=10).json()


def make_rag_tool(thread_id: str):
    """
    Factory returning a rag_tool closure bound to a specific thread_id.
    Avoids relying on the LLM to pass thread_id as an argument.
    """

    @tool
    def rag_tool(query: str) -> dict:
        """
        Retrieve relevant information from the uploaded PDF for this chat thread.
        Use this whenever the user asks about the contents of the uploaded document.
        """
        retriever = _get_retriever(thread_id)
        if retriever is None:
            return {
                "error": "No document indexed for this chat. Please upload a PDF first.",
                "query": query,
            }

        result = retriever.invoke(query)
        source_file = _THREAD_METADATA.get(str(thread_id), {}).get("filename")

        citations = []
        for doc in result:
            metadata = doc.metadata or {}
            page_number = metadata.get("page")
            if isinstance(page_number, int):
                page_number += 1

            snippet = " ".join(doc.page_content.split())
            citations.append(
                {
                    "page": page_number,
                    "source": metadata.get("source") or source_file,
                    "snippet": snippet[:280],
                }
            )

        return {
            "query": query,
            "context": [doc.page_content for doc in result],
            "metadata": [doc.metadata for doc in result],
            "citations": citations,
            "source_file": source_file,
        }

    return rag_tool


# -------------------
# 6. MCP tools
# -------------------
MCP_SERVER_CONFIG = {
    "demo_tools": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [os.path.join(BASE_DIR, "mcp_server", "mcp_server.py")],
    }
}


async def _fetch_mcp_tools() -> list[BaseTool]:
    """
    Spawn a fresh MCP client and load tools from the subprocess.
    A new client means a fresh subprocess, so dead processes are restarted.
    """
    client = MultiServerMCPClient(MCP_SERVER_CONFIG)
    try:
        tools = await asyncio.wait_for(client.get_tools(), timeout=15.0)
        print(f"MCP tools loaded ({len(tools)}): {[t.name for t in tools]}")
        return tools
    except Exception as exc:
        print(f"MCP tools failed to load: {exc}. Continuing without MCP tools.")
        return []


_startup_mcp_tools: list[BaseTool] = run_async(_fetch_mcp_tools())


# -------------------
# 7. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 8. System prompt
# -------------------
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to tools. Use them when needed:\n"
    "search_tool                              -> current news, events, weather, products, model lineups, and other live facts\n"
    "get_stock_price                          -> live stock prices (e.g. AAPL, TSLA)\n"
    "get_exchange_rate                        -> currency exchange rates (e.g. USD to INR, EUR to GBP)\n"
    "calculate_expression                     -> math expressions like (245 * 18) / 6\n"
    "reverse_string                           -> reverse any text\n"
    "word_count                               -> count words in text\n"
    "to_uppercase                             -> convert text to uppercase\n"
    "to_lowercase                             -> convert text to lowercase\n"
    "celsius_to_fahrenheit / fahrenheit_to_celsius -> temperature conversion\n"
    "km_to_miles / miles_to_km                -> distance conversion\n"
    "get_current_datetime                     -> current date and time\n"
    "days_between_dates                       -> day difference between two YYYY-MM-DD dates\n"
    "calculate_percentage / percentage_change -> percentage calculations\n"
    "extract_urls / slugify_text              -> text utility helpers\n"
    "rag_tool                                 -> answer questions about the uploaded PDF document\n"
    "For any time-sensitive question about latest/current/recent/today/newest/prices/releases/models/versions, you must use a live tool first and answer from tool results.\n"
    "Do not answer time-sensitive questions from memory.\n"
    "When rag_tool returns evidence, cite the source file and page number in your answer.\n"
)


# -------------------
# 9. Graph builder
# -------------------
def build_graph(thread_id: str):
    """
    Build and compile a chatbot graph for a specific thread.
    Fetches fresh MCP tools each time and injects a rag_tool bound to thread_id.
    """
    mcp_tools: list[BaseTool] = run_async(_fetch_mcp_tools())

    rag_tool = make_rag_tool(thread_id)
    tools = [search_tool, get_stock_price, *mcp_tools, rag_tool]
    llm_with_tools = llm.bind_tools(tools)
    system_prompt = SystemMessage(content=SYSTEM_PROMPT_TEMPLATE)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        filtered = [m for m in messages if not isinstance(m, SystemMessage)]

        trimmed = filtered[-6:] if len(filtered) > 6 else filtered

        cleaned = []
        for message in trimmed:
            if isinstance(message, ToolMessage) and len(str(message.content)) > 800:
                truncated = str(message.content)[:800] + "...[truncated]"
                message = ToolMessage(
                    content=truncated,
                    tool_call_id=message.tool_call_id,
                    name=getattr(message, "name", None),
                )
            cleaned.append(message)

        live_data_prompt: list[SystemMessage] = []
        last_human = next(
            (message for message in reversed(cleaned) if isinstance(message, HumanMessage)),
            None,
        )
        recent_tool_used = any(isinstance(message, ToolMessage) for message in cleaned)
        if last_human and is_time_sensitive_query(str(last_human.content)) and not recent_tool_used:
            live_data_prompt = [
                SystemMessage(
                    content=(
                        "The current user request is time-sensitive. "
                        "You must call an appropriate live tool before answering. "
                        "Prefer search_tool for current product lineups, releases, and latest facts."
                    )
                )
            ]

        try:
            response = await llm_with_tools.ainvoke([system_prompt] + live_data_prompt + cleaned)
            if live_data_prompt and not getattr(response, "tool_calls", None):
                response = await llm_with_tools.ainvoke(
                    [
                        system_prompt,
                        *live_data_prompt,
                        SystemMessage(
                            content=(
                                "You still need live verification. "
                                "Do not answer yet. Call a live tool now."
                            )
                        ),
                        *cleaned,
                    ]
                )
        except Exception as exc:
            if "Failed to call a function" in str(exc) or "failed_generation" in str(exc):
                print("Tool call failed, retrying with clean context...")
                last_human_messages = [m for m in cleaned if isinstance(m, HumanMessage)]
                retry_msgs = [last_human_messages[-1]] if last_human_messages else cleaned[-1:]
                response = await llm_with_tools.ainvoke([system_prompt] + live_data_prompt + retry_msgs)
            else:
                raise

        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer)


# -------------------
# 10. Checkpointer
# -------------------
_db_conn: aiosqlite.Connection | None = None


async def _init_checkpointer() -> AsyncSqliteSaver:
    global _db_conn
    _db_conn = await aiosqlite.connect(database=DB_PATH)
    await _ensure_thread_metadata_table()
    await _ensure_thread_documents_table()
    return AsyncSqliteSaver(_db_conn)


async def _close_checkpointer():
    if _db_conn is not None:
        await _db_conn.close()


checkpointer: AsyncSqliteSaver = run_async(_init_checkpointer())

_GRAPH_CACHE: Dict[str, Any] = {}


def get_or_build_graph(thread_id: str):
    """Return a cached compiled graph for this thread, building it if needed."""
    if thread_id not in _GRAPH_CACHE:
        _GRAPH_CACHE[thread_id] = build_graph(thread_id)
    return _GRAPH_CACHE[thread_id]


def invalidate_graph_cache(thread_id: str) -> None:
    """
    Drop the cached graph for a thread.
    Call this after ingesting a new PDF so the next invocation picks up updates.
    """
    _GRAPH_CACHE.pop(thread_id, None)


# -------------------
# 11. Helpers
# -------------------
async def _alist_threads() -> list[str]:
    all_threads: set[str] = set()
    async for checkpoint in checkpointer.alist(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def retrieve_all_threads() -> list[str]:
    threads = run_async(_alist_threads())
    run_async(_ensure_thread_records(threads))
    return threads
