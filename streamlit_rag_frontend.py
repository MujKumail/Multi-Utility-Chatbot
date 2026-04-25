import ast
import hashlib
import json
import queue
import re
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_rag_backend import (
    get_or_build_graph,
    get_thread_last_updated_at,
    get_thread_metadata,
    get_thread_title,
    ingest_pdf,
    rename_thread,
    remove_thread_document,
    retrieve_all_threads,
    run_async,
    set_thread_title,
    submit_async_task,
    touch_thread,
)


st.set_page_config(page_title="Multi Utility Chatbot", page_icon=":robot_face:", layout="wide")


def generate_thread_id() -> str:
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread(thread_id)
    st.session_state["message_history"] = []


def add_thread(thread_id: str):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def sorted_threads(threads: list[str]) -> list[str]:
    return sorted(threads, key=lambda tid: get_thread_last_updated_at(tid), reverse=True)


def parse_tool_payload(content) -> dict:
    if isinstance(content, dict):
        return content

    if isinstance(content, list):
        text_bits = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_bits.append(str(item["text"]))
                else:
                    text_bits.append(json.dumps(item))
            else:
                text_bits.append(str(item))
        content = "".join(text_bits)

    if not isinstance(content, str):
        return {}

    content = content.strip()
    if not content:
        return {}

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(content)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return {}


def normalize_citations(citations: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for citation in citations:
        source = str(citation.get("source") or "Document")
        page = "" if citation.get("page") is None else str(citation.get("page"))
        snippet = str(citation.get("snippet") or "").strip()
        key = (source, page, snippet)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "source": source,
                "page": page,
                "snippet": snippet,
            }
        )
    return unique


def extract_citations_from_tool_message(message: ToolMessage) -> list[dict]:
    payload = parse_tool_payload(message.content)
    citations = payload.get("citations", []) if isinstance(payload, dict) else []
    if not isinstance(citations, list):
        return []
    return normalize_citations([citation for citation in citations if isinstance(citation, dict)])


def render_citations(citations: list[dict]):
    citations = normalize_citations(citations)
    if not citations:
        return

    with st.expander("Sources", expanded=False):
        for index, citation in enumerate(citations, start=1):
            heading = f"{index}. {citation['source']}"
            if citation["page"]:
                heading += f" - page {citation['page']}"
            st.markdown(f"**{heading}**")
            if citation["snippet"]:
                st.caption(citation["snippet"])


def load_conversation(thread_id: str) -> list[dict]:
    """Load persisted messages for a thread and return UI-ready messages."""
    graph = get_or_build_graph(thread_id)
    state = run_async(
        graph.aget_state(config={"configurable": {"thread_id": thread_id}})
    )
    messages = state.values.get("messages", [])

    result = []
    pending_citations: list[dict] = []
    first_user_message: str | None = None

    for msg in messages:
        if isinstance(msg, HumanMessage):
            content = str(msg.content)
            if first_user_message is None:
                first_user_message = content
            result.append({"role": "user", "content": content})
            continue

        if isinstance(msg, ToolMessage):
            pending_citations.extend(extract_citations_from_tool_message(msg))
            continue

        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            result.append(
                {
                    "role": "assistant",
                    "content": str(msg.content),
                    "citations": normalize_citations(pending_citations),
                }
            )
            pending_citations = []

    if first_user_message:
        set_thread_title(thread_id, first_user_message)

    return result


def export_chat_as_text(thread_id: str, title: str, messages: list[dict]) -> str:
    lines = [
        f"Title: {title}",
        f"Thread ID: {thread_id}",
        "",
    ]

    for message in messages:
        role = str(message.get("role", "assistant")).upper()
        lines.append(f"{role}:")
        lines.append(str(message.get("content", "")))
        citations = normalize_citations(message.get("citations", []))
        if citations:
            lines.append("Sources:")
            for citation in citations:
                source_line = citation["source"]
                if citation["page"]:
                    source_line += f" (page {citation['page']})"
                lines.append(f"- {source_line}")
                if citation["snippet"]:
                    lines.append(f"  {citation['snippet']}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_export_filename(title: str, extension: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()
    slug = slug or "chat-export"
    return f"{slug}.{extension}"


def build_export_summary(messages: list[dict]) -> str:
    assistant_messages = sum(1 for message in messages if message.get("role") == "assistant")
    citation_count = sum(
        len(normalize_citations(message.get("citations", [])))
        for message in messages
        if message.get("role") == "assistant"
    )
    return f"{len(messages)} messages | {assistant_messages} replies | {citation_count} sources"


if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_key: str = str(st.session_state["thread_id"])
thread_docs: dict = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads: list[str] = st.session_state["chat_threads"]
selected_thread: str | None = None
current_title = get_thread_title(thread_key)
uploader_key = f"pdf-uploader-{thread_key}"
persisted_doc_meta = get_thread_metadata(thread_key)

if persisted_doc_meta.get("filename"):
    thread_docs.clear()
    thread_docs[persisted_doc_meta["filename"]] = persisted_doc_meta
else:
    thread_docs.clear()


st.sidebar.title("Multi Utility Chatbot")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

with st.sidebar.form("rename-thread-form"):
    renamed_title = st.text_input("Rename current chat", value=current_title)
    rename_submitted = st.form_submit_button("Save Title", use_container_width=True)
if rename_submitted:
    rename_thread(thread_key, renamed_title)
    st.rerun()

st.sidebar.divider()

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"],
    key=uploader_key,
)
if uploaded_pdf:
    uploaded_bytes = uploaded_pdf.getvalue()
    uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()
    existing_doc = thread_docs.get(uploaded_pdf.name)

    if existing_doc and existing_doc.get("file_hash") == uploaded_hash:
        st.sidebar.info(f"{uploaded_pdf.name} already processed for this chat.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            summary = ingest_pdf(
                uploaded_bytes,
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )
            thread_docs.clear()
            thread_docs[uploaded_pdf.name] = summary
            status_box.update(label="PDF indexed", state="complete", expanded=False)

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    doc_summary = (
        f"Document: {latest_doc.get('filename')} | "
        f"{latest_doc.get('chunks')} chunks | {latest_doc.get('documents')} pages"
    )
    if latest_doc.get("uploaded_at"):
        doc_summary += f" | Uploaded: {latest_doc['uploaded_at']}"
    st.sidebar.success(doc_summary)
    if st.sidebar.button("Remove Document", use_container_width=True):
        remove_thread_document(thread_key)
        thread_docs.clear()
        st.rerun()
else:
    st.sidebar.info("No PDF indexed yet for this chat.")

st.sidebar.divider()

thread_filter = st.sidebar.text_input("Search chats", placeholder="Filter by title or thread id")
st.sidebar.subheader("Past conversations")

matching_threads = []
for tid in sorted_threads(threads):
    if tid == thread_key:
        continue
    title = get_thread_title(tid)
    haystack = f"{title} {tid[:8]}".lower()
    if thread_filter.strip() and thread_filter.strip().lower() not in haystack:
        continue
    matching_threads.append((tid, title))

if not matching_threads:
    st.sidebar.write("No matching conversations.")
else:
    for tid, title in matching_threads:
        if st.sidebar.button(
            title,
            key=f"side-thread-{tid}",
            use_container_width=True,
        ):
            selected_thread = tid


title_col, new_chat_col, export_col = st.columns([5, 1, 1])
with title_col:
    st.title("Multi Utility Chatbot")
with new_chat_col:
    st.write("")
    st.write("")
    if st.button("New Chat", key="main-new-chat", use_container_width=True):
        reset_chat()
        st.rerun()
with export_col:
    st.write("")
    st.write("")
    with st.popover("Export", use_container_width=True):
        st.caption(build_export_summary(st.session_state["message_history"]))
        export_format = st.radio(
            "Format",
            options=["TXT", "JSON"],
            horizontal=True,
            label_visibility="collapsed",
            key="header-export-format",
        )
        if export_format == "TXT":
            export_data = export_chat_as_text(
                thread_key,
                current_title,
                st.session_state["message_history"],
            )
            export_file_name = build_export_filename(current_title, "txt")
            export_mime = "text/plain"
        else:
            export_data = json.dumps(st.session_state["message_history"], indent=2)
            export_file_name = build_export_filename(current_title, "json")
            export_mime = "application/json"

        st.caption(f"File: {export_file_name}")
        st.download_button(
            "Download Chat",
            data=export_data,
            file_name=export_file_name,
            mime=export_mime,
            use_container_width=True,
            key="header-export-download",
        )

st.caption(f"Current chat: {current_title} | Thread: {thread_key[:8]}...")

doc_meta = get_thread_metadata(thread_key)
if doc_meta:
    doc_caption = (
        f"Document: {doc_meta.get('filename')} | "
        f"{doc_meta.get('chunks')} chunks | {doc_meta.get('documents')} pages"
    )
    if doc_meta.get("uploaded_at"):
        doc_caption += f" | Uploaded: {doc_meta['uploaded_at']}"
    st.caption(doc_caption)

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_citations(message.get("citations", []))


user_input = st.chat_input("Ask about your document or use any tool...")

if user_input:
    set_thread_title(thread_key, user_input)
    touch_thread(thread_key)
    current_title = get_thread_title(thread_key)

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    graph = get_or_build_graph(thread_key)

    with st.chat_message("assistant"):
        stream_state: dict = {"box": None, "citations": []}

        def ai_only_stream():
            chunk_queue: queue.Queue = queue.Queue()
            sentinel = object()

            async def _stream_to_queue():
                try:
                    async for chunk, _ in graph.astream(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                        stream_mode="messages",
                    ):
                        chunk_queue.put(chunk)
                finally:
                    chunk_queue.put(sentinel)

            submit_async_task(_stream_to_queue())

            while True:
                chunk = chunk_queue.get()
                if chunk is sentinel:
                    break

                if isinstance(chunk, ToolMessage):
                    tool_name = getattr(chunk, "name", "tool")
                    tool_citations = extract_citations_from_tool_message(chunk)
                    if tool_citations:
                        stream_state["citations"] = normalize_citations(
                            stream_state["citations"] + tool_citations
                        )

                    if stream_state["box"] is None:
                        stream_state["box"] = st.status(
                            f"Using {tool_name}...",
                            expanded=True,
                        )
                    else:
                        stream_state["box"].update(
                            label=f"Using {tool_name}...",
                            state="running",
                            expanded=True,
                        )

                if (
                    isinstance(chunk, AIMessage)
                    and chunk.content
                    and not chunk.tool_calls
                ):
                    yield chunk.content

        ai_response = st.write_stream(ai_only_stream())

        if stream_state["box"] is not None:
            stream_state["box"].update(
                label="Tool finished",
                state="complete",
                expanded=False,
            )

        render_citations(stream_state["citations"])

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_response,
            "citations": normalize_citations(stream_state["citations"]),
        }
    )

st.divider()

if selected_thread:
    st.session_state["thread_id"] = selected_thread
    st.session_state["message_history"] = load_conversation(selected_thread)
    st.session_state["ingested_docs"].setdefault(str(selected_thread), {})
    st.rerun()
