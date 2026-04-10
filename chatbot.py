from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
import PyPDF2

# ---------- LOAD ENV ----------
load_dotenv()

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Rebin's Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
    <style>
    .title {
        text-align: center;
        font-size: 38px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: gray;
        margin-bottom: 20px;
    }
    .user-msg {
        background-color: #1f2937;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: right;
        color: white;
    }
    .bot-msg {
        background-color: #111827;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        color: #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.markdown("<div class='title'>🤖 Rebin's Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Ask anything... I'm here to help 🚀</div>", unsafe_allow_html=True)

# ---------- CLEAR CHAT ----------
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.pop("file_text", None)
    st.rerun()

st.divider()

# ===============================
# 🔥 SIDEBAR MULTIPLE FILE UPLOAD
# ===============================
st.sidebar.header("📂 Upload Files")

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["txt", "pdf"],
    accept_multiple_files=True   # 🔥 IMPORTANT
)

# ---------- HANDLE MULTIPLE FILES ----------
if uploaded_files:
    all_text = ""

    for file in uploaded_files:
        if file.type == "text/plain":
            all_text += file.read().decode("utf-8") + "\n\n"

        elif file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                all_text += page.extract_text() + "\n\n"

    st.session_state.file_text = all_text
    st.sidebar.success(f"✅ {len(uploaded_files)} files uploaded!")

# ---------- SHOW STATUS ----------
if "file_text" in st.session_state:
    st.info("📄 Files loaded. Ask questions below.")

# ---------- CHAT HISTORY ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- DISPLAY CHAT ----------
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"<div class='user-msg'>👤 {message['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {message['content']}</div>", unsafe_allow_html=True)

# ---------- LLM ----------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
)

# ---------- INPUT ----------
user_prompt = st.chat_input("Type your message...")

if user_prompt:
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_prompt
    })

    st.markdown(f"<div class='user-msg'>👤 {user_prompt}</div>", unsafe_allow_html=True)

    messages = [
        {"role": "system", "content": "You are a helpful assistant"}
    ]

    # 🔥 include ALL files content
    if "file_text" in st.session_state:
        messages.append({
            "role": "user",
            "content": f"Use these files:\n{st.session_state.file_text}"
        })

    messages += st.session_state.chat_history

    with st.spinner("Thinking... 🤔"):
        response = llm.invoke(messages)

    reply = response.content

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": reply
    })

    st.markdown(f"<div class='bot-msg'>🤖 {reply}</div>", unsafe_allow_html=True)
