from dotenv import load_dotenv
import streamlit as st
import PyPDF2
from PIL import Image
import requests
import base64
import os
import io

# ---------- LOAD ENV ----------
load_dotenv()
GROQ_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_KEY:
    st.error("❌ GROQ_API_KEY missing. Check .env file")
    st.stop()

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Rebin's QA Chatbot", page_icon="🧪")
st.markdown("<h2 style='text-align:center;'>🧪 Rebin's QA Chatbot</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Upload Screenshots, PDFs, Docs → Get Test Cases instantly 🚀</p>", unsafe_allow_html=True)

# ---------- SESSION STATE INIT ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "images" not in st.session_state:
    st.session_state.images = []
if "file_text" not in st.session_state:
    st.session_state.file_text = ""
if "prev_file_names" not in st.session_state:
    st.session_state.prev_file_names = []

# ---------- SIDEBAR ----------
st.sidebar.header("📂 Upload Files")
st.sidebar.markdown("*Supports: Screenshots, PDFs, TXT*")

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["txt", "pdf", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

# ---------- QUICK PROMPT BUTTONS ----------
st.sidebar.markdown("---")
st.sidebar.markdown("**⚡ Quick Prompts**")

quick_prompt = None
if st.sidebar.button("📋 Generate Test Cases"):
    quick_prompt = "Analyze the uploaded screenshot/document and generate detailed test cases in proper format with Test ID, Description, Steps, Expected Result and Priority."
if st.sidebar.button("🐛 Find Bug Scenarios"):
    quick_prompt = "Based on the uploaded content, identify all possible bug scenarios and edge cases a QA should test."
if st.sidebar.button("✅ Smoke Test Cases"):
    quick_prompt = "Generate smoke test cases for the uploaded content covering only the critical functionality."
if st.sidebar.button("🔁 Regression Test Cases"):
    quick_prompt = "Generate a full regression test suite based on the uploaded screenshots or documents."
if st.sidebar.button("📝 Write BDD Scenarios"):
    quick_prompt = "Convert the uploaded content into BDD test scenarios using Given/When/Then format."

# ---------- CLEAR BUTTON ----------
st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat & Files"):
    st.session_state.chat_history = []
    st.session_state.images = []
    st.session_state.file_text = ""
    st.session_state.prev_file_names = []
    st.rerun()

# ---------- DETECT NEW FILES & RESET ----------
current_file_names = [f.name for f in uploaded_files] if uploaded_files else []

if current_file_names != st.session_state.prev_file_names:
    st.session_state.chat_history = []
    st.session_state.images = []
    st.session_state.file_text = ""
    st.session_state.prev_file_names = current_file_names

    if uploaded_files:
        all_text = ""
        for file in uploaded_files:
            if file.type == "text/plain":
                all_text += f"=== FILE: {file.name} ===\n"
                all_text += file.read().decode("utf-8") + "\n\n"

            elif file.type == "application/pdf":
                try:
                    all_text += f"=== PDF: {file.name} ===\n"
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            all_text += text + "\n\n"
                except Exception as e:
                    st.sidebar.error(f"❌ Error reading {file.name}: {e}")

            elif file.type in ["image/png", "image/jpg", "image/jpeg", "image/webp"]:
                image = Image.open(file)
                st.sidebar.image(image, caption=file.name, use_column_width=True)
                st.session_state.images.append(image)

        st.session_state.file_text = all_text.strip()

        if st.session_state.file_text:
            st.sidebar.success("✅ Document(s) loaded!")
        if st.session_state.images:
            st.sidebar.success(f"✅ {len(st.session_state.images)} screenshot(s) loaded!")

st.divider()

# ---------- STATUS ----------
col1, col2 = st.columns(2)
if st.session_state.file_text:
    col1.info("📄 Document loaded")
if st.session_state.images:
    col2.info(f"🖼️ {len(st.session_state.images)} screenshot(s) loaded")

# ---------- DISPLAY CHAT ----------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===============================
# 🔧 HELPER: image to base64
# ===============================
def image_to_base64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((1024, 1024))
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ===============================
# 🤖 GROQ API CALL
# ===============================
def call_groq(messages: list, images: list = None) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }

    if images:
        model = "meta-llama/llama-4-scout-17b-16e-instruct"
        api_messages = []
        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg["role"] == "user":
                content = [{"type": "text", "text": msg["content"]}]
                for img in images:
                    b64 = image_to_base64(img)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                api_messages.append({"role": "user", "content": content})
            else:
                api_messages.append(msg)
    else:
        model = "llama-3.3-70b-versatile"
        api_messages = messages

    payload = {
        "model": model,
        "messages": api_messages,
        "max_tokens": 2048,
        "temperature": 0.3,
    }

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        return "❌ Request timed out. Please try again."
    except requests.exceptions.HTTPError:
        return f"❌ API error {response.status_code}: {response.text}"
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"


# ===============================
# 💬 CHAT INPUT
# ===============================
user_prompt = st.chat_input("e.g. Generate test cases for this screenshot...")

# Use quick prompt if sidebar button was clicked
active_prompt = quick_prompt or user_prompt

if active_prompt:

    with st.chat_message("user"):
        st.markdown(active_prompt)
    st.session_state.chat_history.append({"role": "user", "content": active_prompt})

    # Detect if question is about image
    image_keywords = [
        "image", "photo", "picture", "screenshot", "screen", "this",
        "what is", "what's", "describe", "show", "see", "look",
        "identify", "tell me about", "what do you see", "whats in",
        "test case", "generate", "analyze", "analyse", "ui", "button",
        "form", "field", "page", "login", "bug", "scenario", "bdd",
        "smoke", "regression", "find", "check"
    ]
    use_image = (
        st.session_state.images and
        any(word in active_prompt.lower() for word in image_keywords)
    )

    # ---- SYSTEM PROMPT (QA FOCUSED) ----
    system_content = """You are an expert QA Engineer assistant with deep knowledge of:
- Manual and automation testing
- Test case design techniques
- Bug reporting best practices
- BDD/TDD methodologies
- UI/UX testing

When generating test cases, ALWAYS use this format:

**TC_ID** | **Test Case Title**
- **Description:** what is being tested
- **Preconditions:** what needs to be set up first
- **Test Steps:**
  1. Step one
  2. Step two
  3. Step three
- **Expected Result:** what should happen
- **Priority:** High / Medium / Low
- **Type:** Functional / UI / Negative / Edge Case

When analyzing screenshots:
- Identify all UI elements (buttons, fields, dropdowns, etc.)
- Note any visible text, labels, error messages
- Suggest both positive and negative test cases
- Include edge cases and boundary value tests
- Consider accessibility and usability issues
"""

    if st.session_state.file_text:
        system_content += f"""

You also have the following document content to reference:

--- DOCUMENT START ---
{st.session_state.file_text[:6000]}
--- DOCUMENT END ---

Use this document along with any screenshots to generate comprehensive test cases.
"""

    # ---- BUILD MESSAGES ----
    api_messages = [{"role": "system", "content": system_content}]

    history = st.session_state.chat_history[:-1]
    if len(history) > 10:
        history = history[-10:]
    for msg in history:
        api_messages.append(msg)

    api_messages.append({"role": "user", "content": active_prompt})

    # ---- CALL GROQ ----
    with st.chat_message("assistant"):
        with st.spinner("Analyzing... 🔍"):
            if use_image:
                reply = call_groq(api_messages, images=st.session_state.images)
            else:
                reply = call_groq(api_messages)
        st.markdown(reply)

    st.session_state.chat_history.append({"role": "assistant", "content": reply})

