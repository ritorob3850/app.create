"""
Frontend Module — AI Chat Assistant Dashboard (ChatGPT / Claude Style)
=====================================================================
Renders a modern, responsive conversational UI connected to the LLM backend.
"""

import streamlit as st
from backend_code import generate_llm_response

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for ChatGPT/Claude Look ────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Controls ──────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ LLM Settings")
    st.caption("ChatGPT / Claude style AI Chatbot")

    api_key_input = st.text_input(
        "🔑 LLM API Key (Groq or Gemini)",
        type="password",
        placeholder="Paste Groq or Gemini API key...",
        help="Paste a free key from Groq or Google AI Studio.",
    )

    st.markdown(
        """
        **Get a Free API Key:**
        - ⚡ [Get Groq Key (Instant & Free)](https://console.groq.com/keys)
        - 🌟 [Get Gemini Key (Google AI Studio)](https://aistudio.google.com/app/apikey) *(use personal @gmail.com)*
        """
    )

    st.divider()

    system_prompt = st.text_area(
        "🧠 System Personality",
        value="You are an intelligent, helpful, and concise AI assistant like ChatGPT and Claude.",
        help="Set custom behavior or role for the chatbot.",
    )

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header ───────────────────────────────────────────────────
st.markdown('<div class="main-title">🤖 AI Chat Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Fully conversational LLM application like ChatGPT and Claude</div>', unsafe_allow_html=True)

# ── Session State for Chat History ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your AI assistant. How can I help you today?"}
    ]

# ── Display Conversation History ──────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat Input & Streaming LLM Response ───────────────────────
if user_prompt := st.chat_input("Ask anything (e.g. Write Python code, explain quantum physics, brainstorm ideas)..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Generate and stream assistant response from backend
    with st.chat_message("assistant"):
        response_generator = generate_llm_response(
            messages=st.session_state.messages,
            api_key=api_key_input,
            system_prompt=system_prompt,
        )
        full_response = st.write_stream(response_generator)

    # 3. Save assistant message to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
