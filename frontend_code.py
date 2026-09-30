"""
Frontend Module — Ultron Conversational AI UI
==============================================
A sleek, modern ChatGPT/Claude-styled UI with curated typography, smooth
curved containers, fluid animations, and minimalist aesthetic.
"""

import streamlit as st
from backend_code import generate_llm_response, ULTRON_DEFAULT_PROMPT

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Ultron",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sleek Ultron Aesthetic CSS ───────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        max-width: 920px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #12161f 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 4px 18px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }

    .brand-icon {
        font-size: 1.6rem;
        color: #ff3344;
        filter: drop-shadow(0 0 8px rgba(255, 51, 68, 0.6));
        animation: subtlePulse 3s ease-in-out infinite alternate;
    }

    .brand-name {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #ffffff;
        text-transform: uppercase;
    }

    .brand-badge {
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        background: rgba(255, 51, 68, 0.15);
        color: #ff4d5a;
        padding: 3px 8px;
        border-radius: 20px;
        border: 1px solid rgba(255, 51, 68, 0.3);
        margin-left: auto;
    }

    /* Smooth Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes subtlePulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 4px rgba(255, 51, 68, 0.4)); }
        100% { transform: scale(1.06); filter: drop-shadow(0 0 12px rgba(255, 51, 68, 0.8)); }
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        animation: fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transition: all 0.2s ease;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: rgba(255, 255, 255, 0.1);
        background: rgba(255, 255, 255, 0.035);
    }

    /* User Message Bubble Accent */
    [data-testid="stChatMessage"][data-test-role="user"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.09);
    }

    /* Streamlit Chat Input Box */
    [data-testid="stChatInput"] {
        border-radius: 26px !important;
        background: rgba(18, 22, 31, 0.85) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(255, 60, 75, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 51, 68, 0.25) !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 14px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(255, 51, 68, 0.5) !important;
        box-shadow: 0 4px 14px rgba(255, 51, 68, 0.18) !important;
    }

    /* Hero Banner */
    .hero-container {
        text-align: center;
        padding: 40px 10px 30px 10px;
        animation: fadeInUp 0.4s ease forwards;
    }

    .hero-title {
        font-size: 2.6rem;
        font-weight: 700;
        letter-spacing: 1px;
        background: linear-gradient(135deg, #ffffff 30%, #a0a5b5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #7d8494;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* Prompt Suggestion Cards */
    .prompt-chip {
        display: inline-block;
        padding: 10px 16px;
        margin: 6px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        font-size: 0.88rem;
        color: #cfd3dc;
        transition: all 0.2s ease;
        text-align: left;
    }

    /* Sidebar Footer Settings Panel */
    .settings-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 12px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar Architecture ───────────────────────────────────────
with st.sidebar:
    # 1. AI Name at the Top of Sidebar (Bold & Styled)
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-icon">⬡</span>
            <span class="brand-name">ULTRON</span>
            <span class="brand-badge">AI v2.0</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New Chat action
    if st.button("＋  New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Dynamic Spacing to push settings to the bottom
    st.markdown("<div style='height: 22vh;'></div>", unsafe_allow_html=True)
    st.divider()

    # 2. Settings Panel at the Bottom of Sidebar
    with st.expander("⚙️ Settings & API Key", expanded=False):
        api_key_input = st.text_input(
            "🔑 API Key",
            type="password",
            placeholder="Paste Groq or Gemini API key...",
            help="Supports free keys from Groq or Google AI Studio.",
        )

        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #8c93a4; margin-top: 6px;">
                ⚡ <a href="https://console.groq.com/keys" target="_blank" style="color: #ff4d5a; text-decoration: none;">Get Free Groq Key (Instant)</a><br>
                🌟 <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #cfd3dc; text-decoration: none;">Get Gemini Key (Google)</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        system_prompt = st.text_area(
            "🧠 Personality Prompt",
            value=ULTRON_DEFAULT_PROMPT,
            height=100,
        )

        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Session State for Chat History ────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Welcome Hero Display when Empty ───────────────────────────
if len(st.session_state.messages) == 0:
    st.markdown(
        """
        <div class="hero-container">
            <div style="font-size: 2.8rem; margin-bottom: 4px;">⬡</div>
            <div class="hero-title">How can Ultron assist you today?</div>
            <div class="hero-subtitle">Sophisticated intelligence for coding, analysis, creative generation, and deep reasoning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="prompt-chip">💡 <strong>Code Architecture:</strong> Build a Python REST API</div>
            <div class="prompt-chip">🔬 <strong>Deep Explanation:</strong> Explain transformer attention mechanisms</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="prompt-chip">⚡ <strong>Algorithm Optimization:</strong> Refactor sorting algorithm</div>
            <div class="prompt-chip">📝 <strong>Creative Brief:</strong> Draft a project proposal for batch AI</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ── Render Chat History ───────────────────────────────────────
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "⬡"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# ── Chat Input & Streaming ────────────────────────────────────
if user_prompt := st.chat_input("Message Ultron..."):
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # 2. Stream Ultron's response
    with st.chat_message("assistant", avatar="⬡"):
        response_generator = generate_llm_response(
            messages=st.session_state.messages,
            api_key=api_key_input,
            system_prompt=system_prompt,
        )
        full_response = st.write_stream(response_generator)

    # 3. Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
