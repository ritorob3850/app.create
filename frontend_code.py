"""
Frontend Module — Ultron Conversational AI UI
==============================================
Features:
- Identity Login Gateway (Name Prompt)
- Claude-style Dynamic Time-Based Greetings (e.g. 'Good afternoon, ritorob. What are we building today?')
- Polished Minimalist Dark Glassmorphic Theme with Curved Elements & Fluid Animations
- Top Bold Sidebar Brand and Bottom Anchored Settings Panel
"""

import streamlit as st
from backend_code import generate_llm_response, get_dynamic_greeting, ULTRON_DEFAULT_PROMPT

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Ultron",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Master Aesthetic CSS ──────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Main Container */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 5.5rem;
        max-width: 900px;
    }

    /* Sidebar Theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d0f14 0%, #12151d 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 4px 18px 4px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 16px;
    }

    .brand-icon {
        font-size: 1.7rem;
        color: #ff3344;
        filter: drop-shadow(0 0 10px rgba(255, 51, 68, 0.65));
        animation: subtlePulse 3s ease-in-out infinite alternate;
    }

    .brand-name {
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 2.5px;
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

    .user-pill {
        display: flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 0.88rem;
        color: #e1e4ea;
        margin-bottom: 14px;
    }

    /* Keyframe Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(12px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes subtlePulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 4px rgba(255, 51, 68, 0.4)); }
        100% { transform: scale(1.08); filter: drop-shadow(0 0 14px rgba(255, 51, 68, 0.85)); }
    }

    /* Login Gateway Card */
    .login-wrapper {
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 5vh;
        animation: fadeInUp 0.5s ease forwards;
    }

    .login-box {
        background: rgba(18, 22, 31, 0.85);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px;
        padding: 36px 32px;
        max-width: 480px;
        width: 100%;
        text-align: center;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5), 0 0 30px rgba(255, 51, 68, 0.08);
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
        animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transition: all 0.2s ease;
    }

    [data-testid="stChatMessage"]:hover {
        border-color: rgba(255, 255, 255, 0.09);
        background: rgba(255, 255, 255, 0.035);
    }

    /* Chat Input Box */
    [data-testid="stChatInput"] {
        border-radius: 26px !important;
        background: rgba(18, 22, 31, 0.9) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        transition: all 0.25s ease !important;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(255, 51, 68, 0.6) !important;
        box-shadow: 0 0 20px rgba(255, 51, 68, 0.22) !important;
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
        padding: 30px 10px 24px 10px;
        animation: fadeInUp 0.4s ease forwards;
    }

    .hero-title {
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        background: linear-gradient(135deg, #ffffff 40%, #a2a8b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }

    .hero-subtitle {
        color: #7e8696;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 24px;
    }

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
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State Initialization ─────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_greeting" not in st.session_state:
    st.session_state.session_greeting = ""


# ═════════════════════════════════════════════════════════════
# STAGE 1: LOGIN GATEWAY (DASHBOARD NAME SCREEN)
# ═════════════════════════════════════════════════════════════
if not st.session_state.user_name:
    st.markdown(
        """
        <div class="login-wrapper">
            <div class="login-box">
                <div style="font-size: 3rem; margin-bottom: 8px; filter: drop-shadow(0 0 14px rgba(255,51,68,0.7));">⬡</div>
                <h2 style="font-weight: 700; letter-spacing: 2px; margin-bottom: 6px; text-transform: uppercase;">ULTRON</h2>
                <p style="color: #8c93a4; font-size: 0.95rem; margin-bottom: 24px;">Please identify yourself to enter the workspace.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([1, 2, 1])
    with col_b:
        name_input = st.text_input(
            "✏️ What is your name?",
            placeholder="e.g. ritorob, Alex, Jordan...",
            label_visibility="collapsed",
        )
        if st.button("🚀 Initialize Ultron", use_container_width=True, type="primary"):
            if name_input.strip():
                st.session_state.user_name = name_input.strip()
                st.session_state.session_greeting = get_dynamic_greeting(st.session_state.user_name)
                st.rerun()
            else:
                st.warning("Please enter your name to proceed.")
    st.stop()


# ═════════════════════════════════════════════════════════════
# STAGE 2: MAIN ULTRON CONVERSATIONAL WORKSPACE
# ═════════════════════════════════════════════════════════════

# Ensure dynamic greeting is initialized
if not st.session_state.session_greeting:
    st.session_state.session_greeting = get_dynamic_greeting(st.session_state.user_name)

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

    # User Profile Pill
    st.markdown(
        f"""
        <div class="user-pill">
            <span>👤</span>
            <span><strong>{st.session_state.user_name.title()}</strong></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # New Chat & Switch User Actions
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("＋ New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_greeting = get_dynamic_greeting(st.session_state.user_name)
            st.rerun()
    with col_s2:
        if st.button("🔄 Switch User", use_container_width=True):
            st.session_state.user_name = ""
            st.session_state.messages = []
            st.rerun()

    # Dynamic vertical space to push settings to the bottom
    st.markdown("<div style='height: 18vh;'></div>", unsafe_allow_html=True)
    st.divider()

    # 2. Settings Panel Anchored at the Bottom
    with st.expander("⚙️ Settings & API Key", expanded=False):
        api_key_input = st.text_input(
            "🔑 API Key",
            type="password",
            placeholder="Paste Groq or Gemini API key...",
            help="Leave blank if configured in secrets or env variables.",
        )

        st.markdown(
            """
            <div style="font-size: 0.8rem; color: #8c93a4; margin-top: 6px;">
                ⚡ <a href="https://console.groq.com/keys" target="_blank" style="color: #ff4d5a; text-decoration: none;">Get Free Groq Key (Instant)</a><br>
                🌟 <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #cfd3dc; text-decoration: none;">Get Gemini Key (Google AI)</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        system_prompt = st.text_area(
            "🧠 Personality Prompt",
            value=ULTRON_DEFAULT_PROMPT,
            height=100,
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Welcome Hero Banner with Claude-style Dynamic Greeting ─────
if len(st.session_state.messages) == 0:
    st.markdown(
        f"""
        <div class="hero-container">
            <div style="font-size: 2.6rem; margin-bottom: 6px;">⬡</div>
            <div class="hero-title">{st.session_state.session_greeting}</div>
            <div class="hero-subtitle">Ultron is active and ready to assist with code, reasoning, research, and design.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="prompt-chip">💡 <strong>Architecture:</strong> Design a microservice backend</div>
            <div class="prompt-chip">🔬 <strong>Deep Reasoning:</strong> Explain quantum entanglement simply</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="prompt-chip">⚡ <strong>Code Optimization:</strong> Refactor a Python script for speed</div>
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


# ── Chat Input & Streaming LLM Response ───────────────────────
if user_prompt := st.chat_input("Message Ultron..."):
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # 2. Stream Ultron's response from backend
    with st.chat_message("assistant", avatar="⬡"):
        response_generator = generate_llm_response(
            messages=st.session_state.messages,
            api_key=api_key_input,
            system_prompt=system_prompt,
        )
        full_response = st.write_stream(response_generator)

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
