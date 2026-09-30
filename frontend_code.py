"""
Frontend Module — Ultron Conversational AI UI (v2.7)
=====================================================
Features:
- Identity Login Gateway (Name Input)
- Claude-style Dynamic Greetings (Time-based + Random prompt)
- Robust Dark & Light Themes (Crystal-clear contrast, no visual glitches)
- Persistent API Key Management with Status Badges
- Auto-detection if API Key is pasted in chat
- Multi-LLM Provider Engine (Google Gemini, Local Ollama, Groq)
- Comprehensive Settings Panel
"""

import streamlit as st
from backend_code import (
    generate_llm_response,
    get_dynamic_greeting,
    get_available_ollama_models,
    ULTRON_DEFAULT_PROMPT,
)

# ── Page Configuration ───────────────────────────────────────
st.set_page_config(
    page_title="Ultron",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session State Initialization ─────────────────────────────
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_greeting" not in st.session_state:
    st.session_state.session_greeting = ""

if "app_theme" not in st.session_state:
    st.session_state.app_theme = "🌑 Dark Mode"

if "provider" not in st.session_state:
    st.session_state.provider = "Google Gemini"

if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-3.8-flash"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7

if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = ""

if "groq_key" not in st.session_state:
    st.session_state.groq_key = ""


# ── Curated Clean Theme Definitions ───────────────────────────
is_dark = "Dark" in st.session_state.app_theme

if is_dark:
    theme_css = """
    :root {
        --bg-main: #0b0d12;
        --sidebar-bg: #10131b;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --card-bg: rgba(255, 255, 255, 0.03);
        --card-border: rgba(255, 255, 255, 0.08);
        --accent: #ff3344;
        --accent-glow: rgba(255, 51, 68, 0.4);
        --input-bg: #131722;
        --input-border: rgba(255, 255, 255, 0.14);
    }
    html, body, [class*="css"], .stApp {
        background-color: #0b0d12 !important;
        color: #f1f5f9 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #10131b !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.07) !important;
        color: #f1f5f9 !important;
    }
    .user-pill {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #f1f5f9;
    }
    """
else:
    theme_css = """
    :root {
        --bg-main: #f8fafc;
        --sidebar-bg: #ffffff;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --card-bg: #ffffff;
        --card-border: #e2e8f0;
        --accent: #dc2626;
        --accent-glow: rgba(220, 38, 38, 0.25);
        --input-bg: #ffffff;
        --input-border: #cbd5e1;
    }
    html, body, [class*="css"], .stApp {
        background-color: #f8fafc !important;
        color: #0f172a !important;
    }
    p, span, div, h1, h2, h3, h4, label, .stMarkdown {
        color: #0f172a !important;
    }
    [data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #e2e8f0 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span, [data-testid="stSidebar"] label {
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        color: #0f172a !important;
    }
    [data-testid="stChatMessage"] p {
        color: #0f172a !important;
    }
    .user-pill {
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        color: #0f172a;
    }
    .login-box {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08) !important;
    }
    """

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    {theme_css}

    html, body, [class*="css"], .stApp {{
        font-family: 'Outfit', sans-serif !important;
    }}
    
    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .block-container {{
        padding-top: 2rem;
        padding-bottom: 5.5rem;
        max-width: 860px;
    }}
    
    /* Brand Header */
    .sidebar-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 4px 18px 4px;
        border-bottom: 1px solid var(--card-border);
        margin-bottom: 16px;
    }}

    .brand-icon {{
        font-size: 1.7rem;
        color: var(--accent);
        filter: drop-shadow(0 0 10px var(--accent-glow));
        animation: subtlePulse 3s ease-in-out infinite alternate;
    }}

    .brand-name {{
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        color: var(--text-primary);
        text-transform: uppercase;
    }}

    .brand-badge {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        background: rgba(255, 51, 68, 0.12);
        color: var(--accent);
        padding: 3px 8px;
        border-radius: 20px;
        border: 1px solid var(--accent-glow);
        margin-left: auto;
    }}

    .user-pill {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 0.88rem;
        margin-bottom: 14px;
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes subtlePulse {{
        0% {{ transform: scale(1); }}
        100% {{ transform: scale(1.08); filter: drop-shadow(0 0 12px var(--accent)); }}
    }}

    /* Login Gateway Card */
    .login-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 6vh;
        animation: fadeInUp 0.4s ease forwards;
    }}

    .login-box {{
        border-radius: 24px;
        padding: 38px 32px;
        max-width: 480px;
        width: 100%;
        text-align: center;
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        animation: fadeInUp 0.3s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transition: all 0.2s ease;
    }}

    /* Chat Input Box */
    [data-testid="stChatInput"] {{
        border-radius: 26px !important;
        background: var(--input-bg) !important;
        border: 1px solid var(--input-border) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
        transition: all 0.2s ease !important;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 16px var(--accent-glow) !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        border-color: var(--accent) !important;
    }}

    /* Hero Banner */
    .hero-container {{
        text-align: center;
        padding: 60px 10px 30px 10px;
        animation: fadeInUp 0.4s ease forwards;
    }}

    .hero-title {{
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        color: var(--text-primary);
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════
# STAGE 1: LOGIN GATEWAY
# ═════════════════════════════════════════════════════════════
if not st.session_state.user_name:
    st.markdown(
        """
        <div class="login-wrapper">
            <div class="login-box">
                <div style="font-size: 3rem; margin-bottom: 8px;">🤖</div>
                <h2 style="font-weight: 700; letter-spacing: 2px; margin-bottom: 6px; text-transform: uppercase;">ULTRON</h2>
                <p style="color: var(--text-secondary); font-size: 0.95rem; margin-bottom: 24px;">Please identify yourself to enter the workspace.</p>
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
# STAGE 2: MAIN WORKSPACE
# ═════════════════════════════════════════════════════════════

if not st.session_state.session_greeting:
    st.session_state.session_greeting = get_dynamic_greeting(st.session_state.user_name)

# ── Sidebar Architecture ───────────────────────────────────────
with st.sidebar:
    # 1. Bold Top Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-icon">🤖</span>
            <span class="brand-name">ULTRON</span>
            <span class="brand-badge">v2.7</span>
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

    # Action buttons
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

    st.markdown("<div style='height: 10vh;'></div>", unsafe_allow_html=True)
    st.divider()

    # 2. Settings Panel Anchored at the Bottom
    with st.expander("⚙️ Settings & Configuration", expanded=False):
        # Clean Theme Selector (Dark vs Light)
        selected_theme = st.selectbox(
            "🎨 Interface Theme",
            options=["🌑 Dark Mode", "☀️ Light Mode"],
            index=0 if "Dark" in st.session_state.app_theme else 1,
        )
        if selected_theme != st.session_state.app_theme:
            st.session_state.app_theme = selected_theme
            st.rerun()

        st.divider()

        # Provider Selector
        provider_options = ["Google Gemini", "Ollama (Local & Free)", "Groq"]
        selected_provider = st.selectbox(
            "🧠 AI Engine Provider",
            options=provider_options,
            index=0 if "Gemini" in st.session_state.provider else (1 if "Ollama" in st.session_state.provider else 2),
        )
        st.session_state.provider = selected_provider

        # Model Selection based on Provider
        if "Gemini" in selected_provider:
            model_options = ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            st.session_state.model_name = st.selectbox("📦 Gemini Model", model_options)
            
            gem_input = st.text_input(
                "🔑 Gemini API Key",
                type="password",
                value=st.session_state.gemini_key,
                placeholder="Paste Gemini API key (AIza...)",
                help="Get a free key from Google AI Studio using personal @gmail.com",
            )
            if gem_input != st.session_state.gemini_key:
                st.session_state.gemini_key = gem_input.strip()

            if st.session_state.gemini_key:
                st.success("🟢 Gemini Key Active")
            else:
                st.markdown(
                    """
                    <div style="font-size: 0.8rem; color: #8c93a4; margin-top: 4px;">
                        🌟 <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #ff4d5a; text-decoration: none;">Get Free Gemini Key (AI Studio)</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        elif "Ollama" in selected_provider:
            ollama_models = get_available_ollama_models()
            st.session_state.model_name = st.selectbox("📦 Local Ollama Model", ollama_models)
            st.success("⚡ 100% Free & Local — Zero API key needed!")
            st.caption("Ensure Ollama is running (`ollama serve`).")

        else:  # Groq
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
            st.session_state.model_name = st.selectbox("📦 Groq Model", groq_models)
            grq_input = st.text_input(
                "🔑 Groq API Key",
                type="password",
                value=st.session_state.groq_key,
                placeholder="Paste Groq key (gsk_...)",
            )
            if grq_input != st.session_state.groq_key:
                st.session_state.groq_key = grq_input.strip()

            if st.session_state.groq_key:
                st.success("🟢 Groq Key Active")

        st.divider()

        # Creativity Temperature
        st.session_state.temperature = st.slider(
            "🌡️ Creativity (Temperature)",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="Higher values make output more creative, lower values more precise.",
        )

        system_prompt = st.text_area(
            "🧠 System Personality",
            value=ULTRON_DEFAULT_PROMPT,
            height=90,
        )

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── Clean Hero Banner (Zero Suggestion clutter) ───────────────
if len(st.session_state.messages) == 0:
    provider_label = "Gemini" if "Gemini" in st.session_state.provider else ("Ollama" if "Ollama" in st.session_state.provider else "Groq")
    st.markdown(
        f"""
        <div class="hero-container">
            <div style="font-size: 3rem; margin-bottom: 8px;">🤖</div>
            <div class="hero-title">{st.session_state.session_greeting}</div>
            <div class="hero-subtitle">Ultron is ready • Powered by {provider_label} ({st.session_state.model_name})</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Render Chat History ───────────────────────────────────────
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ── Chat Input & Streaming LLM Response ───────────────────────
if user_prompt := st.chat_input("Message Ultron..."):
    # Check if user accidentally pasted their API key in the main chat prompt
    if user_prompt.strip().startswith("AIzaSy"):
        st.session_state.gemini_key = user_prompt.strip()
        st.success("✅ Gemini API Key detected and saved! You can now chat with Ultron.")
        st.rerun()
    elif user_prompt.strip().startswith("gsk_"):
        st.session_state.groq_key = user_prompt.strip()
        st.success("✅ Groq API Key detected and saved! You can now chat with Ultron.")
        st.rerun()

    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # 2. Stream Ultron's response from backend
    provider_name = "Gemini" if "Gemini" in st.session_state.provider else ("Ollama" if "Ollama" in st.session_state.provider else "Groq")
    active_key = st.session_state.gemini_key if provider_name == "Gemini" else st.session_state.groq_key

    with st.chat_message("assistant", avatar="🤖"):
        response_generator = generate_llm_response(
            messages=st.session_state.messages,
            provider=provider_name,
            model_name=st.session_state.model_name,
            api_key=active_key,
            system_prompt=system_prompt if 'system_prompt' in locals() else ULTRON_DEFAULT_PROMPT,
            temperature=st.session_state.temperature,
        )
        full_response = st.write_stream(response_generator)

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
