"""
Frontend Module — Ultron Conversational AI UI (v2.5)
=====================================================
Features:
- Identity Login Gateway (Name Input)
- Claude-style Dynamic Greetings (Time-based + Random prompt)
- Multi-Theme Engine (Obsidian Dark, Clean Light, Cyberpunk Neon)
- Multi-LLM Provider Engine (Google Gemini, Local Ollama, Groq)
- Comprehensive Settings Panel (Theme, Provider, Model, Temperature, Prompt, API Key)
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
    st.session_state.app_theme = "🌑 Obsidian Dark"

if "provider" not in st.session_state:
    st.session_state.provider = "Google Gemini"

if "model_name" not in st.session_state:
    st.session_state.model_name = "gemini-2.5-flash"

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.7


# ── Dynamic Theme CSS ─────────────────────────────────────────
THEMES = {
    "🌑 Obsidian Dark": """
        --bg-main: #0c0e12;
        --sidebar-bg: linear-gradient(180deg, #0d0f14 0%, #12151d 100%);
        --text-color: #f0f2f6;
        --subtext-color: #8c93a4;
        --card-bg: rgba(255, 255, 255, 0.025);
        --card-border: rgba(255, 255, 255, 0.06);
        --accent-glow: #ff3344;
        --accent-glow-rgba: rgba(255, 51, 68, 0.4);
        --input-bg: rgba(18, 22, 31, 0.9);
        --input-border: rgba(255, 255, 255, 0.12);
        --chip-bg: rgba(255, 255, 255, 0.03);
    """,
    "☀️ Clean Light": """
        --bg-main: #f8f9fc;
        --sidebar-bg: linear-gradient(180deg, #f1f3f8 0%, #e9edf5 100%);
        --text-color: #1a1d24;
        --subtext-color: #5d6474;
        --card-bg: #ffffff;
        --card-border: rgba(0, 0, 0, 0.08);
        --accent-glow: #e63946;
        --accent-glow-rgba: rgba(230, 57, 70, 0.3);
        --input-bg: #ffffff;
        --input-border: rgba(0, 0, 0, 0.15);
        --chip-bg: #ffffff;
    """,
    "🌌 Cyberpunk Neon": """
        --bg-main: #06070a;
        --sidebar-bg: linear-gradient(180deg, #0a0c14 0%, #0d111c 100%);
        --text-color: #00f2fe;
        --subtext-color: #7b88a8;
        --card-bg: rgba(0, 242, 254, 0.03);
        --card-border: rgba(0, 242, 254, 0.15);
        --accent-glow: #00f2fe;
        --accent-glow-rgba: rgba(0, 242, 254, 0.5);
        --input-bg: rgba(10, 15, 26, 0.95);
        --input-border: rgba(0, 242, 254, 0.25);
        --chip-bg: rgba(0, 242, 254, 0.05);
    """,
}

current_theme_vars = THEMES.get(st.session_state.app_theme, THEMES["🌑 Obsidian Dark"])

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        {current_theme_vars}
    }}

    html, body, [class*="css"], .stApp {{
        font-family: 'Outfit', sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-color) !important;
    }}
    
    code, pre {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    .block-container {{
        padding-top: 1.8rem;
        padding-bottom: 5.5rem;
        max-width: 900px;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
    }}
    
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
        color: var(--accent-glow);
        filter: drop-shadow(0 0 10px var(--accent-glow-rgba));
        animation: subtlePulse 3s ease-in-out infinite alternate;
    }}

    .brand-name {{
        font-size: 1.45rem;
        font-weight: 700;
        letter-spacing: 2.5px;
        color: var(--text-color);
        text-transform: uppercase;
    }}

    .brand-badge {{
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1px;
        background: rgba(255, 51, 68, 0.15);
        color: var(--accent-glow);
        padding: 3px 8px;
        border-radius: 20px;
        border: 1px solid var(--accent-glow-rgba);
        margin-left: auto;
    }}

    .user-pill {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        padding: 8px 12px;
        border-radius: 12px;
        font-size: 0.88rem;
        color: var(--text-color);
        margin-bottom: 14px;
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes subtlePulse {{
        0% {{ transform: scale(1); filter: drop-shadow(0 0 4px var(--accent-glow-rgba)); }}
        100% {{ transform: scale(1.08); filter: drop-shadow(0 0 14px var(--accent-glow)); }}
    }}

    /* Login Gateway Card */
    .login-wrapper {{
        display: flex;
        justify-content: center;
        align-items: center;
        padding-top: 5vh;
        animation: fadeInUp 0.5s ease forwards;
    }}

    .login-box {{
        background: var(--input-bg);
        backdrop-filter: blur(16px);
        border: 1px solid var(--card-border);
        border-radius: 24px;
        padding: 36px 32px;
        max-width: 480px;
        width: 100%;
        text-align: center;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.4), 0 0 30px var(--accent-glow-rgba);
    }}

    /* Chat Messages */
    [data-testid="stChatMessage"] {{
        background: var(--card-bg) !important;
        border: 1px solid var(--card-border) !important;
        border-radius: 18px;
        padding: 16px 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        animation: fadeInUp 0.35s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        transition: all 0.2s ease;
    }}

    /* Chat Input Box */
    [data-testid="stChatInput"] {{
        border-radius: 26px !important;
        background: var(--input-bg) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid var(--input-border) !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25) !important;
        transition: all 0.25s ease !important;
    }}

    [data-testid="stChatInput"]:focus-within {{
        border-color: var(--accent-glow) !important;
        box-shadow: 0 0 20px var(--accent-glow-rgba) !important;
    }}

    /* Buttons */
    .stButton > button {{
        border-radius: 14px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        border: 1px solid var(--card-border) !important;
    }}

    .stButton > button:hover {{
        transform: translateY(-1px);
        border-color: var(--accent-glow) !important;
        box-shadow: 0 4px 14px var(--accent-glow-rgba) !important;
    }}

    /* Hero Banner */
    .hero-container {{
        text-align: center;
        padding: 30px 10px 24px 10px;
        animation: fadeInUp 0.4s ease forwards;
    }}

    .hero-title {{
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: var(--text-color);
        margin-bottom: 8px;
    }}

    .hero-subtitle {{
        color: var(--subtext-color);
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 24px;
    }}

    .prompt-chip {{
        display: inline-block;
        padding: 10px 16px;
        margin: 6px;
        background: var(--chip-bg);
        border: 1px solid var(--card-border);
        border-radius: 16px;
        font-size: 0.88rem;
        color: var(--text-color);
        transition: all 0.2s ease;
        text-align: left;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ═════════════════════════════════════════════════════════════
# STAGE 1: LOGIN GATEWAY (NAME IDENTITY SCREEN)
# ═════════════════════════════════════════════════════════════
if not st.session_state.user_name:
    st.markdown(
        """
        <div class="login-wrapper">
            <div class="login-box">
                <div style="font-size: 3rem; margin-bottom: 8px;">🤖</div>
                <h2 style="font-weight: 700; letter-spacing: 2px; margin-bottom: 6px; text-transform: uppercase;">ULTRON</h2>
                <p style="color: var(--subtext-color); font-size: 0.95rem; margin-bottom: 24px;">Please identify yourself to enter the workspace.</p>
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

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    # 1. Bold Top Brand
    st.markdown(
        """
        <div class="sidebar-brand">
            <span class="brand-icon">🤖</span>
            <span class="brand-name">ULTRON</span>
            <span class="brand-badge">v2.5</span>
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

    # 2. Rich Settings Panel at the Bottom
    with st.expander("⚙️ Settings & Configuration", expanded=False):
        # Theme Selector
        selected_theme = st.selectbox(
            "🎨 Interface Theme",
            options=["🌑 Obsidian Dark", "☀️ Clean Light", "🌌 Cyberpunk Neon"],
            index=["🌑 Obsidian Dark", "☀️ Clean Light", "🌌 Cyberpunk Neon"].index(st.session_state.app_theme),
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
            model_options = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
            st.session_state.model_name = st.selectbox("📦 Gemini Model", model_options)
            
            api_key_input = st.text_input(
                "🔑 Gemini API Key",
                type="password",
                placeholder="Paste Gemini API key (AIza...)",
                help="Get a free key from Google AI Studio using a personal @gmail.com account.",
            )
            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #8c93a4; margin-top: 4px;">
                    🌟 <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color: #ff4d5a; text-decoration: none;">Get Free Gemini Key (AI Studio)</a>
                    <br><span style="color: #6c7384;">*Use personal Gmail account if student account shows permission denied.*</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif "Ollama" in selected_provider:
            ollama_models = get_available_ollama_models()
            st.session_state.model_name = st.selectbox("📦 Local Ollama Model", ollama_models)
            api_key_input = ""
            st.success("⚡ 100% Free & Local — Zero API key required!")
            st.caption("Ensure Ollama is running (`ollama serve` or Ollama desktop app).")

        else:  # Groq
            groq_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]
            st.session_state.model_name = st.selectbox("📦 Groq Model", groq_models)
            api_key_input = st.text_input(
                "🔑 Groq API Key",
                type="password",
                placeholder="Paste Groq key (gsk_...)",
            )
            st.markdown(
                """
                <div style="font-size: 0.8rem; color: #8c93a4; margin-top: 4px;">
                    ⚡ <a href="https://console.groq.com/keys" target="_blank" style="color: #ff4d5a; text-decoration: none;">Get Free Groq Key (Instant)</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

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


# ── Welcome Hero Banner with Claude Dynamic Greeting ───────────
if len(st.session_state.messages) == 0:
    provider_label = "Gemini" if "Gemini" in st.session_state.provider else ("Ollama" if "Ollama" in st.session_state.provider else "Groq")
    st.markdown(
        f"""
        <div class="hero-container">
            <div style="font-size: 2.6rem; margin-bottom: 6px;">🤖</div>
            <div class="hero-title">{st.session_state.session_greeting}</div>
            <div class="hero-subtitle">Ultron is active ({provider_label} • {st.session_state.model_name}) and ready to assist with code, reasoning, research, and analysis.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="prompt-chip">💡 <strong>Architecture:</strong> Build a Python REST API</div>
            <div class="prompt-chip">🔬 <strong>Deep Reasoning:</strong> Explain quantum entanglement</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="prompt-chip">⚡ <strong>Code Optimization:</strong> Refactor script for speed</div>
            <div class="prompt-chip">📝 <strong>Creative Brief:</strong> Draft a project proposal for AI</div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)


# ── Render Chat History ───────────────────────────────────────
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


# ── Chat Input & Streaming LLM Response ───────────────────────
if user_prompt := st.chat_input("Message Ultron..."):
    # 1. Append & render user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # 2. Stream Ultron's response from backend
    provider_name = "Gemini" if "Gemini" in st.session_state.provider else ("Ollama" if "Ollama" in st.session_state.provider else "Groq")
    with st.chat_message("assistant", avatar="🤖"):
        response_generator = generate_llm_response(
            messages=st.session_state.messages,
            provider=provider_name,
            model_name=st.session_state.model_name,
            api_key=api_key_input if 'api_key_input' in locals() else None,
            system_prompt=system_prompt if 'system_prompt' in locals() else ULTRON_DEFAULT_PROMPT,
            temperature=st.session_state.temperature,
        )
        full_response = st.write_stream(response_generator)

    # 3. Save assistant message to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
