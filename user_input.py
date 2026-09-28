"""
Frontend Module — Streamlit Dashboard GUI
==========================================
Gets the user's name as input, sends it to the backend
for processing, and displays the hello message on a dashboard.
"""

import streamlit as st
from backend import process_username

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Welcome Dashboard",
    page_icon="🎉",
    layout="centered",
)

# ── Custom Styling ───────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        color: #4F8BF9;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .greeting-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 600;
        margin-top: 1.5rem;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.35);
    }
    .info-card {
        background: #f0f2f6;
        padding: 1.2rem;
        border-radius: 0.8rem;
        text-align: center;
        margin-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Dashboard Header ────────────────────────────────────────
st.markdown('<div class="main-title">🎉 Welcome Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Enter your name below and get a personalised greeting!</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── User Input Section ──────────────────────────────────────
name_input = st.text_input(
    "👤 Your Name",
    placeholder="Type your name here...",
    help="Enter your name and click the button to see your greeting.",
)

submit = st.button("Say Hello! 🚀", use_container_width=True)

# ── Processing & Display ────────────────────────────────────
if submit:
    # Send input to the backend for computation
    result = process_username(name_input)

    if result["success"]:
        # Display the greeting on the dashboard
        st.markdown(
            f'<div class="greeting-box">{result["message"]}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="info-card">📛 Display Name: <strong>{result["display_name"]}</strong></div>',
            unsafe_allow_html=True,
        )
        st.balloons()
    else:
        st.error(result["error"])
