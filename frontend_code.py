
"""
Frontend Module — Streamlit Dashboard
======================================
Collects the user's name, hands it off to the backend
for processing, and renders the result on a simple dashboard.
"""

import streamlit as st
from backend_code import generate_greeting

# ── Page Setup ───────────────────────────────────────────────
st.set_page_config(page_title="Greeting Dashboard", page_icon="👋", layout="centered")

# ── Header ───────────────────────────────────────────────────
st.title("👋 Greeting Dashboard")
st.subheader("Your personalised welcome awaits!")

st.write(
    "Type your name below and we'll craft a greeting just for you — "
    "powered by a dedicated backend module."
)

st.divider()

# ── User Input ───────────────────────────────────────────────
name = st.text_input("✏️ What's your name?", placeholder="e.g. Riya, Aman, Jordan ...")

# ── Process & Display ────────────────────────────────────────
greeting = generate_greeting(name)

if name.strip():
    st.success(greeting)
    st.balloons()
else:
    st.info(greeting)
