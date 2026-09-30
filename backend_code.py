"""
Backend Module — Ultron LLM Engine & Dynamic Personalization
============================================================
Powers conversational AI 'Ultron' with:
- Dynamic time-based Claude-style greetings
- Multi-LLM provider support (Groq & Google Gemini)
- Automatic key detection via parameters, st.secrets, and env variables
"""

import os
import random
from datetime import datetime
from typing import List, Dict, Generator

# Try importing Streamlit for secrets detection
try:
    import streamlit as st
except ImportError:
    st = None

# Try importing SDKs
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

try:
    from groq import Groq
except ImportError:
    Groq = None


ULTRON_DEFAULT_PROMPT = (
    "You are Ultron, a sophisticated, hyper-intelligent, and capable AI assistant. "
    "You provide clear, well-structured, insightful, and accurate answers. "
    "Maintain a sleek, confident, and engaging tone while remaining deeply helpful."
)


def get_time_based_salutation() -> str:
    """Returns 'Good morning', 'Good afternoon', 'Good evening', or 'Good night' based on current hour."""
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "Good morning"
    elif 12 <= hour < 17:
        return "Good afternoon"
    elif 17 <= hour < 22:
        return "Good evening"
    else:
        return "Good night"


def get_dynamic_greeting(user_name: str = "") -> str:
    """
    Generates a personalized Claude-style dynamic greeting based on time of day
    and a randomized thoughtful prompt.

    Example: 'Good afternoon, ritorob. What are we building today?'
    """
    salutation = get_time_based_salutation()
    clean_name = (user_name or "").strip().title()
    name_clause = f", {clean_name}" if clean_name else ""

    claude_prompts = [
        "What are we building today?",
        "What are we working on right now?",
        "How can I assist your workflow today?",
        "What challenge are we tackling today?",
        "Ready to explore some new ideas?",
        "What's on your mind today?",
        "Where shall we start?",
        "I'm at your command. What are we solving today?",
        "How can I help you innovate today?",
    ]

    selected_prompt = random.choice(claude_prompts)
    return f"{salutation}{name_clause}. {selected_prompt}"


def resolve_api_key(passed_key: str = None) -> str:
    """Resolves API key from parameter, Streamlit secrets, or environment variables."""
    if passed_key and passed_key.strip():
        return passed_key.strip()

    # Check Streamlit secrets
    if st is not None:
        try:
            if "LLM_API_KEY" in st.secrets:
                return str(st.secrets["LLM_API_KEY"]).strip()
            if "GROQ_API_KEY" in st.secrets:
                return str(st.secrets["GROQ_API_KEY"]).strip()
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

    # Check environment variables
    for env_var in ["LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    return ""


def generate_llm_response(
    messages: List[Dict[str, str]],
    api_key: str = None,
    system_prompt: str = ULTRON_DEFAULT_PROMPT
) -> Generator[str, None, None]:
    """
    Sends conversation history to the selected LLM and streams back Ultron's response.
    Auto-detects whether the key is Groq (gsk_...) or Gemini (AIza...).
    """
    key = resolve_api_key(api_key)

    if not key:
        yield (
            "### ⚡ Ultron Systems Standby\n\n"
            "To activate Ultron, please provide an API Key via **Settings & API Key** in the sidebar.\n\n"
            "* 🚀 **Groq Key (Free & Instant):** [console.groq.com/keys](https://console.groq.com/keys)\n"
            "* 🌟 **Gemini Key (Google AI Studio):** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n\n"
            "*Tip: You can also add `GROQ_API_KEY = \"your-key\"` to Streamlit Cloud Secrets!*"
        )
        return

    # Check if Groq key
    if key.startswith("gsk_"):
        if Groq is None:
            yield "❌ `groq` library is missing. Please run `pip install groq`."
            return
        try:
            client = Groq(api_key=key)
            formatted_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

            stream = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=formatted_messages,
                stream=True,
                temperature=0.7,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except Exception as e:
            yield f"❌ **Ultron Core Error (Groq)**: {str(e)}"
            return

    # Otherwise treat as Gemini key
    if genai is None:
        yield "❌ `google-genai` library is missing. Please run `pip install google-genai`."
        return

    try:
        client = genai.Client(api_key=key)
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=msg["content"])]
                )
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )

        response = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"❌ **Ultron Core Error (Gemini)**: {str(e)}"
