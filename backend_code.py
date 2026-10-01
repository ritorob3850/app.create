"""
Backend Module — Ultron Multi-Provider LLM Engine
=================================================
Powers conversational AI 'Ultron' with:
- Google Gemini (gemini-3.8-flash) with Auto-Fallback
- Groq (openai/gpt-oss-20b, qwen/qwen3.8-27b, openai/gpt-oss-120b)
- Ollama (Local offline AI: llama3:latest, gemma3:1b, qwen2.5:0.5b)
- Dynamic time-based Claude-style greetings
- Temperature, custom personas, and streaming support
"""

import os
import random
from datetime import datetime
from typing import List, Dict, Generator

# Streamlit secrets check
try:
    import streamlit as st
except ImportError:
    st = None

# Google GenAI SDK
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

# Ollama SDK (local)
try:
    import ollama
except ImportError:
    ollama = None

# Groq SDK
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
        "Good to see you!",
    ]

    selected_prompt = random.choice(claude_prompts)
    return f"{salutation}{name_clause}. {selected_prompt}"


def resolve_api_key(passed_key: str = None, provider: str = "Gemini") -> str:
    """Resolves API key from parameter, session state, Streamlit secrets, or environment variables."""
    # 1. Passed argument
    if passed_key and passed_key.strip():
        return passed_key.strip()

    # 2. Streamlit session state & secrets
    if st is not None:
        try:
            if provider == "Gemini" and st.session_state.get("gemini_key"):
                return str(st.session_state["gemini_key"]).strip()
            if provider == "Groq" and st.session_state.get("groq_key"):
                return str(st.session_state["groq_key"]).strip()
            if st.session_state.get("api_key"):
                return str(st.session_state["api_key"]).strip()
        except Exception:
            pass

        try:
            if provider == "Gemini":
                if "GEMINI_API_KEY" in st.secrets:
                    return str(st.secrets["GEMINI_API_KEY"]).strip()
            elif provider == "Groq":
                if "GROQ_API_KEY" in st.secrets:
                    return str(st.secrets["GROQ_API_KEY"]).strip()
            if "LLM_API_KEY" in st.secrets:
                return str(st.secrets["LLM_API_KEY"]).strip()
        except Exception:
            pass

    # 3. Environment variables
    env_keys = ["GEMINI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"] if provider == "Gemini" else ["GROQ_API_KEY", "LLM_API_KEY"]
    for env_var in env_keys:
        val = os.environ.get(env_var, "").strip()
        if val:
            return val

    return ""


def get_available_ollama_models() -> List[str]:
    """Fetches list of installed local Ollama models."""
    if ollama is None:
        return ["llama3:latest", "gemma3:1b", "qwen2.5:0.5b"]
    try:
        models_info = ollama.list()
        if isinstance(models_info, dict) and "models" in models_info:
            return [m.get("name") or m.get("model") for m in models_info["models"]]
        elif hasattr(models_info, "models"):
            return [getattr(m, "model", None) or getattr(m, "name", "llama3:latest") for m in models_info.models]
    except Exception:
        pass
    return ["llama3:latest", "gemma3:1b", "qwen2.5:0.5b"]


def generate_llm_response(
    messages: List[Dict[str, str]],
    provider: str = "Gemini",
    model_name: str = "gemini-3.8-flash",
    api_key: str = None,
    system_prompt: str = ULTRON_DEFAULT_PROMPT,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Unified streaming generator supporting Google Gemini, Ollama (Local), and Groq.
    """
    # ── 1. OLLAMA (Local & Free) ───────────────────────────────────
    if provider.lower() == "ollama":
        if ollama is None:
            yield "❌ `ollama` Python library is missing. Run `pip install ollama`."
            return
        try:
            formatted_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

            response_stream = ollama.chat(
                model=model_name or "llama3:latest",
                messages=formatted_messages,
                stream=True,
                options={"temperature": temperature},
            )
            for chunk in response_stream:
                content = None
                if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                    content = chunk.message.content
                elif isinstance(chunk, dict) and "message" in chunk:
                    content = chunk["message"].get("content", "")
                
                if content:
                    yield content
            return
        except Exception as e:
            yield (
                f"❌ **Ollama Error**: {str(e)}\n\n"
                "💡 *Make sure you are running the app locally on your PC (Ollama cannot be reached from the public Streamlit Cloud link).*"
            )
            return

    # ── 2. GOOGLE GEMINI ──────────────────────────────────────────
    if provider.lower() == "gemini":
        key = resolve_api_key(api_key, "Gemini")
        if not key:
            yield (
                "### ⚡ Gemini API Key Required\n\n"
                "Please paste your **Gemini API Key** into the sidebar settings.\n\n"
                "👉 **Get a free key:** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)"
            )
            return

        if genai is None:
            yield "❌ `google-genai` library is missing. Run `pip install google-genai`."
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
                temperature=temperature,
            )

            # Gemini-3.8-flash is the mandatory model for Google GenAI v1beta
            target_model = model_name if (model_name and "3.8" in model_name) else "gemini-3.8-flash"
            candidates = [target_model, "gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash"]
            
            seen = set()
            models_to_try = [m for m in candidates if m and not (m in seen or seen.add(m))]

            last_error = None
            for candidate in models_to_try:
                try:
                    response = client.models.generate_content_stream(
                        model=candidate,
                        contents=contents,
                        config=config,
                    )
                    has_output = False
                    for chunk in response:
                        if chunk.text:
                            has_output = True
                            yield chunk.text
                    if has_output:
                        return
                except Exception as stream_err:
                    last_error = str(stream_err)
                    continue

            if last_error:
                yield f"⚠️ **Notice**: {last_error}"

        except Exception as e:
            yield f"❌ **Gemini Error**: {str(e)}"
            return

    # ── 3. GROQ (Ultra-Fast & Stable) ───────────────────────────────
    if provider.lower() == "groq":
        key = resolve_api_key(api_key, "Groq")
        if not key:
            yield (
                "### ⚡ Groq API Key Required\n\n"
                "Please paste your **Groq API Key** in the sidebar settings."
            )
            return

        if Groq is None:
            yield "❌ `groq` library is missing. Run `pip install groq`."
            return

        try:
            client = Groq(api_key=key)
            formatted_messages = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                formatted_messages.append({"role": msg["role"], "content": msg["content"]})

            # Verified active Groq models
            groq_candidates = [model_name, "openai/gpt-oss-20b", "qwen/qwen3.8-27b", "openai/gpt-oss-120b"]
            seen_g = set()
            models_to_try_g = [m for m in groq_candidates if m and not (m in seen_g or seen_g.add(m))]

            for candidate_model in models_to_try_g:
                try:
                    stream = client.chat.completions.create(
                        model=candidate_model,
                        messages=formatted_messages,
                        stream=True,
                        temperature=temperature,
                    )
                    for chunk in stream:
                        content = chunk.choices[0].delta.content
                        if content:
                            yield content
                    return
                except Exception as g_err:
                    continue

        except Exception as e:
            yield f"❌ **Groq LLM Error**: {str(e)}"
            return
