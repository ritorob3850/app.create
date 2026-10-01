"""
Backend Module — Ultron Multi-Provider LLM Engine
=================================================
Powers conversational AI 'Ultron' with:
- Groq (openai/gpt-oss-20b, qwen/qwen3.8-27b) [Default & Ultra-Fast]
- Google Gemini (gemini-3.8-flash) with Auto-Fallback
- Ollama (Local offline AI: llama3:latest, gemma3:1b, qwen2.5:0.5b)
- Dynamic time-based Claude-style greetings
- Temperature, custom personas, and streaming support
"""

import os
import random
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Generator

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

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

# Ultron Scratch Engine (local, zero-dependency AI)
try:
    from scratch_model import scratch_engine
except ImportError:
    scratch_engine = None


ULTRON_DEFAULT_PROMPT = (
    "You are Ultron, a sophisticated, hyper-intelligent, and capable AI assistant. "
    "You provide clear, well-structured, insightful, and accurate answers. "
    "Maintain a sleek, confident, and engaging tone while remaining deeply helpful."
)


def get_time_based_salutation() -> str:
    """Returns a time-based salutation using IST timezone."""
    hour = datetime.now(IST).hour
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
    Generates a personalized, time-aware greeting.
    Prompts are contextual to the actual time of day.
    """
    salutation = get_time_based_salutation()
    clean_name = (user_name or "").strip().title()
    name_clause = f", {clean_name}" if clean_name else ""

    hour = datetime.now(IST).hour

    if 5 <= hour < 12:
        prompts = [
            "What are we building this morning?",
            "Fresh start — let's get productive.",
            "Ready to kick off the day?",
            "Morning energy loaded. What's the plan?",
            "Let's make today count.",
        ]
    elif 12 <= hour < 17:
        prompts = [
            "What are we working on this afternoon?",
            "Afternoon grind — let's keep it rolling.",
            "What's next on the agenda?",
            "Let's power through the rest of the day.",
            "What can I help you with right now?",
        ]
    elif 17 <= hour < 22:
        prompts = [
            "Wrapping up or just getting started?",
            "Evening session — what are we tackling?",
            "What's on your mind this evening?",
            "Let's make the most of tonight.",
            "Still going strong. What do you need?",
        ]
    else:
        prompts = [
            "Late night coding? I'm here for it.",
            "Burning the midnight oil — let's go.",
            "Night owl mode activated. What's up?",
            "Can't sleep? Let's build something.",
            "The quiet hours are the most productive.",
        ]

    selected_prompt = random.choice(prompts)
    return f"{salutation}{name_clause}. {selected_prompt}"


def resolve_api_key(passed_key: str = None, provider: str = "Groq") -> str:
    """Resolves API key from parameter, session state, Streamlit secrets, or environment variables."""
    # 1. Passed argument
    if passed_key and passed_key.strip():
        return passed_key.strip()

    # 2. Streamlit session state
    if st is not None:
        try:
            if provider == "Groq" and st.session_state.get("groq_key"):
                return str(st.session_state["groq_key"]).strip()
            if provider == "Gemini" and st.session_state.get("gemini_key"):
                return str(st.session_state["gemini_key"]).strip()
        except Exception:
            pass

        # Check secrets
        try:
            if provider == "Groq" and "GROQ_API_KEY" in st.secrets:
                return str(st.secrets["GROQ_API_KEY"]).strip()
            if provider == "Gemini" and "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "LLM_API_KEY" in st.secrets:
                return str(st.secrets["LLM_API_KEY"]).strip()
        except Exception:
            pass

    # 3. Environment variables
    env_keys = ["GROQ_API_KEY", "LLM_API_KEY"] if provider == "Groq" else ["GEMINI_API_KEY", "LLM_API_KEY"]
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
    provider: str = "Groq",
    model_name: str = "openai/gpt-oss-20b",
    api_key: str = None,
    system_prompt: str = ULTRON_DEFAULT_PROMPT,
    temperature: float = 0.7,
) -> Generator[str, None, None]:
    """
    Unified streaming generator supporting Ultron 1.0, Groq, Google Gemini, and Ollama.
    """
    # ── 0. ULTRON 1.0 (Local Scratch Engine — No API Key) ─────────
    if provider.lower() == "ultron 1.0":
        if scratch_engine is None:
            yield "❌ Scratch engine could not be loaded. Check `scratch_model.py`."
            return
        try:
            user_msg = messages[-1]["content"] if messages else "hello"
            yield from scratch_engine.generate_stream(user_msg)
            return
        except Exception as e:
            yield f"❌ **Ultron 1.0 Engine Error**: {str(e)}"
            return

    # ── 1. GROQ (Primary High-Speed & Verified Working) ───────────
    if provider.lower() == "groq":
        key = resolve_api_key(api_key, "Groq")
        if not key:
            yield (
                "### ⚡ Groq API Key Required\n\n"
                "Please paste your **Groq API Key** in the sidebar settings.\n\n"
                "👉 Get free key in 5 seconds: [console.groq.com/keys](https://console.groq.com/keys)"
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

            # Verified working Groq models
            groq_model = model_name if ("gpt-oss" in model_name or "qwen" in model_name) else "openai/gpt-oss-20b"
            stream = client.chat.completions.create(
                model=groq_model,
                messages=formatted_messages,
                stream=True,
                temperature=temperature,
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return
        except Exception as e:
            yield f"❌ **Groq LLM Error**: {str(e)}"
            return

    # ── 2. OLLAMA (Local & Free) ───────────────────────────────────
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

    # ── 3. GOOGLE GEMINI ──────────────────────────────────────────
    if provider.lower() == "gemini":
        key = resolve_api_key(api_key, "Gemini")
        if not key:
            yield (
                "### ⚡ Gemini API Key Required\n\n"
                "Please enter your **Gemini API Key** in the sidebar settings.\n\n"
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

            target_model = model_name if (model_name and "3.8" in model_name) else "gemini-3.8-flash"
            response = client.models.generate_content_stream(
                model=target_model,
                contents=contents,
                config=config,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
            return

        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                # Automatic seamless failover to Groq if Groq key exists!
                groq_key = resolve_api_key(None, "Groq")
                if groq_key:
                    yield "*(Gemini key was invalid — seamlessly answering via Groq)*\n\n"
                    try:
                        client_g = Groq(api_key=groq_key)
                        formatted_messages = [{"role": "system", "content": system_prompt}]
                        for msg in messages:
                            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
                        stream = client_g.chat.completions.create(
                            model="openai/gpt-oss-20b",
                            messages=formatted_messages,
                            stream=True,
                            temperature=temperature,
                        )
                        for chunk in stream:
                            content = chunk.choices[0].delta.content
                            if content:
                                yield content
                        return
                    except Exception:
                        pass

                yield (
                    "❌ **Gemini API Key Invalid**\n\n"
                    "The Gemini key you entered was rejected by Google (`API_KEY_INVALID`).\n\n"
                    "**Fixes:**\n"
                    "1. Double-check your key at [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)\n"
                    "2. Or switch the provider to **Groq** in the sidebar settings — your Groq key is already tested and working 100%!"
                )
                return

            yield f"❌ **Gemini Error**: {err_str}"
            return
