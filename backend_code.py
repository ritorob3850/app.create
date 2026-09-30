"""
Backend Module — Ultron LLM Engine
===================================
Powers the conversational AI assistant 'Ultron' using Google Gemini and Groq.
"""

import os
from typing import List, Dict, Generator

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


def generate_llm_response(
    messages: List[Dict[str, str]],
    api_key: str = None,
    system_prompt: str = ULTRON_DEFAULT_PROMPT
) -> Generator[str, None, None]:
    """
    Sends conversation history to the selected LLM and streams back Ultron's response.
    Supports auto-detection between Groq (gsk_...) and Gemini (AIza...).
    """
    key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("GROQ_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or ""
    ).strip()

    if not key:
        yield (
            "### ⚡ Ultron Systems Standby\n\n"
            "To activate Ultron, please enter your free **API Key** in the Settings panel at the bottom of the sidebar.\n\n"
            "* 🚀 **Groq Key (Instant & Free):** [console.groq.com/keys](https://console.groq.com/keys)\n"
            "* 🌟 **Gemini Key (Google AI Studio):** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)"
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
