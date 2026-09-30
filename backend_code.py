"""
Backend Module — Multi-LLM Engine (Gemini & Groq)
==================================================
Powers conversational AI like ChatGPT and Claude using either:
- Google Gemini (gemini-2.5-flash)
- Groq (llama-3.3-70b-versatile / llama-3.1-8b-instant)
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


def generate_llm_response(
    messages: List[Dict[str, str]],
    api_key: str = None,
    system_prompt: str = "You are a helpful, intelligent, and friendly AI assistant like ChatGPT or Claude."
) -> Generator[str, None, None]:
    """
    Sends chat history to the selected LLM and streams back the assistant response.
    Auto-detects whether the key is Gemini (AIza...) or Groq (gsk_...).
    """
    key = (api_key or os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY") or os.environ.get("GROQ_API_KEY") or "").strip()

    if not key:
        yield (
            "⚠️ **API Key Required**\n\n"
            "Please paste your free API key in the sidebar:\n\n"
            "- ⚡ **Groq Key (Instant & Free):** [console.groq.com/keys](https://console.groq.com/keys)\n"
            "- 🌟 **Gemini Key (Free):** [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) *(use personal Gmail)*"
        )
        return

    # Check if Groq key
    if key.startswith("gsk_"):
        if Groq is None:
            yield "❌ `groq` library not installed. Please run `pip install groq`."
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
            yield f"❌ **Groq LLM Error**: {str(e)}"
            return

    # Otherwise treat as Gemini key
    if genai is None:
        yield "❌ `google-genai` library not installed. Please run `pip install google-genai`."
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
        yield f"❌ **Gemini Error**: {str(e)}"
