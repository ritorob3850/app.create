"""
Backend Module — LLM Core Engine
================================
Connects to Google Gemini LLM to power a full conversational AI assistant
similar to ChatGPT and Claude.
"""

import os
from typing import List, Dict, Generator
try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


def generate_llm_response(
    messages: List[Dict[str, str]],
    api_key: str = None,
    system_prompt: str = "You are a helpful, intelligent, and friendly AI assistant like ChatGPT or Claude."
) -> Generator[str, None, None]:
    """
    Sends chat history to Gemini LLM and streams back the assistant response.

    Args:
        messages: List of message dictionaries with 'role' ('user' or 'assistant') and 'content'.
        api_key: Google Gemini API key. If omitted, checks GEMINI_API_KEY environment variable.
        system_prompt: System instruction directing the AI's personality.

    Yields:
        Chunks of text response as they stream from the LLM.
    """
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

    if not effective_api_key:
        yield (
            "⚠️ **Gemini API Key Required**\n\n"
            "To activate the AI chatbot, please enter your free **Gemini API Key** in the sidebar on the left.\n\n"
            "👉 You can get a free key in 10 seconds at: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)"
        )
        return

    if genai is None:
        yield "❌ Error: `google-genai` library is not installed. Please run `pip install google-genai`."
        return

    try:
        client = genai.Client(api_key=effective_api_key)

        # Convert messages into contents format for Gemini
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
        yield f"❌ **LLM Error**: {str(e)}"
