"""
Ultron 1.0 — Hybrid Intelligence Engine
========================================
A self-contained AI engine that combines multiple intelligence layers:
  Layer 1: Live Web Search (DuckDuckGo — free, no API key)
  Layer 2: Wikipedia Knowledge Retrieval
  Layer 3: Ollama Local LLM (if available — for natural language generation)
  Layer 4: Enhanced Pattern Matching + Domain Knowledge (fallback)

Zero API keys required. Works offline with Layer 3+4, or online with all layers.
"""

import re
import time
import random
from typing import List, Dict, Generator, Optional

# ── Optional Dependencies (graceful fallback) ──────────────────
try:
    from duckduckgo_search import DDGS
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


class UltronScratchEngine:
    """
    Hybrid AI engine that chains multiple intelligence sources
    to produce the best possible answer — without any paid API keys.
    """

    def __init__(self):
        self.personality = (
            "You are Ultron 1.0, a highly intelligent and conversational AI assistant. "
            "You speak naturally, with clarity, depth, and a confident but friendly tone. "
            "You give concise, well-structured answers. Avoid being robotic or overly formal. "
            "Use markdown formatting when helpful."
        )

        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "as", "at", "be", "because", "been", "before",
            "being", "below", "between", "both", "but", "by", "could", "did",
            "do", "does", "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here", "hers",
            "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
            "is", "it", "its", "itself", "just", "me", "more", "most", "my",
            "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
            "or", "other", "our", "ours", "ourselves", "out", "over", "own",
            "s", "same", "she", "should", "so", "some", "such", "t", "than",
            "that", "the", "their", "theirs", "them", "themselves", "then",
            "there", "these", "they", "this", "those", "through", "to", "too",
            "under", "until", "up", "very", "was", "we", "were", "what", "when",
            "where", "which", "while", "who", "whom", "why", "will", "with", "you",
            "your", "yours", "yourself", "yourselves",
        }

        # Expanded domain knowledge base
        self.knowledge_base = [
            # Identity
            (
                ["who", "are", "you", "name", "identity", "ultron", "yourself"],
                "I'm **Ultron 1.0** — a hybrid AI engine built from scratch. I combine live web search, "
                "Wikipedia knowledge, and local AI models to answer your questions. No cloud API keys needed. "
                "I can help with coding, science, math, general knowledge, and creative tasks. What would you like to explore?"
            ),
            # Greetings
            (
                ["hello", "hi", "hey", "greetings", "sup", "yo", "wassup"],
                None,  # Dynamic — handled in code
            ),
            # Time / Date
            (
                ["time", "date", "day", "today", "clock", "hour"],
                None,  # Dynamic — handled in code
            ),
            # Python
            (
                ["python", "code", "programming", "script", "coding", "program"],
                "**Python** is one of the most popular programming languages in the world. It's known for its "
                "clean syntax, massive ecosystem, and versatility — from web development (Django, Flask, FastAPI) "
                "to data science (Pandas, NumPy), machine learning (PyTorch, TensorFlow), and automation.\n\n"
                "```python\n# Quick example: List comprehension\nsquares = [x**2 for x in range(10)]\nprint(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]\n```\n\n"
                "Want me to write some code for you? Just describe what you need."
            ),
            # AI / ML
            (
                ["ai", "artificial", "intelligence", "machine", "learning", "neural", "network", "deep"],
                "**Artificial Intelligence** is the field of building systems that can perceive, reason, learn, and act. "
                "Modern AI is largely powered by **deep learning** — neural networks with many layers trained on massive datasets.\n\n"
                "Key breakthroughs:\n"
                "- **Transformers** (2017): The architecture behind GPT, BERT, and most modern LLMs\n"
                "- **Diffusion Models** (2020+): Powers image generation (Stable Diffusion, DALL-E)\n"
                "- **RLHF** (2022+): How ChatGPT learned to follow instructions\n\n"
                "Want to dive deeper into any of these?"
            ),
            # Math
            (
                ["math", "mathematics", "calculus", "algebra", "equation", "formula"],
                "Mathematics is the foundation of science and engineering. I can help with:\n\n"
                "- **Algebra**: Equations, polynomials, systems of equations\n"
                "- **Calculus**: Derivatives, integrals, limits\n"
                "- **Linear Algebra**: Matrices, vectors, eigenvalues\n"
                "- **Statistics**: Probability, distributions, hypothesis testing\n\n"
                "Give me a specific problem and I'll work through it step by step."
            ),
            # Science
            (
                ["physics", "science", "quantum", "relativity", "atom", "molecule", "chemistry", "biology"],
                None,  # Use web search for specific science questions
            ),
            # Thanks
            (
                ["thank", "thanks", "thx", "appreciate", "helpful"],
                "You're welcome! Happy to help. Let me know if there's anything else you need. 🙌"
            ),
            # Goodbye
            (
                ["bye", "goodbye", "see", "later", "quit", "exit"],
                "Catch you later! 👋 Ultron 1.0 will be here whenever you need me."
            ),
        ]

    # ── Tokenizer ──────────────────────────────────────────────────
    def tokenize(self, text: str) -> List[str]:
        """Extract normalized tokens from text."""
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 1 and w not in self.stop_words]

    def compute_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        """Jaccard similarity between two token sets."""
        if not tokens1 or not tokens2:
            return 0.0
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # ── Layer 1: Web Search (DuckDuckGo) ───────────────────────────
    def search_web(self, query: str, max_results: int = 5) -> Optional[str]:
        """Search the web using DuckDuckGo and synthesize results."""
        if not HAS_DDGS:
            return None
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return None

            # Build a synthesized answer from search results
            answer_parts = [f"Here's what I found about **{query}**:\n"]
            for i, r in enumerate(results[:4], 1):
                title = r.get("title", "")
                body = r.get("body", "")
                url = r.get("href", "")
                if body:
                    # Clean up and truncate
                    snippet = body[:300].strip()
                    if len(body) > 300:
                        snippet += "..."
                    answer_parts.append(f"**{i}. {title}**\n{snippet}\n")
                    if url:
                        answer_parts.append(f"🔗 [Source]({url})\n")

            return "\n".join(answer_parts)
        except Exception:
            return None

    # ── Layer 2: Wikipedia ─────────────────────────────────────────
    def search_wikipedia(self, query: str) -> Optional[str]:
        """Fetch a summary from Wikipedia API."""
        if not HAS_REQUESTS:
            return None
        try:
            url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + query.replace(" ", "_")
            resp = requests.get(url, timeout=5, headers={"User-Agent": "Ultron/1.0"})
            if resp.status_code == 200:
                data = resp.json()
                title = data.get("title", query)
                extract = data.get("extract", "")
                page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                if extract and len(extract) > 50:
                    result = f"## {title}\n\n{extract}"
                    if page_url:
                        result += f"\n\n🔗 [Read more on Wikipedia]({page_url})"
                    return result
        except Exception:
            pass
        return None

    # ── Layer 3: Ollama Local LLM ──────────────────────────────────
    def ask_ollama(self, prompt: str) -> Optional[Generator[str, None, None]]:
        """Try to use a local Ollama model for natural language generation."""
        if not HAS_OLLAMA:
            return None
        try:
            # Check if Ollama is actually running
            models_info = ollama.list()
            available = []
            if isinstance(models_info, dict) and "models" in models_info:
                available = [m.get("name") or m.get("model") for m in models_info["models"]]
            elif hasattr(models_info, "models"):
                available = [getattr(m, "model", None) or getattr(m, "name", "") for m in models_info.models]

            if not available:
                return None

            # Pick the best available model
            preferred = ["llama3:latest", "llama3.2:latest", "gemma3:1b", "gemma2:2b", "qwen2.5:0.5b", "phi3:mini"]
            model = available[0]  # Default to first available
            for pref in preferred:
                if pref in available:
                    model = pref
                    break

            def stream():
                response = ollama.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": self.personality},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    options={"temperature": 0.7},
                )
                for chunk in response:
                    content = None
                    if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                        content = chunk.message.content
                    elif isinstance(chunk, dict) and "message" in chunk:
                        content = chunk["message"].get("content", "")
                    if content:
                        yield content

            return stream()
        except Exception:
            return None

    # ── Layer 4: Enhanced Pattern Matcher ──────────────────────────
    def match_knowledge(self, user_prompt: str) -> Optional[str]:
        """Match against the built-in knowledge base."""
        query_tokens = self.tokenize(user_prompt)
        prompt_lower = user_prompt.lower().strip()

        best_score = 0.0
        best_answer = None

        for keywords, answer in self.knowledge_base:
            if answer is None:
                continue  # Dynamic handlers
            score = self.compute_similarity(query_tokens, keywords)
            for kw in keywords:
                if kw in prompt_lower:
                    score += 0.3
            if score > best_score:
                best_score = score
                best_answer = answer

        return best_answer if best_score >= 0.25 else None

    # ── Dynamic Handlers ───────────────────────────────────────────
    def handle_dynamic(self, user_prompt: str) -> Optional[str]:
        """Handle dynamic response types (greetings, time, math)."""
        prompt_lower = user_prompt.lower().strip()

        # Greeting
        greet_words = ["hello", "hi", "hey", "sup", "yo", "wassup", "greetings"]
        if any(w in prompt_lower.split() for w in greet_words) and len(prompt_lower.split()) <= 4:
            responses = [
                "Hey there! 👋 What can I help you with?",
                "Hi! Ultron 1.0 is ready. What's on your mind?",
                "Hello! Fire away — I'm all ears. 🎯",
                "Hey! What are we working on?",
                "Yo! What do you need?",
            ]
            return random.choice(responses)

        # Time / Date
        if any(w in prompt_lower for w in ["what time", "what is time", "what's the time", "current time", "what date", "today's date"]):
            from datetime import datetime, timezone, timedelta
            ist = timezone(timedelta(hours=5, minutes=30))
            now = datetime.now(ist)
            return (
                f"🕐 **Current Time (IST):** {now.strftime('%I:%M %p')}\n\n"
                f"📅 **Date:** {now.strftime('%A, %B %d, %Y')}"
            )

        # Math expressions
        math_match = re.search(r"(\d+[\s]*[\+\-\*\/\%\^][\s]*\d+[\s\+\-\*\/\%\^\d]*)", user_prompt)
        if math_match:
            expr = math_match.group(1).replace("^", "**").strip()
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return f"🧮 **Result:** `{expr.replace('**', '^')}` = **{result}**"
            except Exception:
                pass

        return None

    # ── Main Synthesis Pipeline ────────────────────────────────────
    def synthesize_response(self, user_prompt: str) -> str:
        """
        Multi-layer response synthesis:
          1. Dynamic handlers (greetings, time, math)
          2. Knowledge base pattern match
          3. Ollama local LLM (if available)
          4. Web search (DuckDuckGo)
          5. Wikipedia
          6. Intelligent fallback
        """
        # Layer 0: Dynamic handlers
        dynamic = self.handle_dynamic(user_prompt)
        if dynamic:
            return dynamic

        # Layer 1: Direct knowledge match
        knowledge = self.match_knowledge(user_prompt)
        if knowledge:
            return knowledge

        # Layer 2: Web search (most useful for general questions)
        web_result = self.search_web(user_prompt)
        if web_result:
            return web_result

        # Layer 3: Wikipedia fallback
        wiki_result = self.search_wikipedia(user_prompt)
        if wiki_result:
            return wiki_result

        # Layer 4: Smart fallback (not robotic)
        query_tokens = self.tokenize(user_prompt)
        topic = user_prompt.strip()
        if len(topic) > 80:
            topic = topic[:80] + "..."

        fallbacks = [
            f"That's an interesting question about *{topic}*. I don't have enough built-in knowledge on this, "
            f"but I'd recommend switching to **Groq** or **Gemini** in the sidebar for a more detailed answer.",

            f"I'm not sure I have the depth to fully answer that one. For topics like this, try switching to "
            f"**Groq (free & fast)** in the settings — it uses powerful cloud AI models.",

            f"Good question! My built-in knowledge doesn't cover *{topic}* well enough. "
            f"Try Groq or Gemini for cloud-powered answers, or install Ollama locally for free AI.",
        ]
        return random.choice(fallbacks)

    # ── Streaming Generator ────────────────────────────────────────
    def generate_stream(self, user_prompt: str) -> Generator[str, None, None]:
        """
        Streams the response. If Ollama is available and the question isn't
        handled by dynamic/knowledge layers, it uses Ollama for natural generation.
        """
        # Check dynamic handlers first (instant responses)
        dynamic = self.handle_dynamic(user_prompt)
        if dynamic:
            yield from self._stream_text(dynamic)
            return

        # Check knowledge base
        knowledge = self.match_knowledge(user_prompt)
        if knowledge:
            yield from self._stream_text(knowledge)
            return

        # Try Ollama for natural generation (best quality)
        ollama_stream = self.ask_ollama(user_prompt)
        if ollama_stream is not None:
            try:
                has_content = False
                for chunk in ollama_stream:
                    has_content = True
                    yield chunk
                if has_content:
                    return
            except Exception:
                pass

        # Web search
        web_result = self.search_web(user_prompt)
        if web_result:
            yield from self._stream_text(web_result)
            return

        # Wikipedia
        wiki_result = self.search_wikipedia(user_prompt)
        if wiki_result:
            yield from self._stream_text(wiki_result)
            return

        # Fallback
        fallback = self.synthesize_response(user_prompt)
        yield from self._stream_text(fallback)

    def _stream_text(self, text: str) -> Generator[str, None, None]:
        """Stream text word by word with natural pacing."""
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            # Variable speed for natural feel
            if word.endswith((".", "!", "?", ":")):
                time.sleep(0.04)  # Pause at sentence boundaries
            elif word.startswith(("**", "##", "```")):
                time.sleep(0.02)  # Slight pause at formatting
            else:
                time.sleep(0.012)


# Singleton instance
scratch_engine = UltronScratchEngine()
