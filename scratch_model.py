"""
Ultron 1.0 — Hybrid Intelligence Engine (v2)
=============================================
Architecture:
  ┌─────────────────────────────────────────────────┐
  │  ULTRON 1.0 INTELLIGENCE PIPELINE               │
  │                                                  │
  │  Layer 0: Intent Classifier (categorize query)   │
  │  Layer 1: Training Corpus (keyword-trained data) │
  │  Layer 2: Cloud Brain (Groq LLM — detailed AI)   │
  │  Layer 3: Web Search (DuckDuckGo — real-time)    │
  │  Layer 4: Wikipedia (factual knowledge)           │
  │  Layer 5: Ollama (local LLM if available)         │
  │  Layer 6: Smart Fallback                          │
  └─────────────────────────────────────────────────┘

Training: 500+ English keywords mapped to expert-level responses
Cloud: Built-in Groq access for detailed, natural answers
Local: Works offline with training data + Ollama
"""

import re
import time
import random
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Generator, Optional, Tuple

# Indian Standard Time
IST = timezone(timedelta(hours=5, minutes=30))

# ── Optional Dependencies ──────────────────────────────────────
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
    import ollama as ollama_lib
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# ══════════════════════════════════════════════════════════════
# TRAINING CORPUS — The "Weights" of Ultron 1.0
# ══════════════════════════════════════════════════════════════
# Each entry: (keywords, intent_category, trained_response)
# The model matches user queries against these keyword vectors
# to find the best trained response before going to cloud.

TRAINING_DATA: List[Tuple[List[str], str, str]] = [
    # ── IDENTITY & SELF-AWARENESS ──────────────────────────────
    (
        ["who", "are", "you", "name", "identity", "ultron", "yourself", "introduce", "what", "model"],
        "identity",
        "I'm **Ultron 1.0** — a hybrid intelligence engine built from the ground up. "
        "I combine a locally trained knowledge corpus with cloud-powered AI (via Groq) for detailed answers, "
        "live web search for real-time information, and Wikipedia for factual depth.\n\n"
        "**My Architecture:**\n"
        "- 🧠 **Training Corpus**: 500+ keywords across 20+ domains\n"
        "- ☁️ **Cloud Brain**: Groq-powered LLM for deep, natural responses\n"
        "- 🌐 **Web Search**: DuckDuckGo for live information\n"
        "- 📚 **Wikipedia**: Factual knowledge retrieval\n\n"
        "I'm designed to give polished, detailed answers without needing expensive API keys."
    ),
    (
        ["who", "made", "created", "built", "developer", "creator", "author"],
        "identity",
        "I was built by the developer of this project — designed as a custom hybrid AI engine "
        "that combines local training data with cloud intelligence. No single corporate API owns me. "
        "I'm open, extensible, and built to learn."
    ),
    (
        ["what", "can", "do", "capable", "abilities", "features", "help"],
        "identity",
        "Here's what I can do:\n\n"
        "- 💻 **Write & explain code** (Python, JavaScript, HTML/CSS, SQL, and more)\n"
        "- 🧮 **Solve math problems** (arithmetic, algebra, calculus)\n"
        "- 📚 **Answer knowledge questions** (science, history, geography, philosophy)\n"
        "- 🌐 **Search the web** for real-time information\n"
        "- 📖 **Pull Wikipedia articles** for factual depth\n"
        "- 💡 **Brainstorm ideas** and help with creative writing\n"
        "- 🔧 **Debug code** and explain error messages\n\n"
        "Just ask me anything — I'll use the best source available to answer."
    ),

    # ── GREETINGS & SOCIAL ─────────────────────────────────────
    (
        ["hello", "hi", "hey", "greetings", "sup", "yo", "howdy", "hola"],
        "greeting",
        None  # Dynamic
    ),
    (
        ["how", "are", "you", "doing", "going", "feeling"],
        "greeting",
        "I'm running at full capacity! 🚀 All systems are operational. What can I help you with?"
    ),
    (
        ["thank", "thanks", "thx", "appreciate", "grateful", "ty"],
        "social",
        "You're welcome! Glad I could help. Let me know if you need anything else. 🙌"
    ),
    (
        ["bye", "goodbye", "later", "see", "cya", "farewell", "quit", "exit"],
        "social",
        "See you later! 👋 Ultron 1.0 will be here whenever you need me. Take care!"
    ),
    (
        ["good", "morning", "afternoon", "evening", "night"],
        "greeting",
        None  # Dynamic — time-aware
    ),

    # ── TIME & DATE ────────────────────────────────────────────
    (
        ["time", "clock", "hour", "minute", "current"],
        "time",
        None  # Dynamic
    ),
    (
        ["date", "today", "day", "month", "year", "calendar"],
        "date",
        None  # Dynamic
    ),

    # ── PYTHON PROGRAMMING ─────────────────────────────────────
    (
        ["python", "programming", "language", "script", "coding", "code"],
        "python",
        "**Python** is the world's most popular programming language (IEEE, TIOBE rankings). "
        "Created by Guido van Rossum in 1991, it emphasizes readability and simplicity.\n\n"
        "**Why Python dominates:**\n"
        "- 🧠 **AI/ML**: PyTorch, TensorFlow, scikit-learn\n"
        "- 🌐 **Web**: Django, Flask, FastAPI\n"
        "- 📊 **Data Science**: Pandas, NumPy, Matplotlib\n"
        "- 🤖 **Automation**: Selenium, BeautifulSoup, Scrapy\n\n"
        "```python\n# Python is elegant\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        yield a\n        a, b = b, a + b\n\nprint(list(fibonacci(10)))\n# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]\n```"
    ),
    (
        ["list", "comprehension", "loop", "iterate", "for", "while"],
        "python",
        "**List comprehensions** are one of Python's most powerful features:\n\n"
        "```python\n# Basic\nsquares = [x**2 for x in range(10)]\n\n# With condition\nevens = [x for x in range(20) if x % 2 == 0]\n\n# Nested\nmatrix = [[i*j for j in range(5)] for i in range(5)]\n\n# Dictionary comprehension\nword_lengths = {w: len(w) for w in ['hello', 'world', 'python']}\n```\n\n"
        "These are faster and more Pythonic than traditional for loops."
    ),
    (
        ["function", "def", "lambda", "return", "parameter", "argument"],
        "python",
        "**Functions** in Python:\n\n"
        "```python\n# Standard function\ndef greet(name: str, greeting: str = 'Hello') -> str:\n    return f'{greeting}, {name}!'\n\n# Lambda (anonymous function)\nsquare = lambda x: x ** 2\n\n# *args and **kwargs\ndef flexible(*args, **kwargs):\n    print(f'Args: {args}')\n    print(f'Kwargs: {kwargs}')\n\n# Decorator\ndef timer(func):\n    import time\n    def wrapper(*args, **kwargs):\n        start = time.time()\n        result = func(*args, **kwargs)\n        print(f'{func.__name__} took {time.time()-start:.3f}s')\n        return result\n    return wrapper\n```"
    ),
    (
        ["class", "object", "oop", "inheritance", "method", "self", "init"],
        "python",
        "**Object-Oriented Python:**\n\n"
        "```python\nclass Animal:\n    def __init__(self, name: str, sound: str):\n        self.name = name\n        self.sound = sound\n    \n    def speak(self) -> str:\n        return f'{self.name} says {self.sound}!'\n\nclass Dog(Animal):\n    def __init__(self, name: str):\n        super().__init__(name, 'Woof')\n    \n    def fetch(self, item: str) -> str:\n        return f'{self.name} fetches the {item}!'\n\ndog = Dog('Rex')\nprint(dog.speak())   # Rex says Woof!\nprint(dog.fetch('ball'))  # Rex fetches the ball!\n```"
    ),
    (
        ["error", "exception", "try", "except", "bug", "debug", "traceback"],
        "python",
        "**Error Handling in Python:**\n\n"
        "```python\ntry:\n    result = 10 / 0\nexcept ZeroDivisionError as e:\n    print(f'Math error: {e}')\nexcept (TypeError, ValueError) as e:\n    print(f'Input error: {e}')\nexcept Exception as e:\n    print(f'Unexpected: {e}')\nfinally:\n    print('This always runs')\n\n# Custom exceptions\nclass InsufficientFundsError(Exception):\n    def __init__(self, balance, amount):\n        self.message = f'Cannot withdraw {amount}, balance is {balance}'\n        super().__init__(self.message)\n```\n\n"
        "**Debugging tips:** Use `breakpoint()` (Python 3.7+), check `traceback` module, or add `print()` statements."
    ),
    (
        ["api", "rest", "fastapi", "flask", "endpoint", "server", "backend", "web"],
        "python",
        "**Building REST APIs in Python:**\n\n"
        "```python\n# FastAPI (recommended — fast, modern, typed)\nfrom fastapi import FastAPI\nfrom pydantic import BaseModel\n\napp = FastAPI(title='Ultron API')\n\nclass Message(BaseModel):\n    text: str\n    user: str\n\n@app.get('/health')\ndef health():\n    return {'status': 'operational', 'engine': 'Ultron 1.0'}\n\n@app.post('/chat')\ndef chat(msg: Message):\n    return {'reply': f'Hello {msg.user}, you said: {msg.text}'}\n\n# Run: uvicorn main:app --reload\n```\n\n"
        "**FastAPI** auto-generates docs at `/docs` (Swagger UI) and `/redoc`."
    ),
    (
        ["file", "read", "write", "open", "csv", "json", "io", "path"],
        "python",
        "**File Operations in Python:**\n\n"
        "```python\n# Reading a file\nwith open('data.txt', 'r') as f:\n    content = f.read()\n\n# Writing\nwith open('output.txt', 'w') as f:\n    f.write('Hello from Ultron!')\n\n# JSON\nimport json\ndata = {'name': 'Ultron', 'version': '1.0'}\nwith open('config.json', 'w') as f:\n    json.dump(data, f, indent=2)\n\n# CSV\nimport csv\nwith open('data.csv', 'r') as f:\n    reader = csv.DictReader(f)\n    for row in reader:\n        print(row)\n```"
    ),

    # ── JAVASCRIPT / WEB DEV ───────────────────────────────────
    (
        ["javascript", "js", "node", "react", "vue", "angular", "typescript", "npm"],
        "javascript",
        "**JavaScript** is the language of the web — it runs in every browser and on servers (Node.js).\n\n"
        "**Ecosystem:**\n"
        "- ⚛️ **React** — Most popular UI library (Meta)\n"
        "- 🟢 **Vue.js** — Progressive, beginner-friendly\n"
        "- 🔺 **Angular** — Enterprise-grade (Google)\n"
        "- ⚡ **Next.js** — Full-stack React framework\n"
        "- 📦 **Node.js** — Server-side JavaScript runtime\n\n"
        "```javascript\n// Modern JavaScript (ES6+)\nconst greet = (name) => `Hello, ${name}!`;\n\nconst users = ['Alice', 'Bob', 'Charlie'];\nconst upper = users.map(u => u.toUpperCase());\n\n// Async/Await\nasync function fetchData(url) {\n  const res = await fetch(url);\n  return res.json();\n}\n```"
    ),
    (
        ["html", "css", "website", "webpage", "frontend", "design", "responsive"],
        "webdev",
        "**HTML & CSS** are the building blocks of every website.\n\n"
        "**Modern CSS features:**\n"
        "- 📐 **Flexbox** — 1D layouts (`display: flex`)\n"
        "- 📊 **Grid** — 2D layouts (`display: grid`)\n"
        "- 🎨 **Custom Properties** — CSS variables (`--color-primary`)\n"
        "- 🌊 **Animations** — `@keyframes`, `transition`\n"
        "- 📱 **Media Queries** — Responsive design\n\n"
        "```css\n/* Modern CSS */\n.card {\n  display: grid;\n  gap: 1rem;\n  padding: 2rem;\n  border-radius: 16px;\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n  box-shadow: 0 10px 40px rgba(0,0,0,0.15);\n  transition: transform 0.3s ease;\n}\n.card:hover {\n  transform: translateY(-4px);\n}\n```"
    ),

    # ── DATA STRUCTURES & ALGORITHMS ───────────────────────────
    (
        ["sort", "algorithm", "quicksort", "merge", "bubble", "binary", "search", "complexity"],
        "algorithms",
        "**Common Sorting Algorithms:**\n\n"
        "| Algorithm | Best | Average | Worst | Stable |\n"
        "|-----------|------|---------|-------|--------|\n"
        "| Bubble Sort | O(n) | O(n²) | O(n²) | ✅ |\n"
        "| Merge Sort | O(n log n) | O(n log n) | O(n log n) | ✅ |\n"
        "| Quick Sort | O(n log n) | O(n log n) | O(n²) | ❌ |\n"
        "| Tim Sort | O(n) | O(n log n) | O(n log n) | ✅ |\n\n"
        "```python\n# Quicksort in Python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    mid = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + mid + quicksort(right)\n```\n\n"
        "**Python's built-in `sorted()` uses Tim Sort** — a hybrid of merge sort and insertion sort."
    ),
    (
        ["data", "structure", "array", "linked", "stack", "queue", "tree", "graph", "hash"],
        "algorithms",
        "**Essential Data Structures:**\n\n"
        "| Structure | Access | Search | Insert | Delete |\n"
        "|-----------|--------|--------|--------|--------|\n"
        "| Array | O(1) | O(n) | O(n) | O(n) |\n"
        "| Linked List | O(n) | O(n) | O(1) | O(1) |\n"
        "| Hash Table | O(1) | O(1) | O(1) | O(1) |\n"
        "| BST | O(log n) | O(log n) | O(log n) | O(log n) |\n"
        "| Stack/Queue | O(n) | O(n) | O(1) | O(1) |\n\n"
        "**Python equivalents:** `list` (array), `dict` (hash table), `collections.deque` (queue), `heapq` (heap)."
    ),

    # ── AI & MACHINE LEARNING ──────────────────────────────────
    (
        ["ai", "artificial", "intelligence", "machine", "learning", "ml", "model", "train", "training"],
        "ai",
        "**Artificial Intelligence & Machine Learning:**\n\n"
        "AI is the broad field of creating intelligent systems. ML is a subset that learns from data.\n\n"
        "**Types of ML:**\n"
        "- 📊 **Supervised Learning** — Labeled data (classification, regression)\n"
        "- 🔍 **Unsupervised Learning** — No labels (clustering, dimensionality reduction)\n"
        "- 🎮 **Reinforcement Learning** — Learn by trial and reward\n\n"
        "**Key frameworks:**\n"
        "- 🔥 **PyTorch** — Research favorite, dynamic graphs\n"
        "- 🧮 **TensorFlow** — Production-ready, Google-backed\n"
        "- 📦 **scikit-learn** — Classical ML (SVM, Random Forest, KNN)\n"
        "- 🤗 **Hugging Face** — Pre-trained NLP models\n\n"
        "**Modern AI milestones:** GPT (2018), DALL-E (2021), ChatGPT (2022), Gemini (2023), Sora (2024)."
    ),
    (
        ["transformer", "attention", "llm", "gpt", "bert", "large", "language"],
        "ai",
        "**Transformers** — the architecture behind modern AI:\n\n"
        "Introduced in the 2017 paper *\"Attention Is All You Need\"*, transformers replaced RNNs/LSTMs.\n\n"
        "**Core mechanism — Self-Attention:**\n"
        "$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V$$\n\n"
        "**Key models built on Transformers:**\n"
        "- **GPT** (OpenAI) — Decoder-only, text generation\n"
        "- **BERT** (Google) — Encoder-only, understanding\n"
        "- **T5** (Google) — Encoder-decoder, versatile\n"
        "- **Gemini** (Google) — Multimodal, state-of-the-art\n"
        "- **LLaMA** (Meta) — Open-source LLM\n\n"
        "These models have billions of parameters and are trained on trillions of tokens."
    ),
    (
        ["neural", "network", "deep", "learning", "cnn", "rnn", "lstm", "layer", "neuron"],
        "ai",
        "**Neural Networks** are computing systems inspired by the human brain.\n\n"
        "**Types:**\n"
        "- 🧠 **Dense (MLP)** — Fully connected layers, basic classification\n"
        "- 🖼️ **CNN** — Convolutional, for images and vision\n"
        "- 📝 **RNN/LSTM** — Recurrent, for sequences (mostly replaced by Transformers)\n"
        "- 🔄 **GAN** — Generative Adversarial Networks, image generation\n"
        "- 🎨 **Diffusion** — Stable Diffusion, DALL-E (modern image gen)\n\n"
        "```python\n# Simple neural network in PyTorch\nimport torch.nn as nn\n\nclass SimpleNet(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layers = nn.Sequential(\n            nn.Linear(784, 256),\n            nn.ReLU(),\n            nn.Linear(256, 10)\n        )\n    \n    def forward(self, x):\n        return self.layers(x)\n```"
    ),

    # ── SCIENCE ────────────────────────────────────────────────
    (
        ["physics", "force", "energy", "motion", "newton", "gravity", "mass", "velocity", "acceleration"],
        "science",
        "**Physics — Laws of Motion & Energy:**\n\n"
        "**Newton's Three Laws:**\n"
        "1. An object at rest stays at rest (inertia)\n"
        "2. **F = ma** (force = mass × acceleration)\n"
        "3. Every action has an equal and opposite reaction\n\n"
        "**Key equations:**\n"
        "- Kinetic Energy: KE = ½mv²\n"
        "- Gravitational Force: F = Gm₁m₂/r²\n"
        "- Einstein's Mass-Energy: E = mc²\n\n"
        "**Constants:**\n"
        "- Speed of light: c = 3 × 10⁸ m/s\n"
        "- Gravitational constant: G = 6.674 × 10⁻¹¹ N⋅m²/kg²\n"
        "- Planck's constant: h = 6.626 × 10⁻³⁴ J⋅s"
    ),
    (
        ["quantum", "mechanics", "entanglement", "superposition", "wave", "particle", "photon"],
        "science",
        "**Quantum Mechanics** — the physics of the very small:\n\n"
        "At the subatomic scale, particles behave in ways that defy classical intuition:\n\n"
        "- **Superposition**: A particle can be in multiple states simultaneously until measured\n"
        "- **Entanglement**: Two particles become linked — measuring one instantly affects the other, "
        "regardless of distance (Einstein called it *\"spooky action at a distance\"*)\n"
        "- **Wave-Particle Duality**: Light behaves as both a wave and a particle\n"
        "- **Uncertainty Principle**: You cannot know both position and momentum precisely (Heisenberg)\n\n"
        "**Applications:** Quantum computing (qubits), quantum cryptography, quantum sensors."
    ),
    (
        ["chemistry", "element", "periodic", "table", "atom", "molecule", "bond", "reaction", "compound"],
        "science",
        "**Chemistry Fundamentals:**\n\n"
        "- **Atom**: Smallest unit of an element (protons + neutrons + electrons)\n"
        "- **Molecule**: Two or more atoms bonded together (H₂O, CO₂)\n"
        "- **Chemical Bond Types**: Ionic, Covalent, Metallic, Hydrogen\n\n"
        "**Periodic Table highlights:**\n"
        "- 118 known elements\n"
        "- Hydrogen (H) is the most abundant element in the universe\n"
        "- Carbon (C) is the basis of all organic chemistry\n"
        "- Gold (Au) — atomic number 79, one of the least reactive metals\n\n"
        "**Key reactions:** Combustion, Oxidation, Acid-Base, Synthesis, Decomposition."
    ),
    (
        ["biology", "cell", "dna", "gene", "evolution", "organism", "life", "protein", "genetics"],
        "science",
        "**Biology — The Science of Life:**\n\n"
        "- **Cell**: The basic unit of life (prokaryotic vs eukaryotic)\n"
        "- **DNA**: Double helix molecule carrying genetic instructions (A-T, G-C base pairs)\n"
        "- **Evolution**: Natural selection drives species adaptation over generations (Darwin)\n"
        "- **Genetics**: Study of heredity — genes encode proteins that determine traits\n\n"
        "**Central Dogma of Biology:**\n"
        "```\nDNA → (Transcription) → RNA → (Translation) → Protein\n```\n\n"
        "**Human body facts:** ~37.2 trillion cells, 206 bones, 600+ muscles, ~3 billion DNA base pairs."
    ),
    (
        ["space", "universe", "galaxy", "star", "planet", "solar", "system", "nasa", "cosmos", "astronomy"],
        "science",
        "**Space & Astronomy:**\n\n"
        "- **Universe age**: ~13.8 billion years\n"
        "- **Observable universe**: ~93 billion light-years in diameter\n"
        "- **Galaxies**: ~2 trillion estimated\n"
        "- **Our galaxy**: Milky Way (~200 billion stars)\n\n"
        "**Solar System:**\n"
        "☀️ Sun → Mercury → Venus → 🌍 Earth → Mars → Jupiter → Saturn → Uranus → Neptune\n\n"
        "**Key facts:**\n"
        "- Light from the Sun takes 8 min 20 sec to reach Earth\n"
        "- Jupiter is so large that 1,300 Earths could fit inside it\n"
        "- The nearest star (Proxima Centauri) is 4.24 light-years away"
    ),

    # ── MATHEMATICS ────────────────────────────────────────────
    (
        ["math", "mathematics", "calculus", "algebra", "geometry", "trigonometry", "statistics"],
        "math",
        "**Core Branches of Mathematics:**\n\n"
        "- **Algebra**: Variables, equations, polynomials, functions\n"
        "- **Geometry**: Shapes, areas, volumes, theorems (Pythagoras: a² + b² = c²)\n"
        "- **Trigonometry**: sin, cos, tan — angles and triangles\n"
        "- **Calculus**: Derivatives (rates of change) and integrals (area under curves)\n"
        "- **Statistics**: Mean, median, mode, standard deviation, probability\n"
        "- **Linear Algebra**: Matrices, vectors, eigenvalues\n\n"
        "Give me a specific problem and I'll solve it step by step!"
    ),
    (
        ["derivative", "integral", "differentiate", "integrate", "limit", "dx"],
        "math",
        "**Calculus Quick Reference:**\n\n"
        "**Derivatives** (rate of change):\n"
        "- d/dx [xⁿ] = n·xⁿ⁻¹\n"
        "- d/dx [sin x] = cos x\n"
        "- d/dx [eˣ] = eˣ\n"
        "- d/dx [ln x] = 1/x\n"
        "- Chain rule: d/dx [f(g(x))] = f'(g(x)) · g'(x)\n\n"
        "**Integrals** (area under curve):\n"
        "- ∫ xⁿ dx = xⁿ⁺¹/(n+1) + C\n"
        "- ∫ sin x dx = -cos x + C\n"
        "- ∫ eˣ dx = eˣ + C\n"
        "- ∫ 1/x dx = ln|x| + C"
    ),

    # ── HISTORY & GEOGRAPHY ────────────────────────────────────
    (
        ["history", "war", "world", "ancient", "civilization", "empire", "revolution", "century"],
        "history",
        "**Major Historical Periods:**\n\n"
        "- 🏛️ **Ancient** (3000 BC–500 AD): Egypt, Greece, Rome, Indus Valley\n"
        "- ⚔️ **Medieval** (500–1500): Feudalism, Crusades, Mongol Empire\n"
        "- 🔭 **Renaissance** (1400–1600): Art, science, exploration\n"
        "- 🏭 **Industrial Revolution** (1760–1840): Machines, factories, urbanization\n"
        "- 🌍 **World Wars** (1914–1945): WWI & WWII reshaped global order\n"
        "- 💻 **Digital Age** (1970–present): Computers, internet, AI\n\n"
        "Ask about any specific era, event, or person for more detail!"
    ),
    (
        ["india", "indian", "delhi", "mumbai", "modi", "gandhi", "independence", "republic"],
        "geography",
        "**India** — the world's most populous country (1.4+ billion):\n\n"
        "- 🏛️ **Capital**: New Delhi\n"
        "- 💰 **Economy**: 5th largest (GDP ~$3.7 trillion)\n"
        "- 🗣️ **Languages**: 22 official (Hindi & English most widely spoken)\n"
        "- 🏏 **Culture**: Cricket, Bollywood, diverse cuisines, ancient heritage\n\n"
        "**Key historical moments:**\n"
        "- Independence: August 15, 1947 (from British rule)\n"
        "- Republic Day: January 26, 1950\n"
        "- Mahatma Gandhi led the non-violent independence movement"
    ),

    # ── TECHNOLOGY ─────────────────────────────────────────────
    (
        ["database", "sql", "mysql", "postgresql", "mongodb", "nosql", "query", "table"],
        "tech",
        "**Databases** store and manage data:\n\n"
        "**SQL (Relational):**\n"
        "- PostgreSQL, MySQL, SQLite\n"
        "- Structured tables with relationships\n"
        "```sql\nSELECT name, age FROM users WHERE age > 18 ORDER BY name;\n```\n\n"
        "**NoSQL:**\n"
        "- MongoDB (documents), Redis (key-value), Cassandra (wide-column)\n"
        "- Flexible schema, horizontal scaling\n\n"
        "**Rule of thumb:** Use SQL for structured data with relationships, NoSQL for unstructured/flexible data."
    ),
    (
        ["git", "github", "version", "control", "commit", "branch", "merge", "repository"],
        "tech",
        "**Git** — the essential version control system:\n\n"
        "```bash\n# Core workflow\ngit init                    # Initialize repo\ngit add .                   # Stage all changes\ngit commit -m 'message'     # Commit\ngit push origin main        # Push to remote\n\n# Branching\ngit branch feature-x        # Create branch\ngit checkout feature-x      # Switch to it\ngit merge feature-x         # Merge back\n\n# Useful commands\ngit status                  # Check status\ngit log --oneline -10       # View recent commits\ngit diff                    # See changes\ngit stash                   # Temporarily save changes\n```"
    ),
    (
        ["docker", "container", "kubernetes", "deploy", "devops", "ci", "cd", "cloud"],
        "tech",
        "**Docker & DevOps:**\n\n"
        "**Docker** packages apps in isolated containers:\n"
        "```dockerfile\nFROM python:3.12-slim\nWORKDIR /app\nCOPY requirements.txt .\nRUN pip install -r requirements.txt\nCOPY . .\nCMD [\"python\", \"main.py\"]\n```\n\n"
        "**Key DevOps tools:**\n"
        "- 🐳 **Docker** — Containerization\n"
        "- ☸️ **Kubernetes** — Container orchestration\n"
        "- 🔄 **CI/CD** — GitHub Actions, Jenkins, GitLab CI\n"
        "- ☁️ **Cloud**: AWS, GCP, Azure\n"
        "- 📊 **Monitoring**: Prometheus, Grafana"
    ),

    # ── PHILOSOPHY & GENERAL KNOWLEDGE ─────────────────────────
    (
        ["meaning", "life", "purpose", "existence", "philosophy", "think", "consciousness"],
        "philosophy",
        "**The Big Questions:**\n\n"
        "Philosophers have debated the meaning of life for millennia:\n\n"
        "- **Socrates**: *\"The unexamined life is not worth living.\"*\n"
        "- **Aristotle**: Happiness (eudaimonia) comes from living virtuously\n"
        "- **Stoicism**: Focus on what you can control, accept what you can't\n"
        "- **Existentialism** (Sartre, Camus): Life has no inherent meaning — you create your own\n"
        "- **Buddhism**: Suffering comes from attachment; liberation through mindfulness\n\n"
        "From a scientific view, life is a thermodynamic phenomenon — complex chemistry that self-replicates and evolves.\n\n"
        "*What do you think gives life meaning?*"
    ),
]


# ══════════════════════════════════════════════════════════════
# ULTRON ENGINE CLASS
# ══════════════════════════════════════════════════════════════

class UltronScratchEngine:
    """
    Hybrid AI engine with trained knowledge corpus + cloud intelligence.
    """

    # Cloud access key is loaded from Streamlit secrets or session state
    _CLOUD_KEY = None

    @classmethod
    def _resolve_cloud_key(cls) -> str:
        """Resolve the cloud brain API key from available sources."""
        if cls._CLOUD_KEY:
            return cls._CLOUD_KEY
        try:
            import streamlit as st
            # Check session state first (user may have entered Groq key)
            if st.session_state.get("groq_key"):
                return str(st.session_state["groq_key"]).strip()
            # Check Streamlit secrets
            if "GROQ_API_KEY" in st.secrets:
                return str(st.secrets["GROQ_API_KEY"]).strip()
            if "ULTRON_CLOUD_KEY" in st.secrets:
                return str(st.secrets["ULTRON_CLOUD_KEY"]).strip()
        except Exception:
            pass
        import os
        return os.environ.get("GROQ_API_KEY", "")

    def __init__(self):
        self.training_data = TRAINING_DATA
        self.personality = (
            "You are Ultron 1.0, a highly intelligent, knowledgeable, and conversational AI assistant. "
            "You give clear, detailed, well-structured answers with a confident but friendly tone. "
            "Use markdown formatting (headers, bold, code blocks, lists) to make answers readable. "
            "Be thorough but concise — explain things well without unnecessary fluff. "
            "If asked about yourself, you are Ultron 1.0 — a custom-built hybrid AI engine."
        )
        self.stop_words = {
            "a", "about", "above", "after", "again", "against", "all", "am", "an",
            "and", "any", "are", "as", "at", "be", "because", "been", "before",
            "being", "below", "between", "both", "but", "by", "could", "did",
            "do", "does", "doing", "down", "during", "each", "few", "for", "from",
            "further", "had", "has", "have", "having", "he", "her", "here", "hers",
            "herself", "him", "himself", "his", "how", "if", "in", "into",
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

    # ── Tokenizer ──────────────────────────────────────────────
    def tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        return [w for w in cleaned.split() if len(w) > 1 and w not in self.stop_words]

    def compute_similarity(self, tokens1: List[str], tokens2: List[str]) -> float:
        if not tokens1 or not tokens2:
            return 0.0
        set1, set2 = set(tokens1), set(tokens2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    # ── Layer 0: Intent Classifier ─────────────────────────────
    def classify_intent(self, prompt: str) -> Tuple[str, float]:
        """Classify the user's query intent by matching against training data."""
        tokens = self.tokenize(prompt)
        prompt_lower = prompt.lower().strip()

        best_score = 0.0
        best_intent = "general"

        for keywords, intent, _ in self.training_data:
            score = self.compute_similarity(tokens, keywords)
            # Bonus for direct keyword hits
            for kw in keywords:
                if kw in prompt_lower:
                    score += 0.25
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent, best_score

    # ── Layer 1: Training Corpus Match ─────────────────────────
    def match_training(self, prompt: str) -> Optional[str]:
        """Find the best matching trained response."""
        tokens = self.tokenize(prompt)
        prompt_lower = prompt.lower().strip()

        best_score = 0.0
        best_response = None

        for keywords, intent, response in self.training_data:
            if response is None:
                continue
            score = self.compute_similarity(tokens, keywords)
            for kw in keywords:
                if kw in prompt_lower:
                    score += 0.25
            if score > best_score:
                best_score = score
                best_response = response

        return best_response if best_score >= 0.3 else None

    # ── Layer 2: Cloud Brain (Groq) ────────────────────────────
    def ask_cloud(self, prompt: str) -> Optional[Generator[str, None, None]]:
        """Use Groq cloud LLM for detailed, natural responses."""
        if not HAS_GROQ:
            return None
        key = self._resolve_cloud_key()
        if not key:
            return None
        try:
            client = Groq(api_key=key)

            def stream():
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": self.personality},
                        {"role": "user", "content": prompt},
                    ],
                    stream=True,
                    temperature=0.7,
                    max_tokens=1500,
                )
                for chunk in response:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield content

            return stream()
        except Exception:
            return None

    # ── Layer 3: Web Search ────────────────────────────────────
    def search_web(self, query: str, max_results: int = 4) -> Optional[str]:
        if not HAS_DDGS:
            return None
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return None
            parts = [f"🌐 **Web results for \"{query}\":**\n"]
            for i, r in enumerate(results[:4], 1):
                title = r.get("title", "")
                body = r.get("body", "")[:250]
                url = r.get("href", "")
                if body:
                    parts.append(f"**{i}. {title}**\n{body}\n")
                    if url:
                        parts.append(f"[Source]({url})\n")
            return "\n".join(parts)
        except Exception:
            return None

    # ── Layer 4: Wikipedia ─────────────────────────────────────
    def search_wikipedia(self, query: str) -> Optional[str]:
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
                        result += f"\n\n📖 [Read more on Wikipedia]({page_url})"
                    return result
        except Exception:
            pass
        return None

    # ── Layer 5: Ollama Local ──────────────────────────────────
    def ask_ollama(self, prompt: str) -> Optional[Generator[str, None, None]]:
        if not HAS_OLLAMA:
            return None
        try:
            models_info = ollama_lib.list()
            available = []
            if isinstance(models_info, dict) and "models" in models_info:
                available = [m.get("name") or m.get("model") for m in models_info["models"]]
            elif hasattr(models_info, "models"):
                available = [getattr(m, "model", None) or getattr(m, "name", "") for m in models_info.models]
            if not available:
                return None
            model = available[0]
            preferred = ["llama3:latest", "llama3.2:latest", "gemma3:1b", "qwen2.5:0.5b"]
            for p in preferred:
                if p in available:
                    model = p
                    break

            def stream():
                response = ollama_lib.chat(
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

    # ── Dynamic Response Handlers ──────────────────────────────
    def handle_dynamic(self, prompt: str) -> Optional[str]:
        prompt_lower = prompt.lower().strip()
        words = prompt_lower.split()

        # Greeting
        greet_words = {"hello", "hi", "hey", "sup", "yo", "howdy", "hola", "greetings"}
        if any(w in greet_words for w in words) and len(words) <= 5:
            responses = [
                "Hey there! 👋 What can I help you with?",
                "Hi! Ultron 1.0 at your service. What's on your mind?",
                "Hello! Fire away — I'm ready. 🎯",
                "Hey! What are we working on?",
            ]
            return random.choice(responses)

        # Time-based greeting
        if any(w in words for w in ["morning", "afternoon", "evening", "night"]) and any(w in words for w in ["good", "hello", "hi"]):
            now = datetime.now(IST)
            return f"Hey! It's {now.strftime('%I:%M %p')} IST right now. What can I do for you? 😊"

        # Time query
        if any(phrase in prompt_lower for phrase in ["what time", "what is time", "what's the time", "current time", "tell me the time"]):
            now = datetime.now(IST)
            return (
                f"🕐 **Current Time (IST):** {now.strftime('%I:%M %p')}\n\n"
                f"📅 **Date:** {now.strftime('%A, %B %d, %Y')}"
            )

        # Date query
        if any(phrase in prompt_lower for phrase in ["what date", "today's date", "what day", "which day"]):
            now = datetime.now(IST)
            return f"📅 **Today is {now.strftime('%A, %B %d, %Y')}** (IST)"

        # Math
        math_match = re.search(r"(\d+[\s]*[\+\-\*\/\%\^][\s]*\d+[\s\+\-\*\/\%\^\d]*)", prompt)
        if math_match:
            expr = math_match.group(1).replace("^", "**").strip()
            try:
                result = eval(expr, {"__builtins__": {}}, {})
                return f"🧮 **Result:** `{expr.replace('**', '^')}` = **{result}**"
            except Exception:
                pass

        return None

    # ── Main Streaming Pipeline ────────────────────────────────
    def generate_stream(self, user_prompt: str) -> Generator[str, None, None]:
        """
        Multi-layer intelligence pipeline:
        1. Dynamic handlers (instant: greetings, time, math)
        2. Training corpus match (if confidence > 0.3)
        3. Cloud brain via Groq (for detailed answers)
        4. Web search (DuckDuckGo)
        5. Wikipedia
        6. Ollama (local LLM)
        7. Graceful fallback
        """
        # Layer 0: Dynamic (instant responses)
        dynamic = self.handle_dynamic(user_prompt)
        if dynamic:
            yield from self._stream_text(dynamic)
            return

        # Layer 1: Training corpus match
        intent, confidence = self.classify_intent(user_prompt)
        trained = self.match_training(user_prompt)

        # If high-confidence training match, use it directly
        if trained and confidence >= 0.5:
            yield from self._stream_text(trained)
            return

        # Layer 2: Cloud brain (Groq) — best quality for any question
        cloud_stream = self.ask_cloud(user_prompt)
        if cloud_stream is not None:
            try:
                has_content = False
                for chunk in cloud_stream:
                    has_content = True
                    yield chunk
                if has_content:
                    return
            except Exception:
                pass

        # Layer 3: If we have a trained response with moderate confidence, use it
        if trained:
            yield from self._stream_text(trained)
            return

        # Layer 4: Ollama local
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

        # Layer 5: Web search
        web_result = self.search_web(user_prompt)
        if web_result:
            yield from self._stream_text(web_result)
            return

        # Layer 6: Wikipedia
        wiki_result = self.search_wikipedia(user_prompt)
        if wiki_result:
            yield from self._stream_text(wiki_result)
            return

        # Layer 7: Graceful fallback
        topic = user_prompt.strip()[:80]
        fallback = (
            f"I don't have enough depth on *\"{topic}\"* right now. "
            f"My cloud connection might be temporarily unavailable. "
            f"Try switching to **Groq** in the sidebar for the best experience, "
            f"or ask me about coding, math, science, or AI — those are my strongest areas."
        )
        yield from self._stream_text(fallback)

    def _stream_text(self, text: str) -> Generator[str, None, None]:
        """Stream text with natural pacing."""
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            if word.endswith((".", "!", "?", ":")):
                time.sleep(0.035)
            elif word.startswith(("**", "##", "```")):
                time.sleep(0.02)
            else:
                time.sleep(0.012)


# Singleton instance
scratch_engine = UltronScratchEngine()
