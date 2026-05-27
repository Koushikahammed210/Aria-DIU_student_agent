import os
from dotenv import load_dotenv
load_dotenv()

# ── Version ────────────────────────────────────────────────────────────────────
ARIA_VERSION = "1.4"

# ── Model Configuration ────────────────────────────────────────────────────────
MODEL_OLLAMA = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── Fast Model for Simple Tasks ──────────────────────────────────────────────
# Lightweight model that fits in 4GB VRAM for instant responses
# Install: ollama pull llama3.2:3b
# Alternatives: gemma2:2b (smaller but less structured output)
FAST_MODEL = "llama3.2:3b"

# ── Keep-Alive Setting ────────────────────────────────────────────────────────
# Keep models loaded in memory for this duration after last request
# Prevents cold-start delay (model loading from disk = 20-60s)
KEEP_ALIVE = "10m"

# ── NVIDIA NIM Configuration ──────────────────────────────────────────────────
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "google/gemma-4-31b-it"          # Default NIM model for vision/complex tasks
NIM_PRESENTATION_MODEL = "meta/llama-3.1-8b-instruct"  # Fast, cheap model for presentation outlines
NIM_RPM_LIMIT = 40

# Presentation generation strategy:
# "nim"       → Always use NIM cloud for outline (fastest, 1 API call)
# "nim+local" → Try NIM first, fallback to local fast model (recommended)
# "local"     → Only use local models (offline mode)
PRESENTATION_STRATEGY = "nim+local"

# ── Legacy Gemini (kept for fallback, but not active) ─────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_GEMINI = "gemini-2.0-flash"

# ── Multi-Model Router ────────────────────────────────────────────────────────
# Streamlined Ollama lineup: Gemma 4 E4B (default) + Specialists + Fast Lane
OLLAMA_MODELS = {
    "gemma4:e4b": {
        "size": "9.6 GB",
        "strengths": ["general", "conversation", "multimodal", "vision", "audio",
                      "tool_calling", "reasoning", "agentic", "tutoring", "creative",
                      "presentation", "structured_output"],
        "category": "general",
    },
    "qwen2.5-coder:7b": {
        "size": "4.7 GB",
        "strengths": ["code", "programming", "debugging", "code_generation", "technical"],
        "category": "coding",
    },
    "deepseek-r1:8b": {
        "size": "5.2 GB",
        "strengths": ["reasoning", "math", "logic", "code_debug", "analysis", "deep_thinking"],
        "category": "reasoning",
    },
    "nomic-embed-text": {
        "size": "274 MB",
        "strengths": ["embeddings", "search", "rag"],
        "category": "embedding",
    },
    "llama3.2:3b": {
        "size": "2.0 GB",
        "strengths": ["fast", "simple", "quick_response", "structured_output", "presentation_outline"],
        "category": "fast",
    },
}

# Task-to-model mapping for intelligent routing
TASK_MODEL_MAP = {
    "code": "qwen2.5-coder:7b",
    "programming": "qwen2.5-coder:7b",
    "debugging": "qwen2.5-coder:7b",
    "reasoning": "deepseek-r1:8b",
    "math": "deepseek-r1:8b",
    "logic": "deepseek-r1:8b",
    "analysis": "deepseek-r1:8b",
    "deep_thinking": "deepseek-r1:8b",
    # Fast lane — lightweight model for instant responses
    "fast": "llama3.2:3b",
    "simple": "llama3.2:3b",
    "quick": "llama3.2:3b",
    "greeting": "llama3.2:3b",
    "presentation_outline": "llama3.2:3b",
    # Everything else goes to gemma4:e4b
    "general": "gemma4:e4b",
    "tutoring": "gemma4:e4b",
    "explanation": "gemma4:e4b",
    "creative": "gemma4:e4b",
    "writing": "gemma4:e4b",
    "presentation": "gemma4:e4b",
    "vision": "gemma4:e4b",
}

# Default fallback model
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"

# ── Quiz Configuration ──────────────────────────────────────────────────────
QUIZ_MAX_QUESTIONS = 30
QUIZ_DIFFICULTY_LEVELS = ["easy", "medium", "hard", "mixed"]
QUIZ_GENERATION_MODEL_NIM = "meta/llama-3.1-8b-instruct"   # Cloud (fast, reliable, structured JSON output — free NIM tier)
QUIZ_GENERATION_MODEL_LOCAL_EASY = "llama3.2:3b"           # Offline easy/medium
QUIZ_GENERATION_MODEL_LOCAL_HARD = "gemma4:e4b"            # Offline hard/mixed
QUIZ_GENERATION_TIMEOUT = 60  # seconds for quiz generation (llama-3.1-8b is fast)
QUIZ_SOURCE_MAX_CHARS = 6000  # max chars of file content to send (smaller = faster NIM response)

# ── General Settings ───────────────────────────────────────────────────────────
MAX_ITERATIONS = 15
MAX_HISTORY = 20  # Reduced from 40 — less context sent per request = faster
SEARCH_MAX_RESULTS = 5
MEMORY_FILE = "memory.json"
AGENT_NAME = "Aria"

# "ollama", "nim", or "auto"
# auto = Ollama does the work, NIM polishes the final answer
DEFAULT_MODE = "auto"

# ── Upload Settings ────────────────────────────────────────────────────────────
UPLOAD_FOLDER = "uploads"
MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_FILE_EXTENSIONS = {
    "pdf", "txt", "pptx", "docx", "csv", "json",
    "py", "js", "html", "css", "java", "cpp", "c",
    "md", "xlsx", "png", "jpg", "jpeg", "gif", "webp",
}

# ── Persona Definitions ────────────────────────────────────────────────────────
PERSONAS = {
    "default": {
        "name": "Aria",
        "icon": "✦",
        "color": "#6c63ff",
        "description": "General-purpose assistant",
    },
    "tutor": {
        "name": "Tutor",
        "icon": "🎓",
        "color": "#34d399",
        "description": "Patient teacher who explains topics step by step",
    },
    "coder": {
        "name": "Coder",
        "icon": "💻",
        "color": "#60a5fa",
        "description": "Code-focused assistant for programming tasks",
    },
    "creative": {
        "name": "Creative",
        "icon": "🎨",
        "color": "#f472b6",
        "description": "Creative writer and brainstormer",
    },
    "analyst": {
        "name": "Analyst",
        "icon": "📊",
        "color": "#fbbf24",
        "description": "Data analyst and research assistant",
    },
}

# ── Persona → Model Mapping ───────────────────────────────────────────────────
# Each persona has a preferred local model. This is HIGHEST priority in routing.
PERSONA_MODELS = {
    "default": "gemma4:e4b",
    "tutor": "gemma4:e4b",
    "coder": "qwen2.5-coder:7b",   # Always use coder model for coding persona
    "creative": "gemma4:e4b",
    "analyst": "deepseek-r1:8b",    # Deep reasoning for analysis
}
