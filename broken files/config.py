import os
from dotenv import load_dotenv
load_dotenv()

# ── Model Configuration ────────────────────────────────────────────────────────
MODEL_OLLAMA = "gemma4:e4b"
OLLAMA_BASE_URL = "http://localhost:11434"

# ── NVIDIA NIM Configuration ──────────────────────────────────────────────────
NIM_API_KEY = os.getenv("NIM_API_KEY", "")
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
NIM_MODEL = "google/gemma-4-31b-it"
NIM_RPM_LIMIT = 40

# ── Legacy Gemini (kept for fallback, but not active) ─────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MODEL_GEMINI = "gemini-2.0-flash"

# ── Multi-Model Router ────────────────────────────────────────────────────────
# Streamlined Ollama lineup: Gemma 4 E4B (default) + specialists
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
    # Everything else goes to gemma4:e4b
    "general": "gemma4:e4b",
    "tutoring": "gemma4:e4b",
    "explanation": "gemma4:e4b",
    "creative": "gemma4:e4b",
    "writing": "gemma4:e4b",
    "presentation": "gemma4:e4b",
    "vision": "gemma4:e4b",
    "fast": "gemma4:e4b",
    "simple": "gemma4:e4b",
    "quick": "gemma4:e4b",
}

# Default fallback model
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"

# ── General Settings ───────────────────────────────────────────────────────────
MAX_ITERATIONS = 15
MAX_HISTORY = 40
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
