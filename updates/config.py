import os
from dotenv import load_dotenv
load_dotenv()

# ── Model Configuration ────────────────────────────────────────────────────────
MODEL_OLLAMA = "llama3.1:8b"
MODEL_GEMINI = "gemini-2.0-flash"
OLLAMA_BASE_URL = "http://localhost:11434"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ── Multi-Model Router ────────────────────────────────────────────────────────
# Available Ollama models with their capabilities
OLLAMA_MODELS = {
    "qwen2.5:7b": {
        "size": "4.7 GB",
        "strengths": ["general", "reasoning", "multilingual", "conversation"],
        "category": "general",
    },
    "granite4.1:3b": {
        "size": "2.1 GB",
        "strengths": ["fast", "lightweight", "simple_tasks", "quick_answers", "structured_output"],
        "category": "lightweight",
    },
    "deepseek-r1:8b": {
        "size": "5.2 GB",
        "strengths": ["reasoning", "math", "logic", "code_debug", "analysis", "deep_thinking"],
        "category": "reasoning",
    },
    "llama3.1:8b": {
        "size": "4.9 GB",
        "strengths": ["general", "conversation", "tool_use", "instruction_following"],
        "category": "general",
    },
    "llama3.2:3b": {
        "size": "2.0 GB",
        "strengths": ["fast", "lightweight", "simple_tasks", "quick_answers"],
        "category": "lightweight",
    },
    "qwen2.5-coder:7b": {
        "size": "4.7 GB",
        "strengths": ["code", "programming", "debugging", "code_generation", "technical"],
        "category": "coding",
    },
    "llama3:8b": {
        "size": "4.7 GB",
        "strengths": ["general", "conversation", "creative"],
        "category": "general",
    },
    # Vision models (install with: ollama pull llava)
    "llava": {
        "size": "4.7 GB",
        "strengths": ["vision", "image_analysis", "image_description", "ocr"],
        "category": "vision",
    },
    "llava:7b": {
        "size": "4.7 GB",
        "strengths": ["vision", "image_analysis", "image_description", "ocr"],
        "category": "vision",
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
    "fast": "granite4.1:3b",
    "simple": "granite4.1:3b",
    "quick": "granite4.1:3b",
    "tutoring": "qwen2.5:7b",
    "explanation": "qwen2.5:7b",
    "teaching": "qwen2.5:7b",
    "presentation": "qwen2.5:7b",
    "creative": "llama3:8b",
    "writing": "qwen2.5:7b",
}

# Default fallback model
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"

# Fast model for quick tasks (presentations, simple queries)
FAST_MODEL = "granite4.1:3b"

# ── General Settings ───────────────────────────────────────────────────────────
MAX_ITERATIONS = 15
MAX_HISTORY = 40
SEARCH_MAX_RESULTS = 5
MEMORY_FILE = "memory.json"
AGENT_NAME = "Aria"

# "ollama", "gemini", or "auto"
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
