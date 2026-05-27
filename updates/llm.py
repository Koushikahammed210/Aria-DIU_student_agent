# llm.py - Smart router: Multi-Ollama model selection + Gemini for final answers

import ollama
from google import genai
from google.genai import types
from config import (
    MODEL_OLLAMA, MODEL_GEMINI, GEMINI_API_KEY, DEFAULT_MODE,
    OLLAMA_MODELS, TASK_MODEL_MAP, DEFAULT_OLLAMA_MODEL
)

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Session-level quota tracker
_gemini_quota_exceeded = False
_gemini_quota_exceeded_at = None  # timestamp of when quota was hit

SIMPLE_PATTERNS = [
    "what time", "what's the time", "current time", "date today",
    "what day", "hello", "hi ", "hey ", "who are you", "your name",
    "what model", "what are you", "status", "help", "who built",
    "who made", "who created", "who is your"
]

COMPLEX_KEYWORDS = [
    "analyze", "write", "explain", "compare", "summarize",
    "code", "debug", "plan", "research", "essay", "report",
    "translate", "review", "generate", "describe", "elaborate"
]

# Task-type keywords for model routing
TASK_TYPE_KEYWORDS = {
    "code": ["code", "program", "function", "script", "debug", "fix code", "implement", "algorithm",
             "python", "javascript", "java", "cpp", "html", "css", "api", "database", "sql",
             "refactor", "compile", "syntax", "variable", "loop", "class", "method", "bug"],
    "reasoning": ["reason", "logic", "prove", "math", "calculate", "derive", "prove that",
                  "why does", "explain why", "what if", "analyze the", "evaluate",
                  "deduce", "infer", "conclude", "critical thinking", "philosophy"],
    "fast": ["quick", "simple", "fast", "briefly", "short", "just tell me", "what is",
             "define", "meaning of", "synonym", "translate"],
    "tutoring": ["teach", "learn", "understand", "explain", "tutorial", "lesson",
                 "example", "step by step", "how to", "guide me", "help me learn",
                 "i don't understand", "confused about", "clarify"],
    "creative": ["creative", "story", "poem", "write a", "imagine", "brainstorm",
                 "idea", "fiction", "novel", "lyrics", "creative writing"],
}


def is_gemini_available() -> bool:
    """Check if Gemini is available, with auto-recovery after quota reset."""
    global _gemini_quota_exceeded, _gemini_quota_exceeded_at
    import time

    # Auto-recover after 60 seconds (quota might reset)
    if _gemini_quota_exceeded and _gemini_quota_exceeded_at:
        if time.time() - _gemini_quota_exceeded_at > 60:
            _gemini_quota_exceeded = False
            print("  [Gemini quota auto-recovery — trying again]")

    if _gemini_quota_exceeded:
        return False
    if not GEMINI_API_KEY or not client:
        return False
    import socket
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except OSError:
        return False


def mark_quota_exceeded():
    global _gemini_quota_exceeded, _gemini_quota_exceeded_at
    import time
    _gemini_quota_exceeded = True
    _gemini_quota_exceeded_at = time.time()
    print("  [Gemini quota exceeded — switching to Ollama for now. Will retry in 60s]")


def select_best_model(user_input: str, persona: str = "default") -> str:
    """
    Intelligently select the best Ollama model based on:
    1. Active persona (tutor, coder, etc.) — HIGHEST priority
    2. User input content (task type detection) — only overrides for EXPLICIT requests
    3. Task complexity

    Returns: model name string
    """
    text = user_input.lower()

    # Persona-based default model selection
    persona_models = {
        "tutor": "qwen2.5:7b",       # Best for explanations & multilingual
        "coder": "qwen2.5-coder:7b",  # Best for code
        "creative": "llama3:8b",       # Good for creative tasks
        "analyst": "deepseek-r1:8b",   # Best for analysis/reasoning
    }

    preferred_model = persona_models.get(persona)

    # Detect task type from input
    best_task_type = None
    best_score = 0

    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_task_type = task_type

    # IMPORTANT: Persona takes priority UNLESS the user makes an EXPLICIT
    # request that contradicts it (e.g., Tutor mode + "write python code")
    # We only override persona if the task detection score is VERY high (>=3)
    # and the task type is different from the persona's specialty.

    if preferred_model and preferred_model in OLLAMA_MODELS:
        # Check if the detected task type conflicts with persona
        persona_task_types = {
            "tutor": ["tutoring"],
            "coder": ["code"],
            "creative": ["creative"],
            "analyst": ["reasoning"],
        }
        persona_tasks = persona_task_types.get(persona, [])

        # If detected task matches persona specialty, stick with persona model
        if best_task_type in persona_tasks or best_score < 3:
            print(f"  [Model Router: {preferred_model} — persona: {persona} (task: {best_task_type}, score: {best_score})]")
            return preferred_model

        # Only override if user makes a VERY explicit request (score >= 3)
        # and it's different from the persona's specialty
        if best_score >= 3 and best_task_type:
            task_model = TASK_MODEL_MAP.get(best_task_type, DEFAULT_OLLAMA_MODEL)
            if task_model in OLLAMA_MODELS:
                print(f"  [Model Router: {task_model} — explicit task override: {best_task_type} (score: {best_score}, persona: {persona})]")
                return task_model

        print(f"  [Model Router: {preferred_model} — persona: {persona}]")
        return preferred_model

    # No persona preference — use task detection
    if best_score >= 2 and best_task_type:
        task_model = TASK_MODEL_MAP.get(best_task_type, DEFAULT_OLLAMA_MODEL)
        if task_model in OLLAMA_MODELS:
            print(f"  [Model Router: {task_model} — task type: {best_task_type}]")
            return task_model

    # Final fallback
    print(f"  [Model Router: {DEFAULT_OLLAMA_MODEL} — default]")
    return DEFAULT_OLLAMA_MODEL


def chat_ollama(messages: list, model: str = None) -> str:
    """Use Ollama with smart model selection."""
    try:
        use_model = model or MODEL_OLLAMA
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        response = ollama.chat(model=use_model, messages=clean)
        return response["message"]["content"].strip()
    except Exception as e:
        err = str(e)
        # If model not found, fallback to default
        if "not found" in err.lower() or "model" in err.lower():
            print(f"  [Model {use_model} not found, falling back to {DEFAULT_OLLAMA_MODEL}]")
            try:
                clean = [{"role": m["role"], "content": m["content"]} for m in messages]
                response = ollama.chat(model=DEFAULT_OLLAMA_MODEL, messages=clean)
                return response["message"]["content"].strip()
            except Exception as e2:
                return f"Ollama error: {e2}"
        return f"Ollama error: {e}"


def chat_gemini(messages: list) -> str:
    """Use Gemini — only for final answers on complex tasks. With improved error handling."""
    global _gemini_quota_exceeded
    if not is_gemini_available():
        return chat_ollama(messages)
    try:
        system_text = ""
        history = []
        for m in messages:
            if m["role"] == "system":
                system_text = m["content"]
            elif m["role"] == "user":
                history.append(types.Content(
                    role="user",
                    parts=[types.Part(text=m["content"])]
                ))
            elif m["role"] == "assistant":
                history.append(types.Content(
                    role="model",
                    parts=[types.Part(text=m["content"])]
                ))
        if not history:
            return chat_ollama(messages)
        last_content = history[-1].parts[0].text
        prior_history = history[:-1]
        config = types.GenerateContentConfig(
            system_instruction=system_text if system_text else None,
        )
        session = client.chats.create(
            model=MODEL_GEMINI,
            history=prior_history,
            config=config,
        )
        response = session.send_message(last_content)
        return response.text.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "exhausted" in err.lower() or "RESOURCE_EXHAUSTED" in err:
            mark_quota_exceeded()
            return chat_ollama(messages)
        if "503" in err or "overloaded" in err.lower():
            print("  [Gemini overloaded — falling back to Ollama]")
            return chat_ollama(messages)
        if "403" in err or "permission" in err.lower():
            print("  [Gemini API key invalid — falling back to Ollama]")
            return chat_ollama(messages)
        return f"Gemini error: {e}"


def chat(messages: list, mode: str = DEFAULT_MODE, model: str = None) -> str:
    """Route to the appropriate chat backend."""
    if mode == "ollama":
        return chat_ollama(messages, model=model)
    if mode == "gemini":
        return chat_gemini(messages)
    return chat_ollama(messages, model=model)


def should_use_gemini_for_final(user_input: str, mode: str) -> bool:
    if mode == "ollama":
        return False
    if not is_gemini_available():
        return False
    last = user_input.lower()
    if any(p in last for p in SIMPLE_PATTERNS):
        return False
    if mode == "gemini":
        return True
    return any(k in last for k in COMPLEX_KEYWORDS)


# Vision-capable Ollama models to try as fallback
_VISION_MODELS = ["llava", "llava:7b", "llava:13b", "bakllava", "moondream", "minicpm-v"]


def _try_ollama_vision(text: str, images: list) -> str:
    """
    Try Ollama vision models (llava, etc.) for image analysis.
    Returns result or None if no vision model available.
    """
    try:
        available = get_available_models()
        for vision_model in _VISION_MODELS:
            if any(vision_model in av for av in available):
                # Find the exact model name
                exact_name = next((av for av in available if vision_model in av), vision_model)
                print(f"  [Trying Ollama vision model: {exact_name}]")
                import base64
                # Ollama vision API accepts images as a list of base64 or URLs
                ollama_images = []
                for img in images:
                    img_data = img.get('base64', '')
                    if img_data:
                        ollama_images.append(img_data)
                response = ollama.chat(
                    model=exact_name,
                    messages=[{
                        "role": "user",
                        "content": text,
                        "images": ollama_images
                    }]
                )
                return response["message"]["content"].strip()
        return None  # No vision model found
    except Exception as e:
        print(f"  [Ollama vision fallback failed: {e}]")
        return None


def chat_with_images(text: str, images: list) -> str:
    """
    Send text + images for vision analysis.
    Tries: 1) Gemini Vision → 2) Ollama vision models (llava) → 3) Text-only fallback
    images: list of {base64: str, mime: str}
    """
    # Step 1: Try Gemini Vision (best quality)
    if is_gemini_available():
        try:
            import base64
            parts = []
            for img in images:
                parts.append(types.Part(
                    inline_data=types.Blob(
                        mime_type=img.get('mime', 'image/jpeg'),
                        data=base64.b64decode(img['base64'])
                    )
                ))
            parts.append(types.Part(text=text))
            response = client.models.generate_content(
                model=MODEL_GEMINI,
                contents=[types.Content(role="user", parts=parts)]
            )
            return response.text.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower():
                mark_quota_exceeded()
                # Don't return error yet — try Ollama vision fallback
            elif "503" in err or "overloaded" in err.lower():
                pass  # Try Ollama vision fallback
            else:
                return f"Image analysis error: {e}"

    # Step 2: Try Ollama vision models (llava, moondream, etc.)
    vision_result = _try_ollama_vision(text, images)
    if vision_result:
        return vision_result

    # Step 3: Helpful fallback message
    return (
        "I can see you've shared an image, but image analysis requires either:\n"
        "1. **Gemini API** (currently unavailable — quota exceeded or not configured)\n"
        "2. **Ollama vision model** like `llava` (not installed yet)\n\n"
        "To enable image analysis:\n"
        "- Run `ollama pull llava` to install a vision model\n"
        "- Or wait for Gemini quota to reset\n\n"
        "If you can describe what's in the image, I can help you with text-based analysis!"
    )


def get_available_models() -> list:
    """Get list of available Ollama models."""
    try:
        models_resp = ollama.list()
        models = []
        for m in models_resp.get('models', []):
            models.append(m.get('name', ''))
        return models
    except Exception:
        return list(OLLAMA_MODELS.keys())


def check_ollama_health() -> dict:
    """Check which Ollama models are actually running/available."""
    result = {}
    try:
        available = get_available_models()
        for model_name in OLLAMA_MODELS:
            is_available = any(model_name in av for av in available)
            result[model_name] = {
                "available": is_available,
                "info": OLLAMA_MODELS[model_name],
            }
    except Exception as e:
        result["error"] = str(e)
    return result
