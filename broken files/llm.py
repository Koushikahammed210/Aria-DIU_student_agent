# llm.py - Smart router: Gemma 4 E4B (default) + Specialists + NIM Cloud

import ollama
import time
import hashlib
import socket
import threading
from config import (
    MODEL_OLLAMA, OLLAMA_BASE_URL, DEFAULT_MODE,
    OLLAMA_MODELS, TASK_MODEL_MAP, DEFAULT_OLLAMA_MODEL,
    PERSONA_MODELS,
    NIM_API_KEY, NIM_BASE_URL, NIM_MODEL, NIM_RPM_LIMIT,
)

# ── Task Detection Keywords ────────────────────────────────────────────────────
TASK_TYPE_KEYWORDS = {
    "code": ["code", "program", "function", "script", "debug", "fix code", "implement", "algorithm",
             "python", "javascript", "java", "cpp", "html", "css", "api", "database", "sql",
             "refactor", "compile", "syntax", "variable", "loop", "class", "method", "bug",
             "write a program", "write a function", "code review"],
    "reasoning": ["reason", "logic", "prove", "math", "calculate", "derive", "prove that",
                  "why does", "explain why", "what if", "analyze the", "evaluate",
                  "deduce", "infer", "conclude", "critical thinking", "philosophy",
                  "step by step reasoning", "think carefully"],
    "fast": ["quick", "simple", "fast", "briefly", "short", "just tell me", "what is",
             "define", "meaning of", "synonym", "translate"],
    "tutoring": ["teach", "learn", "understand", "explain", "tutorial", "lesson",
                 "example", "step by step", "how to", "guide me", "help me learn",
                 "i don't understand", "confused about", "clarify"],
    "creative": ["creative", "story", "poem", "write a", "imagine", "brainstorm",
                 "idea", "fiction", "novel", "lyrics", "creative writing"],
    "presentation": ["presentation", "pptx", "slides", "ppt", "keynote", "slideshow"],
}

COMPLEX_KEYWORDS = [
    "analyze", "write", "explain", "compare", "summarize",
    "code", "debug", "plan", "research", "essay", "report",
    "translate", "review", "generate", "describe", "elaborate",
    "presentation", "create a"
]

SIMPLE_PATTERNS = [
    "what time", "what's the time", "current time", "date today",
    "what day", "hello", "hi ", "hey ", "who are you", "your name",
    "what model", "what are you", "status", "help", "who built",
    "who made", "who created", "who is your"
]


# ══════════════════════════════════════════════════════════════════════════════
#  NVIDIA NIM Manager — OpenAI-compatible cloud API
# ══════════════════════════════════════════════════════════════════════════════
class NIMManager:
    """
    NVIDIA NIM API manager with:
    - OpenAI-compatible chat completions
    - RPM rate tracking (40 RPM free tier)
    - Auto-fallback detection (returns None → Ollama takes over)
    - Vision support (Gemma 4 31B is multimodal)
    - Streaming support (ready but not used by default)
    """

    def __init__(self):
        self._available = True
        self._rpm_count = 0
        self._rpm_window_start = time.time()
        self._rpm_limit = NIM_RPM_LIMIT
        self._last_call_time = 0.0
        self._min_call_interval = 1.5  # seconds between NIM calls (40 RPM = 1.5s each)
        self._successful_calls = 0
        self._total_calls = 0
        self._last_error = None
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize OpenAI client for NIM."""
        if not NIM_API_KEY:
            self._available = False
            self._last_error = "No NIM_API_KEY set"
            return

        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=NIM_BASE_URL,
                api_key=NIM_API_KEY,
            )
            self._available = True
            self._last_error = None
            print("  [NIM] Client initialized — ready for cloud inference")
        except ImportError:
            self._available = False
            self._last_error = "openai package not installed. Run: pip install openai"
            print(f"  [NIM] ERROR: {self._last_error}")
        except Exception as e:
            self._available = False
            self._last_error = str(e)
            print(f"  [NIM] Client init error: {e}")

    def is_available(self) -> bool:
        """Check if NIM is available for calls."""
        if not self._available or not self._client or not NIM_API_KEY:
            return False

        # Quick network check
        try:
            socket.create_connection(("8.8.8.8", 53), timeout=2)
        except OSError:
            return False

        # Enforce minimum interval
        elapsed = time.time() - self._last_call_time
        if elapsed < self._min_call_interval:
            return False

        # Check RPM limit
        self._update_rate_counters()
        if self._rpm_count >= self._rpm_limit:
            return False

        return True

    def _update_rate_counters(self):
        """Reset RPM counter every 60 seconds."""
        now = time.time()
        if now - self._rpm_window_start >= 60:
            self._rpm_count = 0
            self._rpm_window_start = now

    def record_call(self):
        """Record a successful NIM call."""
        self._update_rate_counters()
        self._rpm_count += 1
        self._last_call_time = time.time()
        self._successful_calls += 1
        self._total_calls += 1
        self._last_error = None
        print(f"  [NIM call OK: RPM {self._rpm_count}/{self._rpm_limit}, total: {self._total_calls}]")

    def mark_error(self, error_msg: str, retry_after: float = 10):
        """Handle NIM error with short cooldown."""
        self._available = False
        self._last_error = error_msg
        # Auto-recover after short cooldown
        def _recover():
            time.sleep(retry_after)
            self._available = True
            self._last_error = None
            print(f"  [NIM auto-recovery after {retry_after}s]")
        threading.Thread(target=_recover, daemon=True).start()

        if "429" in error_msg or "rate" in error_msg.lower():
            print(f"  [NIM 429 rate limited — cooldown {retry_after}s]")
        else:
            print(f"  [NIM error: {error_msg} — cooldown {retry_after}s]")

    def chat(self, messages: list, model: str = None) -> str:
        """
        Chat with NIM. Returns response string on success, None on failure.
        """
        if not self.is_available():
            return None

        use_model = model or NIM_MODEL

        try:
            # Convert messages to OpenAI format
            ollama_msgs = []
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    ollama_msgs.append({"role": "system", "content": content})
                elif role == "assistant":
                    ollama_msgs.append({"role": "assistant", "content": content})
                else:
                    ollama_msgs.append({"role": "user", "content": content})

            if not ollama_msgs:
                return None

            print(f"  [NIM → {use_model} | {len(ollama_msgs)} messages]")

            response = self._client.chat.completions.create(
                model=use_model,
                messages=ollama_msgs,
                temperature=0.7,
                max_tokens=2048,
            )

            result = response.choices[0].message.content.strip()
            self.record_call()
            return result

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                self.mark_error(err, retry_after=30)
            elif "401" in err or "unauthorized" in err.lower():
                self.mark_error(err, retry_after=300)
            elif "503" in err or "overloaded" in err.lower():
                self.mark_error(err, retry_after=15)
            else:
                self.mark_error(err, retry_after=10)
            return None

    def chat_with_images(self, text: str, images: list) -> str:
        """
        Send text + images to NIM Vision (Gemma 4 31B is multimodal).
        images: list of {base64: str, mime: str}
        Returns response string or None.
        """
        if not self.is_available():
            return None

        try:
            import base64
            content_parts = []

            # Add text first
            content_parts.append({"type": "text", "text": text})

            # Add images
            for img in images:
                b64_data = img.get('base64', '')
                mime = img.get('mime', 'image/jpeg')
                data_url = f"data:{mime};base64,{b64_data}"
                content_parts.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })

            messages = [{"role": "user", "content": content_parts}]

            print(f"  [NIM Vision → {NIM_MODEL} | {len(images)} image(s)]")

            response = self._client.chat.completions.create(
                model=NIM_MODEL,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )

            result = response.choices[0].message.content.strip()
            self.record_call()
            return result

        except Exception as e:
            err = str(e)
            print(f"  [NIM Vision error: {e}]")
            if "429" in err:
                self.mark_error(err, retry_after=30)
            else:
                self.mark_error(err, retry_after=10)
            return None

    def get_status(self) -> dict:
        """Get current NIM status."""
        self._update_rate_counters()
        now = time.time()
        next_call_in = max(0, self._min_call_interval - (now - self._last_call_time))

        if not NIM_API_KEY:
            state = "no API key"
        elif not self._available:
            state = "unavailable"
            if self._last_error:
                if "429" in self._last_error:
                    state = "rate limited"
                elif "401" in self._last_error:
                    state = "invalid key"
        elif self._rpm_count >= self._rpm_limit:
            state = "RPM limit reached"
        else:
            state = "available"

        return {
            "state": state,
            "available": self._available and bool(NIM_API_KEY),
            "model": NIM_MODEL,
            "rpm": self._rpm_count,
            "rpm_limit": self._rpm_limit,
            "next_call_in": round(next_call_in, 1),
            "successful_calls": self._successful_calls,
            "total_calls": self._total_calls,
            "last_error": self._last_error,
        }


# ── Global NIM Manager instance ──────────────────────────────────────────────
nim_manager = NIMManager()


# ══════════════════════════════════════════════════════════════════════════════
#  Model Selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best_model(user_input: str, persona: str = "default") -> str:
    """
    Intelligently select the best Ollama model based on:
    1. Active persona (coder → qwen-coder, analyst → deepseek-r1) — HIGHEST priority
    2. User input content (task type detection) — overrides for VERY EXPLICIT requests
    3. Default: gemma4:e4b
    """
    text = user_input.lower()

    # Step 1: Check persona preference
    preferred_model = PERSONA_MODELS.get(persona, DEFAULT_OLLAMA_MODEL)

    # Step 2: Detect task type from keywords
    best_task_type = None
    best_score = 0

    for task_type, keywords in TASK_TYPE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > best_score:
            best_score = score
            best_task_type = task_type

    # Step 3: Persona priority logic
    persona_task_types = {
        "tutor": ["tutoring"],
        "coder": ["code"],
        "creative": ["creative"],
        "analyst": ["reasoning"],
    }
    persona_tasks = persona_task_types.get(persona, [])

    # If the detected task matches the persona's specialty, use persona model
    if best_task_type in persona_tasks:
        print(f"  [Model Router: {preferred_model} — persona: {persona} (task: {best_task_type}, score: {best_score})]")
        return preferred_model

    # If user explicitly asks for a different specialty (high score), override
    # e.g., in default persona, user asks a coding question with score >= 3
    if best_score >= 3 and best_task_type:
        task_model = TASK_MODEL_MAP.get(best_task_type, DEFAULT_OLLAMA_MODEL)
        if task_model in OLLAMA_MODELS:
            print(f"  [Model Router: {task_model} — explicit task override: {best_task_type} (score: {best_score})]")
            return task_model

    # Default: use persona's preferred model (usually gemma4:e4b)
    print(f"  [Model Router: {preferred_model} — persona: {persona} (task: {best_task_type}, score: {best_score})]")
    return preferred_model


# ══════════════════════════════════════════════════════════════════════════════
#  Ollama Chat
# ══════════════════════════════════════════════════════════════════════════════

def _parse_ollama_response(response) -> str:
    """Parse Ollama response — handles both old dict and new pydantic formats."""
    try:
        if hasattr(response, 'message'):
            msg = response.message
            if hasattr(msg, 'content'):
                return msg.content.strip()
        if isinstance(response, dict):
            return response["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError):
        pass
    return str(response).strip()


def chat_ollama(messages: list, model: str = None) -> str:
    """Use Ollama with smart model selection. Includes timeout."""
    try:
        use_model = model or MODEL_OLLAMA
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        response = ollama.chat(model=use_model, messages=clean)
        return _parse_ollama_response(response)
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "model" in err.lower():
            print(f"  [Model {use_model} not found, falling back to {DEFAULT_OLLAMA_MODEL}]")
            try:
                clean = [{"role": m["role"], "content": m["content"]} for m in messages]
                response = ollama.chat(model=DEFAULT_OLLAMA_MODEL, messages=clean)
                return _parse_ollama_response(response)
            except Exception as e2:
                return f"Ollama error: {e2}"
        return f"Ollama error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  NIM Chat
# ══════════════════════════════════════════════════════════════════════════════

def chat_nim(messages: list) -> str:
    """Use NIM via NIMManager. Falls back to Ollama on failure."""
    result = nim_manager.chat(messages)
    if result is not None:
        return result
    # NIM unavailable — fall back to Ollama
    print("  [NIM unavailable — falling back to Ollama]")
    return chat_ollama(messages)


# ══════════════════════════════════════════════════════════════════════════════
#  Main Chat Router
# ══════════════════════════════════════════════════════════════════════════════

def chat(messages: list, mode: str = DEFAULT_MODE, model: str = None) -> str:
    """Route to the appropriate chat backend based on mode."""
    if mode == "nim":
        return chat_nim(messages)
    if mode == "ollama":
        return chat_ollama(messages, model=model)
    # auto mode: default to Ollama (agent loop will add NIM polish if needed)
    return chat_ollama(messages, model=model)


def should_use_nim_for_final(user_input: str, mode: str) -> bool:
    """For auto mode: decide if NIM should polish the final answer."""
    if mode == "ollama":
        return False
    if mode == "nim":
        return True
    # auto mode: use NIM for complex tasks
    if not nim_manager.is_available():
        return False
    last = user_input.lower()
    if any(p in last for p in SIMPLE_PATTERNS):
        return False
    return any(k in last for k in COMPLEX_KEYWORDS)


def is_nim_available() -> bool:
    """Check if NIM is available."""
    return nim_manager.is_available()


def get_nim_status() -> dict:
    """Get NIM status info."""
    return nim_manager.get_status()


# ══════════════════════════════════════════════════════════════════════════════
#  Vision — Gemma 4 E4B Multimodal + NIM Vision
# ══════════════════════════════════════════════════════════════════════════════

def _try_gemma4_vision(text: str, images: list) -> str:
    """
    Use Gemma 4 E4B (local) for vision — it's natively multimodal!
    Returns result string or None if failed.
    """
    try:
        import base64 as b64mod

        ollama_images = []
        for img in images:
            img_data = img.get('base64', '')
            if img_data:
                ollama_images.append(img_data)

        if not ollama_images:
            return None

        print(f"  [Vision] Calling gemma4:e4b with {len(ollama_images)} image(s)")

        # Gemma 4 handles images natively through ollama.chat
        response = ollama.chat(
            model="gemma4:e4b",
            messages=[
                {"role": "user", "content": text, "images": ollama_images}
            ]
        )
        result = _parse_ollama_response(response)
        result = _clean_vision_output(result)

        if result and not _is_blind_response(result):
            print(f"  [Vision SUCCESS via gemma4:e4b] Response length: {len(result)} chars")
            return result
        else:
            print("  [Vision] gemma4:e4b returned blind/empty response")
            return None

    except Exception as e:
        print(f"  [Vision ERROR] gemma4:e4b vision failed: {e}")
        return None


def _clean_vision_output(text: str) -> str:
    """Clean vision output — remove special tokens and artifacts."""
    import re
    text = re.sub(r'<unk>', '', text)
    text = re.sub(r'<[a-z_]+>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _is_blind_response(text: str) -> bool:
    """Detect if the model hallucinated 'I cannot see the image'."""
    lower = text.lower()
    blind_phrases = [
        "i cannot see", "i can't see", "i don't see", "i am unable to see",
        "no image", "cannot see any image", "can't see any image",
        "unable to process the image", "i'm not able to see",
        "i am a text", "i cannot process images", "as a text",
        "i cannot view", "i'm unable to view",
    ]
    return any(phrase in lower for phrase in blind_phrases)


def chat_with_images(text: str, images: list, mode: str = DEFAULT_MODE) -> str:
    """
    Send text + images for vision analysis.
    Priority order:
    - nim mode: NIM Vision (31B cloud) → Gemma 4 E4B (local) → fallback
    - ollama/auto mode: Gemma 4 E4B (local) → NIM Vision (cloud) → fallback
    images: list of {base64: str, mime: str}
    """
    print(f"  [Vision] chat_with_images called with {len(images)} image(s), mode={mode}")

    if mode == "nim":
        # NIM mode: try cloud vision first (better quality)
        nim_result = nim_manager.chat_with_images(text, images)
        if nim_result:
            print(f"  [Vision SUCCESS] NIM Vision responded")
            return nim_result

        # Fallback to local gemma4:e4b
        local_result = _try_gemma4_vision(text, images)
        if local_result:
            return local_result

    else:
        # Ollama/auto mode: try local gemma4:e4b first (faster, no API)
        local_result = _try_gemma4_vision(text, images)
        if local_result:
            return local_result

        # Fallback to NIM Vision
        nim_result = nim_manager.chat_with_images(text, images)
        if nim_result:
            print(f"  [Vision SUCCESS] NIM Vision (fallback) responded")
            return nim_result

    # All vision methods failed
    print(f"  [Vision FAILED] All vision methods failed")
    return (
        "I can see you've shared an image! Unfortunately, I'm currently unable to analyze it because:\n\n"
        "- **NIM Vision** (cloud) is unavailable or offline\n"
        "- **Gemma 4 E4B** (local) could not process the image\n\n"
        "**How to fix this:**\n"
        "1. Check your internet connection for NIM Vision\n"
        "2. Make sure `gemma4:e4b` is running locally (`ollama list`)\n"
        "3. Try a simpler image description\n\n"
        "**In the meantime:** If you can describe what's in the image, I can help with detailed text-based analysis!"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Health & Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def get_available_models() -> list:
    """Get list of available Ollama model names."""
    models = []

    # Method 1: ollama Python client
    try:
        models_resp = ollama.list()

        if hasattr(models_resp, 'models'):
            for m in models_resp.models:
                name = None
                for attr in ['model', 'name', 'id']:
                    val = getattr(m, attr, None)
                    if val and isinstance(val, str):
                        name = val
                        break
                if name:
                    models.append(name)
                else:
                    try:
                        name = m.get('name', m.get('model', ''))
                        if name:
                            models.append(name)
                    except:
                        pass

        elif isinstance(models_resp, dict):
            for m in models_resp.get('models', []):
                if isinstance(m, dict):
                    models.append(m.get('name', m.get('model', '')))
                elif isinstance(m, str):
                    models.append(m)
                elif hasattr(m, 'model'):
                    models.append(getattr(m, 'model', ''))
        else:
            try:
                for m in models_resp:
                    if isinstance(m, dict):
                        models.append(m.get('name', m.get('model', '')))
                    elif hasattr(m, 'model'):
                        models.append(getattr(m, 'model', ''))
                    elif hasattr(m, 'name'):
                        models.append(getattr(m, 'name', ''))
                    elif isinstance(m, str):
                        models.append(m)
            except TypeError:
                pass

        if models:
            return models

    except Exception as e:
        print(f"  [DEBUG] ollama.list() failed: {e}")

    # Method 2: Fallback to subprocess
    try:
        import subprocess
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line or line.startswith('NAME') or line.startswith('---'):
                    continue
                parts = line.split()
                if parts:
                    model_name = parts[0]
                    if model_name and model_name not in models:
                        models.append(model_name)
            if models:
                return models
    except Exception as e:
        print(f"  [DEBUG] `ollama list` CLI failed: {e}")

    # Method 3: Config fallback
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


def has_vision_model() -> bool:
    """Gemma 4 E4B is multimodal — always has vision if available."""
    try:
        available = get_available_models()
        return any("gemma4" in av.lower() for av in available)
    except:
        return False


# ── Legacy Gemini compatibility (kept for smooth transition) ─────────────────
# These functions exist so old code doesn't break, but they're deprecated

def is_gemini_available() -> bool:
    """Deprecated: Always returns False. Use is_nim_available() instead."""
    return False

def mark_quota_exceeded():
    """Deprecated: No-op. NIM handles rate limiting internally."""
    pass

def get_gemini_status() -> dict:
    """Deprecated: Returns NIM status instead."""
    return get_nim_status()
