# llm.py - Smart router: Gemma 4 E4B (default) + Specialists + Fast Lane + NIM Cloud
# OPTIMIZED: keep_alive, fast model, pre-warming, minimal prompts for simple tasks

import ollama
import time
import hashlib
import socket
import threading
from config import (
    MODEL_OLLAMA, OLLAMA_BASE_URL, DEFAULT_MODE,
    OLLAMA_MODELS, TASK_MODEL_MAP, DEFAULT_OLLAMA_MODEL,
    PERSONA_MODELS, FAST_MODEL, KEEP_ALIVE,
    NIM_API_KEY, NIM_BASE_URL, NIM_MODEL, NIM_RPM_LIMIT,
    NIM_PRESENTATION_MODEL, PRESENTATION_STRATEGY,
    ARIA_VERSION,
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
    "who made", "who created", "who is your",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "bye", "goodbye", "ok", "okay",
    "how are you", "what's up", "sup", "what up",
    "yes", "no", "maybe", "sure", "cool",
    "lol", "haha", "nice", "great", "awesome",
]

# Patterns that are definitely simple and should use the fast model
# These are greetings, acknowledgments, and very short Q&A
FAST_PATH_PATTERNS = [
    "hello", "hi", "hey", "howdy", "sup", "what's up", "hola",
    "good morning", "good afternoon", "good evening", "good night",
    "how are you", "how r u", "how's it going",
    "thanks", "thank you", "thx", "ty",
    "bye", "goodbye", "see you", "later",
    "ok", "okay", "k", "kk", "sure", "cool", "nice", "great",
    "who are you", "what are you", "your name", "what's your name",
    "what model", "what models", "who made you", "who built you",
    "who created you", "who is your creator",
    "yes", "no", "maybe", "lol", "haha", "awesome",
]


# ── Simple Message Detection ──────────────────────────────────────────────────

def is_simple_message(text: str) -> bool:
    """
    Detect if a message is simple enough for the fast model.
    Returns True for greetings, short questions, acknowledgments.
    Returns False for anything needing deep reasoning, code, or tools.
    """
    text_lower = text.lower().strip()

    # Very short messages (under 15 chars) are likely simple
    if len(text_lower) < 15:
        return True

    # Check against known simple patterns
    for pattern in FAST_PATH_PATTERNS:
        if text_lower == pattern or text_lower.startswith(pattern + " ") or text_lower.startswith(pattern + "?"):
            return True

    # Questions that are just definitions or basic facts
    basic_question_starters = [
        "what is ", "what's ", "define ", "meaning of ",
        "who is ", "who was ", "when is ", "when was ",
        "where is ", "where was ",
    ]
    if any(text_lower.startswith(s) for s in basic_question_starters) and len(text_lower) < 60:
        return True

    # Messages with complex keywords should NOT use fast model
    for keyword in COMPLEX_KEYWORDS:
        if keyword in text_lower:
            return False

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  NVIDIA NIM Manager — OpenAI-compatible cloud API
# ══════════════════════════════════════════════════════════════════════════════
class NIMManager:
    """
    NVIDIA NIM API manager v1.3 with:
    - OpenAI-compatible chat completions
    - RPM rate tracking (40 RPM free tier)
    - GRACEFUL error recovery (no more 30s blackouts!)
    - Vision support (Gemma 4 31B is multimodal)
    - Per-model availability tracking
    - Detailed logging for debugging
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
        self._failed_models = {}   # {model_name: fail_count} — track per-model failures
        self._network_ok = True    # Cache network status
        self._last_network_check = 0.0
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

    def is_available(self, model: str = None) -> bool:
        """
        Check if NIM is available for calls.
        v1.3: Cached network check (don't DNS every call!), per-model tracking.
        """
        if not self._client or not NIM_API_KEY:
            return False

        # v1.3: If we're in a short cooldown from mark_error, still allow
        # trying DIFFERENT models. Only block if auth failure or rate limit.
        if not self._available:
            # Check if it's a temporary error that might have recovered
            if self._last_error and ("401" in self._last_error or "unauthorized" in self._last_error.lower()):
                return False  # Auth errors are permanent until key is changed
            # For other errors, check if cooldown has passed
            # (auto-recovery thread will set _available=True)
            return False

        # v1.3: Cached network check — only check every 30 seconds, not every call
        now = time.time()
        if now - self._last_network_check > 30:
            try:
                socket.create_connection(("8.8.8.8", 53), timeout=2)
                self._network_ok = True
                self._last_network_check = now
            except OSError:
                self._network_ok = False
                self._last_network_check = now
                return False
        elif not self._network_ok:
            return False

        # v1.3: Per-model failure check — if a specific model failed 3+ times,
        # skip it for this session (it probably doesn't exist on the free tier)
        if model and model in self._failed_models and self._failed_models[model] >= 3:
            print(f"  [NIM: Skipping {model} — too many failures ({self._failed_models[model]})]")
            return False

        # Enforce minimum interval between calls
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

    def mark_error(self, error_msg: str, retry_after: float = 10, model: str = None):
        """
        Handle NIM error with SHORT cooldown.
        v1.3: Per-model failure tracking, shorter cooldowns, smarter recovery.
        """
        self._last_error = error_msg

        # v1.3: Track per-model failures
        if model:
            self._failed_models[model] = self._failed_models.get(model, 0) + 1
            print(f"  [NIM error on {model}: {error_msg} (fail #{self._failed_models[model]})]")
        else:
            print(f"  [NIM error: {error_msg}]")

        # v1.3: Only mark completely unavailable for auth errors
        # For other errors, just track the model failure and let is_available() decide
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            self._available = False
            retry_after = 300  # 5 min for auth errors
        elif "429" in error_msg or "rate" in error_msg.lower():
            self._available = False
            retry_after = 30  # 30s for rate limits
            print(f"  [NIM 429 rate limited — cooldown {retry_after}s]")
        elif "404" in error_msg or "not found" in error_msg.lower():
            # Model not found — don't mark NIM as unavailable, just this model
            # is_available() with model= will check _failed_models
            print(f"  [NIM: Model {model} not found — will skip in future]")
            return  # Don't start recovery timer — NIM is still available for other models
        elif "503" in error_msg or "overloaded" in error_msg.lower():
            self._available = False
            retry_after = 15  # 15s for overload
        else:
            # Generic errors — short cooldown, don't kill NIM completely
            self._available = False
            retry_after = 5  # Only 5s for unknown errors (was 10s)

        # Auto-recover after cooldown
        def _recover():
            time.sleep(retry_after)
            self._available = True
            if not self._last_error or "401" not in self._last_error:
                self._last_error = None
            print(f"  [NIM auto-recovery after {retry_after}s]")
        threading.Thread(target=_recover, daemon=True).start()

    def chat(self, messages: list, model: str = None, max_tokens: int = 2048) -> str:
        """
        Chat with NIM. Returns response string on success, None on failure.
        v1.3: Per-model availability check, better error handling.
        """
        use_model = model or NIM_MODEL

        if not self.is_available(model=use_model):
            return None

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

            print(f"  [NIM → {use_model} | {len(ollama_msgs)} messages | max_tokens={max_tokens}]")

            response = self._client.chat.completions.create(
                model=use_model,
                messages=ollama_msgs,
                temperature=0.7,
                max_tokens=max_tokens,
            )

            result = response.choices[0].message.content.strip()
            self.record_call()
            # v1.3: Reset per-model failure count on success
            if use_model in self._failed_models:
                self._failed_models[use_model] = 0
            return result

        except Exception as e:
            err = str(e)
            self.mark_error(err, model=use_model)
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
        """Get current NIM status. v1.3: includes per-model failure info."""
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
            "failed_models": dict(self._failed_models),  # v1.3
        }


# ── Global NIM Manager instance ──────────────────────────────────────────────
nim_manager = NIMManager()


# ══════════════════════════════════════════════════════════════════════════════
#  Model Selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best_model(user_input: str, persona: str = "default") -> str:
    """
    Intelligently select the best Ollama model based on:
    1. Simple messages → FAST_MODEL (llama3.2:3b) — NEW fast lane
    2. Active persona (coder → qwen-coder, analyst → deepseek-r1) — HIGHEST priority
    3. User input content (task type detection) — overrides for VERY EXPLICIT requests
    4. Default: gemma4:e4b
    """
    text = user_input.lower()

    # Step 0: Fast-path — route simple messages to the lightweight model
    if is_simple_message(user_input):
        fast_model = TASK_MODEL_MAP.get("fast", FAST_MODEL)
        if fast_model in OLLAMA_MODELS:
            print(f"  [Model Router: {fast_model} — fast lane (simple message)]")
            return fast_model

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
#  Ollama Chat — with keep_alive for warm models
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
    """Use Ollama with smart model selection. Includes keep_alive and timeout."""
    try:
        use_model = model or MODEL_OLLAMA
        clean = [{"role": m["role"], "content": m["content"]} for m in messages]
        # OPTIMIZATION: keep_alive keeps the model in memory for faster subsequent calls
        response = ollama.chat(model=use_model, messages=clean, keep_alive=KEEP_ALIVE)
        return _parse_ollama_response(response)
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "model" in err.lower():
            print(f"  [Model {use_model} not found, falling back to {DEFAULT_OLLAMA_MODEL}]")
            try:
                clean = [{"role": m["role"], "content": m["content"]} for m in messages]
                response = ollama.chat(model=DEFAULT_OLLAMA_MODEL, messages=clean, keep_alive=KEEP_ALIVE)
                return _parse_ollama_response(response)
            except Exception as e2:
                return f"Ollama error: {e2}"
        return f"Ollama error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  Fast Chat — lightweight model with minimal prompt
# ══════════════════════════════════════════════════════════════════════════════

FAST_SYSTEM_PROMPT = (
    "You are Aria, a helpful AI assistant. Keep responses concise and friendly. "
    "You run on local models: Gemma 4 E4B + Qwen Coder + DeepSeek R1 + Llama 3.2 3B (fast). "
    "Built by Koushik Ahammed."
)


def chat_fast(user_input: str) -> str:
    """
    Fast-path chat using the lightweight model (llama3.2:3b).
    Uses minimal system prompt and no history for maximum speed.
    Ideal for greetings, simple Q&A, and basic conversations.
    """
    try:
        messages = [
            {"role": "system", "content": FAST_SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]
        response = ollama.chat(model=FAST_MODEL, messages=messages, keep_alive=KEEP_ALIVE)
        result = _parse_ollama_response(response)
        print(f"  [Fast lane: {FAST_MODEL} responded]")
        return result
    except Exception as e:
        err = str(e)
        if "not found" in err.lower() or "model" in err.lower():
            print(f"  [Fast model {FAST_MODEL} not found, falling back to {DEFAULT_OLLAMA_MODEL}]")
            try:
                messages = [
                    {"role": "system", "content": FAST_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ]
                response = ollama.chat(model=DEFAULT_OLLAMA_MODEL, messages=messages, keep_alive=KEEP_ALIVE)
                return _parse_ollama_response(response)
            except Exception as e2:
                return f"Ollama error: {e2}"
        return f"Ollama error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  Model Pre-warming — load model into memory on startup
# ══════════════════════════════════════════════════════════════════════════════

def prewarm_model(model_name: str = None) -> bool:
    """
    Pre-warm a model by making a tiny request, loading it into memory.
    This eliminates cold-start delay for the first real request.
    Call this on app startup.
    """
    target = model_name or FAST_MODEL
    try:
        print(f"  [Pre-warming {target}...]", end=" ", flush=True)
        start_time = time.time()
        response = ollama.chat(
            model=target,
            messages=[{"role": "user", "content": "hi"}],
            keep_alive=KEEP_ALIVE,
        )
        elapsed = time.time() - start_time
        print(f"ready in {elapsed:.1f}s")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False


def prewarm_all_models():
    """Pre-warm the fast model and default model on startup."""
    print("\n  [Pre-warming models...]")
    # Always pre-warm the fast model first (it's small and most impactful)
    prewarm_model(FAST_MODEL)
    # Then pre-warm the default model
    prewarm_model(DEFAULT_OLLAMA_MODEL)


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
#  Presentation Outline Generation — NIM-first with local fallback
# ══════════════════════════════════════════════════════════════════════════════

def generate_presentation_outline(messages: list) -> str:
    """
    Generate a presentation outline using the best available model.
    
    Strategy (configurable via PRESENTATION_STRATEGY):
    - "nim+local": Try NIM first (fast, cloud GPU), fallback to local fast model
    - "nim":       Always use NIM, no fallback (fails if NIM unavailable)
    - "local":     Only use local models (offline mode)
    
    Returns: outline text string
    """
    strategy = PRESENTATION_STRATEGY
    
    if strategy == "nim":
        # NIM only — no fallback
        result = nim_manager.chat(messages, model=NIM_PRESENTATION_MODEL)
        if result is not None:
            print(f"  [Presentation outline: NIM ({NIM_PRESENTATION_MODEL}) — SUCCESS]")
            return result
        print("  [Presentation outline: NIM FAILED — no fallback configured]")
        return None
    
    elif strategy == "local":
        # Local only — use fast model
        print(f"  [Presentation outline: Local ({FAST_MODEL}) — offline mode]")
        return chat_ollama(messages, model=FAST_MODEL)
    
    else:  # "nim+local" (default recommended)
        # Try NIM first
        if nim_manager.is_available():
            print(f"  [Presentation outline: Trying NIM ({NIM_PRESENTATION_MODEL})...]")
            result = nim_manager.chat(messages, model=NIM_PRESENTATION_MODEL)
            if result is not None:
                print(f"  [Presentation outline: NIM — SUCCESS]")
                return result
            print("  [Presentation outline: NIM failed — falling back to local]")
        else:
            print("  [Presentation outline: NIM unavailable — using local model]")
        
        # Fallback to local fast model
        try:
            result = chat_ollama(messages, model=FAST_MODEL)
            if result and not result.startswith("Ollama error"):
                print(f"  [Presentation outline: Local ({FAST_MODEL}) — SUCCESS]")
                return result
        except Exception as e:
            print(f"  [Presentation outline: Local model failed: {e}]")
        
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Quiz Generation — 3-tier fallback (NIM → local → template)
# ══════════════════════════════════════════════════════════════════════════════

QUIZ_SYSTEM_PROMPT = (
    "You are a quiz generator. Create multiple-choice questions based on the given content.\n"
    "Output ONLY a valid JSON array — no markdown, no explanation, no code fences.\n"
    "Each element must have: question, A, B, C, D, correct (letter), explanation (2-3 sentences)\n"
    "Rules:\n"
    "1. Exactly 4 options per question (A, B, C, D)\n"
    "2. Only ONE correct answer\n"
    "3. Wrong options must be plausible\n"
    "4. Distribute correct answers across A, B, C, D evenly\n"
    "5. Explanations must include additional context beyond just restating the answer"
)

QUIZ_DIFFICULTY_RULES = {
    "easy": "Questions test direct recall, definitions, basic facts. Wrong options are somewhat plausible but one is clearly correct.",
    "medium": "Questions require application and comparison. All options are plausible with subtle differences. Use 'which of the following' style.",
    "hard": "Questions require synthesis, edge cases, multi-step reasoning. All options are very plausible. Requires deep understanding to distinguish.",
    "mixed": "Blend of easy, medium, and hard questions distributed evenly across the quiz.",
}


def generate_quiz_questions(content: str, difficulty: str, count: int) -> list:
    """
    Generate quiz questions with GUARANTEED fallback support.
    
    v1.3 Priority chain:
    1. NIM cloud (meta/llama-3.1-8b-instruct) — fast, reliable structured JSON
    2. Local model (llama3.2:3b for easy/medium, gemma4:e4b for hard) — offline
    3. Template engine (zero AI) — ALWAYS works, pure Python
    
    Returns: list of question dicts (NEVER returns empty list)
    """
    from config import (
        QUIZ_GENERATION_MODEL_NIM, QUIZ_GENERATION_MODEL_LOCAL_EASY,
        QUIZ_GENERATION_MODEL_LOCAL_HARD, QUIZ_SOURCE_MAX_CHARS,
        QUIZ_GENERATION_TIMEOUT, DEFAULT_OLLAMA_MODEL,
    )
    
    # Truncate content if too long
    if len(content) > QUIZ_SOURCE_MAX_CHARS:
        content = content[:QUIZ_SOURCE_MAX_CHARS] + "..."
    
    diff_rules = QUIZ_DIFFICULTY_RULES.get(difficulty, QUIZ_DIFFICULTY_RULES["medium"])
    
    user_prompt = (
        f"CONTENT/TOPIC:\n{content}\n\n"
        f"DIFFICULTY: {difficulty}\n"
        f"NUMBER OF QUESTIONS: {count}\n\n"
        f"DIFFICULTY RULES: {diff_rules}\n\n"
        f"Generate exactly {count} questions. Output ONLY the JSON array."
    )
    
    messages = [
        {"role": "system", "content": QUIZ_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    # ── TIER 1: Try NIM cloud (meta/llama-3.1-8b-instruct) ─────────────────────
    if nim_manager.is_available(model=QUIZ_GENERATION_MODEL_NIM):
        print(f"  [Quiz TIER 1: Trying NIM ({QUIZ_GENERATION_MODEL_NIM})...]")
        try:
            quiz_max_tokens = min(4096, 512 + count * 150)
            result = nim_manager.chat(messages, model=QUIZ_GENERATION_MODEL_NIM, max_tokens=quiz_max_tokens)
            if result:
                from quiz_generator import parse_quiz_json
                parsed = parse_quiz_json(result)
                if parsed and len(parsed) > 0:
                    print(f"  [Quiz TIER 1: NIM SUCCESS — {len(parsed)} questions generated]")
                    return parsed[:count]
                print("  [Quiz TIER 1: NIM returned data but parsing failed — moving to local]")
            else:
                print("  [Quiz TIER 1: NIM returned None — moving to local]")
        except Exception as e:
            print(f"  [Quiz TIER 1: NIM exception: {e} — moving to local]")
    else:
        nim_reason = "unavailable"
        if QUIZ_GENERATION_MODEL_NIM in nim_manager._failed_models:
            nim_reason = f"model failed {nim_manager._failed_models[QUIZ_GENERATION_MODEL_NIM]}x"
        elif not nim_manager._available:
            nim_reason = f"cooldown ({nim_manager._last_error})"
        print(f"  [Quiz TIER 1: NIM skipped — {nim_reason}]")
    
    # ── TIER 2: Local model — pick based on difficulty ──────────────────────────
    if difficulty in ("easy", "medium"):
        local_model = QUIZ_GENERATION_MODEL_LOCAL_EASY
    else:
        local_model = QUIZ_GENERATION_MODEL_LOCAL_HARD
    
    print(f"  [Quiz TIER 2: Trying local model ({local_model})...]")
    try:
        result = chat_ollama(messages, model=local_model)
        if result and not result.startswith("Ollama error"):
            from quiz_generator import parse_quiz_json
            parsed = parse_quiz_json(result)
            if parsed and len(parsed) > 0:
                print(f"  [Quiz TIER 2: Local ({local_model}) SUCCESS — {len(parsed)} questions generated]")
                return parsed[:count]
            print(f"  [Quiz TIER 2: Local model returned data but parsing failed]")
        else:
            print(f"  [Quiz TIER 2: Local model error: {result[:100] if result else 'None'}]")
    except Exception as e:
        print(f"  [Quiz TIER 2: Local model exception: {e}]")
    
    # ── TIER 2b: Try default model if different from local ──────────────────────
    if local_model != DEFAULT_OLLAMA_MODEL:
        print(f"  [Quiz TIER 2b: Trying default model ({DEFAULT_OLLAMA_MODEL})...]")
        try:
            result = chat_ollama(messages, model=DEFAULT_OLLAMA_MODEL)
            if result and not result.startswith("Ollama error"):
                from quiz_generator import parse_quiz_json
                parsed = parse_quiz_json(result)
                if parsed and len(parsed) > 0:
                    print(f"  [Quiz TIER 2b: Default model SUCCESS — {len(parsed)} questions generated]")
                    return parsed[:count]
        except Exception as e:
            print(f"  [Quiz TIER 2b: Default model exception: {e}]")
    
    # ── TIER 3: Template engine (ALWAYS works, zero AI) ─────────────────────────
    print("  [Quiz TIER 3: Using template engine (no AI needed) — GUARANTEED to work]")
    from quiz_generator import generate_template_quiz
    topic = content[:100] if content else "General Knowledge"
    template_questions = generate_template_quiz(content, topic, difficulty, count)
    if template_questions:
        print(f"  [Quiz TIER 3: Template engine generated {len(template_questions)} questions]")
        return template_questions
    
    # This should NEVER happen, but just in case
    print("  [Quiz: CRITICAL — even template engine failed! Returning minimal fallback]")
    return [{
        "id": 1,
        "question": f"What is {content[:50]} about?" if content else "What is the main topic?",
        "A": "A key concept in this field",
        "B": "An unrelated subject",
        "C": "A mathematical formula",
        "D": "A historical event",
        "correct": "A",
        "explanation": "This is a basic question about the topic. Review your materials for more details.",
        "difficulty": difficulty,
    }]


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
    # auto mode: use NIM for complex tasks only (not simple ones)
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
            ],
            keep_alive=KEEP_ALIVE,
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


def is_fast_model_available() -> bool:
    """Check if the fast model (llama3.2:3b) is installed."""
    try:
        available = get_available_models()
        return any(FAST_MODEL in av for av in available)
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
