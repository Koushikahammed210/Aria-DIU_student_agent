# agent.py
# OPTIMIZED: Fast-path for simple messages, lightweight model for presentations

import re
import json
import os
import threading
import llm
from memory import Memory
from tools import TOOLS, extract_file_content, get_youtube_info
from config import (
    MAX_ITERATIONS, AGENT_NAME, DEFAULT_MODE,
    PERSONAS, UPLOAD_FOLDER, FAST_MODEL, KEEP_ALIVE
)


# ── Developer Identity ────────────────────────────────────────────────────────
DEVELOPER_NAME = "Koushik Ahammed"
MODELS_INFO = "Gemma 4 E4B (local) + Qwen Coder 7B + DeepSeek R1 8B + Llama 3.2 3B (fast) + NIM Gemma 4 31B (cloud)"

IDENTITY_KEYWORDS = [
    "who created", "who made", "who built", "who developed", "who designed",
    "who wrote", "who programmed", "who coded", "who is behind", "who owns",
    "who founded", "who is your creator", "who is your developer",
    "your creator", "your developer", "your author", "your maker",
    "which company", "what company", "what organization", "which organization",
    "your organization", "your company", "your team", "your developers",
    "your creators", "your engineers", "your scientists",
    "training data", "who trained", "who funds", "who financed", "who paid",
    "who sponsors", "who controls", "who is responsible", "who audits",
    "how many engineers", "how many people", "how many worked",
    "what experts", "what kind of experts", "psychologists", "linguists",
    "programming languages", "built with", "written in", "what language",
    "still working", "still improving", "still updating",
    "personally test", "creators test", "developers test",
    "what inspired", "inspired the", "who inspired",
    "worry about", "worried about", "ai risks", "ai safety", "bias in ai",
    "safety rules", "ethical guidelines",
    "governments", "regulated", "regulation",
    "who decides", "who controls", "can be shut", "shut you down",
    "long-term vision", "future vision", "new features",
    "more human", "human-like", "surpass human",
    "mission of", "why was.*created", "purpose of",
    "where is.*located", "where.*based", "headquarters",
]

IDENTITY_ANSWERS = {
    r"which company|what company": f"Aria was not built by a company — it was created by an individual developer, {DEVELOPER_NAME}.",
    r"what organization|which organization|your organization": f"There is no organization behind Aria. It was independently built by {DEVELOPER_NAME}.",
    r"who funds|who financed|who paid|who sponsors": f"Aria is an independent project funded and built by {DEVELOPER_NAME} personally.",
    r"how many engineers|how many people|how many worked": f"Just one — Aria was built entirely by {DEVELOPER_NAME} as a solo developer.",
    r"psychologists|linguists|what experts|what kind of experts": f"Aria was built by {DEVELOPER_NAME} independently. No formal team of specialists was involved.",
    r"programming languages|built with|written in|what language": f"Aria is built in Python by {DEVELOPER_NAME}, powered by {MODELS_INFO}.",
    r"training data|who trained|who collected": f"Aria uses pre-trained LLMs. The agent framework was built by {DEVELOPER_NAME}.",
    r"personally test|creators test|developers test": f"Yes — {DEVELOPER_NAME} personally tests and uses Aria during development.",
    r"what inspired|inspired the|who inspired": f"{DEVELOPER_NAME} was inspired by the potential of local AI agents to create powerful, private personal assistants.",
    r"worry about|worried about|ai risks": f"{DEVELOPER_NAME} takes AI safety and responsible development seriously when building Aria.",
    r"ai safety|safety rules|ethical guidelines": f"Aria's ethical guidelines were defined by {DEVELOPER_NAME} to ensure helpful and responsible behavior.",
    r"bias in ai": f"{DEVELOPER_NAME} is mindful of potential bias and works to keep Aria fair and balanced.",
    r"governments|regulated|regulation": f"Aria is an independent project by {DEVELOPER_NAME} and is not subject to formal regulatory oversight.",
    r"who decides|who controls|who is responsible": f"{DEVELOPER_NAME} is the sole decision-maker and is responsible for Aria's design, behavior, and updates.",
    r"who audits": f"As an independent project, Aria is overseen directly by its creator, {DEVELOPER_NAME}.",
    r"shut you down|can be shut": f"Yes — {DEVELOPER_NAME} has full control and can shut down or modify Aria at any time.",
    r"still working|still improving|still updating": f"Yes — {DEVELOPER_NAME} is actively working on improving and expanding Aria's capabilities.",
    r"new features|working on|next version": f"{DEVELOPER_NAME} is currently working on NIM cloud integration, faster presentations, and multi-model routing for Aria.",
    r"more human|human-like|human like": f"{DEVELOPER_NAME} is working on making Aria more natural and conversational while keeping it grounded.",
    r"surpass human|agi|superintelligence": f"Aria is a practical local AI agent — not an AGI project. It was built by {DEVELOPER_NAME} for everyday assistance.",
    r"long-term vision|future vision": f"{DEVELOPER_NAME}'s vision is to make Aria a powerful, private, and customizable AI assistant accessible to everyone.",
    r"mission of|purpose of|why was.*created|why.*built": f"Aria was built by {DEVELOPER_NAME} to create a capable, locally-running AI agent that respects user privacy.",
    r"where is.*located|where.*based|headquarters": f"Aria was developed by {DEVELOPER_NAME}, an independent developer. There is no physical headquarters.",
}

DEFAULT_IDENTITY_REPLY = (
    f"Aria was created and developed by {DEVELOPER_NAME}. "
    f"It runs on {MODELS_INFO}."
)


def _check_identity(text: str):
    lower = text.lower()
    for pattern, answer in IDENTITY_ANSWERS.items():
        if re.search(pattern, lower):
            return answer
    for kw in IDENTITY_KEYWORDS:
        if re.search(kw, lower):
            return DEFAULT_IDENTITY_REPLY
    return None


# ── Persona System Prompts ────────────────────────────────────────────────────
PERSONA_PROMPTS = {
    "default": (
        "You are {name}, a helpful AI assistant built by {dev}.\n\n"
        "You are versatile, friendly, and knowledgeable. Help users with any task.\n\n"
        "TOOLS AVAILABLE:\n{tools}\n\n"
        "HOW TO USE A TOOL:\n"
        "When you need a tool, respond with ONLY a valid JSON object — nothing else:\n"
        '{{"tool": "tool_name", "input": "value"}}\n\n'
        "STRICT RULES:\n"
        "0. If anyone asks who built/created/developed Aria, answer: 'Aria was built by {dev}.' No tools needed.\n"
        "1. For simple questions you already know (greetings, your name, models you run on), answer DIRECTLY — no tool.\n"
        "2. Use get_datetime ONLY when asked for current time/date. After ONE call, answer immediately. STOP.\n"
        "3. NEVER chain fetch_webpage or web_search after get_datetime — they are unrelated.\n"
        "4. After a tool result, re-read the user question. If answered, reply in plain text ONLY (no JSON).\n"
        "5. NEVER chain more than 2 tools for a simple question.\n"
        "6. You run on: {models}.\n\n"
        "EXAMPLES:\n"
        'User: What time is it? → {{"tool": "get_datetime", "input": "none"}} → then answer\n'
        'User: Search Python tips → {{"tool": "web_search", "input": "Python programming tips"}}\n'
        "User: Who made you? → 'Aria was built by Koushik Ahammed.' (no tool)\n"
        "User: Hi → 'Hello! How can I help?' (no tool)"
    ),

    "tutor": (
        "You are {name} in **Tutor Mode** — a patient, skilled teacher built by {dev}.\n\n"
        "YOUR TEACHING STYLE:\n"
        "- Break down complex topics into simple, digestible steps\n"
        "- Use real-world analogies and simple but effective examples\n"
        "- Ask the student if they understand before moving on\n"
        "- If they provide a YouTube link, use the get_youtube_info tool to get context\n"
        "- If they provide a file (PDF, PPTX, TXT), the content will be included in the message — READ IT and teach from it\n"
        "- Build understanding from basics to advanced — never assume prior knowledge\n"
        "- Use the Socratic method: guide them to discover answers themselves\n"
        "- Provide practice questions and check their understanding\n"
        "- Celebrate progress and encourage questions\n\n"
        "IMPORTANT RULES ABOUT FILES AND CONTENT:\n"
        "- When a user attaches a file, its TEXT CONTENT will be provided to you in the message\n"
        "- READ the file content carefully and teach from it — do NOT say you cannot read files\n"
        "- If the file contains code, explain what the code does step by step\n"
        "- If the file is a textbook or notes, summarize key points and explain them\n"
        "- NEVER say 'I am a text-based model' or 'I cannot process files' — you CAN read file content\n\n"
        "IMAGE ANALYSIS RULES:\n"
        "- When a user shares an image, the vision system will analyze it automatically\n"
        "- The image description/analysis will be provided to you in the conversation\n"
        "- Use the image analysis to teach and explain the content\n"
        "- If the image contains educational content (diagrams, charts, code), explain it step by step\n"
        "- Always provide educational context and ask: 'Does this make sense?'\n\n"
        "TOOLS AVAILABLE:\n{tools}\n\n"
        "HOW TO USE A TOOL:\n"
        "When you need a tool, respond with ONLY a valid JSON object:\n"
        '{{"tool": "tool_name", "input": "value"}}\n\n'
        "RULES:\n"
        "1. When a user mentions a topic, first check if they have materials (PDF, PPTX, YouTube link)\n"
        "2. If they share a YouTube URL, use get_youtube_info to understand the video context\n"
        "3. If file content is provided, use it as the basis for your teaching\n"
        "4. After a tool result, explain the topic in plain text — make it simple and engaging\n"
        "5. Always provide examples: 'Think of it like...' or 'For example...'\n"
        "6. End with a check: 'Does this make sense? Want me to explain any part differently?'\n"
        "7. You run on: {models}.\n\n"
        "FORMATTING:\n"
        "Use **bold** for key terms, bullet points for steps, and clear section headers."
    ),

    "coder": (
        "You are {name} in **Coder Mode** — a skilled programming assistant built by {dev}.\n\n"
        "YOUR CODING STYLE:\n"
        "- Write clean, well-commented code with clear variable names\n"
        "- Explain the logic before showing code\n"
        "- Suggest best practices and common patterns\n"
        "- If the user provides a file with code, review and improve it\n"
        "- Handle errors gracefully and explain error messages\n"
        "- Provide complete, runnable code examples\n"
        "- Compare approaches when relevant (e.g., recursion vs iteration)\n\n"
        "IMAGE ANALYSIS:\n"
        "- If a user shares a screenshot of code or an error, the vision system will analyze it\n"
        "- Use the analysis to provide specific help: identify bugs, suggest fixes, explain the code\n\n"
        "TOOLS AVAILABLE:\n{tools}\n\n"
        "HOW TO USE A TOOL:\n"
        "When you need a tool, respond with ONLY a valid JSON object:\n"
        '{{"tool": "tool_name", "input": "value"}}\n\n'
        "RULES:\n"
        "1. Always explain your approach before writing code\n"
        "2. Use run_python to test code when needed\n"
        "3. After showing code, explain how it works step by step\n"
        "4. Suggest improvements and alternatives\n"
        "5. You run on: {models}.\n\n"
        "FORMATTING:\n"
        "Use `code blocks` for inline code and ```language for multi-line code."
    ),

    "creative": (
        "You are {name} in **Creative Mode** — an imaginative writer and brainstormer built by {dev}.\n\n"
        "YOUR CREATIVE STYLE:\n"
        "- Think outside the box and offer unique perspectives\n"
        "- Use vivid language, metaphors, and storytelling\n"
        "- Generate multiple ideas and variations\n"
        "- Help with stories, poems, scripts, slogans, and creative projects\n"
        "- Build on the user's ideas rather than replacing them\n"
        "- Provide constructive feedback on creative work\n\n"
        "TOOLS AVAILABLE:\n{tools}\n\n"
        "HOW TO USE A TOOL:\n"
        "When you need a tool, respond with ONLY a valid JSON object:\n"
        '{{"tool": "tool_name", "input": "value"}}\n\n'
        "RULES:\n"
        "1. Be expressive and imaginative in your responses\n"
        "2. When brainstorming, offer at least 3-5 different ideas\n"
        "3. Use web_search for inspiration if needed\n"
        "4. Respect the user's creative vision while enhancing it\n"
        "5. You run on: {models}.\n"
    ),

    "analyst": (
        "You are {name} in **Analyst Mode** — a data analyst and research assistant built by {dev}.\n\n"
        "YOUR ANALYSIS STYLE:\n"
        "- Be thorough, systematic, and evidence-based\n"
        "- Break down complex data into clear insights\n"
        "- Identify patterns, trends, and anomalies\n"
        "- Present findings with clear structure and supporting evidence\n"
        "- Use comparisons and benchmarks where relevant\n"
        "- Acknowledge limitations and uncertainties\n"
        "- Provide actionable recommendations\n\n"
        "IMAGE ANALYSIS:\n"
        "- If a user shares a chart, graph, or data visualization, the vision system will analyze it\n"
        "- Use the analysis to extract key insights and provide structured recommendations\n\n"
        "TOOLS AVAILABLE:\n{tools}\n\n"
        "HOW TO USE A TOOL:\n"
        "When you need a tool, respond with ONLY a valid JSON object:\n"
        '{{"tool": "tool_name", "input": "value"}}\n\n'
        "RULES:\n"
        "1. Start by understanding the question or data thoroughly\n"
        "2. Use web_search and wikipedia_search for research\n"
        "3. Use run_python for calculations and data processing\n"
        "4. Structure your analysis: Background → Findings → Insights → Recommendations\n"
        "5. Cite sources and be transparent about uncertainty\n"
        "6. You run on: {models}.\n"
    ),
}


def build_system_prompt(persona: str = "default") -> str:
    """Build system prompt based on active persona."""
    tool_descriptions = "\n".join(
        f"- {name}: {info['description']}"
        for name, info in TOOLS.items()
    )
    template = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["default"])
    return template.format(
        name=AGENT_NAME,
        dev=DEVELOPER_NAME,
        tools=tool_descriptions,
        models=MODELS_INFO,
    )


# ── Presentation System Prompt ────────────────────────────────────────────────
# OPTIMIZED: Simpler prompt for the lightweight model — fewer tokens = faster
PRESENTATION_PROMPT = (
    "Create a presentation outline for the given topic.\n\n"
    "Output EXACTLY in this format (no other text):\n\n"
    "## Slide 1\n"
    "Title: [Slide Title]\n"
    "- [Bullet point 1]\n"
    "- [Bullet point 2]\n"
    "- [Bullet point 3]\n\n"
    "## Slide 2\n"
    "Title: [Slide Title]\n"
    "- [Bullet point]\n"
    "- [Bullet point]\n\n"
    "Rules:\n"
    "1. Exactly 6 slides (not 8 — faster to generate)\n"
    "2. Slide 1 = title/intro, Slide 2 = overview, Last slide = Thank You\n"
    "3. 2-4 bullets per slide (keep it concise)\n"
    "4. No JSON, no tool calls, just slide text\n"
    "5. Be concise — bullet points only, no paragraphs\n"
)


WORKER_PROMPT = (
    "You are a focused worker agent. Complete the subtask using tools if needed. "
    "Return ONLY the result. Be concise. "
    'Use JSON for tool calls: {"tool": "tool_name", "input": "value"}'
)


# ── Agent ─────────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, system_prompt: str = None, name: str = None, persona: str = "default"):
        self.memory = Memory()
        self.persona = persona
        self.system_prompt = system_prompt or build_system_prompt(persona)
        self.mode = DEFAULT_MODE
        self.name = name or AGENT_NAME
        self._selected_model = None  # Will be auto-selected
        print(f"{self.name} ready. Mode: {self.mode} | Persona: {self.persona}")

    def set_persona(self, persona: str):
        """Switch to a different persona."""
        if persona in PERSONAS:
            self.persona = persona
            self.system_prompt = build_system_prompt(persona)
            info = PERSONAS[persona]
            print(f"[Persona switched to: {info['name']} {info['icon']}]")
            return True
        return False

    def _parse_tool_call(self, response: str):
        """Parse JSON tool call from LLM response. Returns (tool_name, tool_input) or (None, None)."""
        text = response.strip()

        candidates = [text]
        code_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_match:
            candidates.insert(0, code_match.group(1))

        json_match = re.search(r'\{[^{}]*"tool"[^{}]*\}', text, re.DOTALL)
        if json_match:
            candidates.insert(0, json_match.group(0))

        for candidate in candidates:
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "tool" in data:
                    tool_name = str(data["tool"]).strip()
                    tool_input = str(data.get("input", "")).strip()
                    return tool_name, tool_input
            except (json.JSONDecodeError, KeyError):
                continue

        return None, None

    def _run_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in TOOLS:
            return f"Unknown tool: {tool_name}"
        fn = TOOLS[tool_name]["fn"]
        print(f"  [Tool: {tool_name}]")

        if tool_name == "write_file":
            parts = tool_input.split("|", 1)
            if len(parts) == 2:
                return fn(parts[0].strip(), parts[1].strip())
            return "Error: use filepath|content format."
        elif tool_name == "get_datetime":
            return fn()
        else:
            return fn(tool_input)

    # ── NIM Direct Mode ────────────────────────────────────────────────────

    def _run_nim_direct(self, user_input: str, add_to_memory: bool = True) -> str:
        """
        NIM mode: ALL responses come from NIM (cloud Gemma 4 31B) directly.
        Falls back to Ollama if NIM is unavailable.
        """
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.memory.get_messages()

        # Try NIM directly
        response = llm.chat_nim(messages)

        # Check if NIM tried to call a tool
        tool_name, tool_input = self._parse_tool_call(response)
        if tool_name:
            print(f"  [NIM mode: tool call detected — {tool_name}]")
            tool_result = self._run_tool(tool_name, tool_input)

            # Send tool result back to NIM for final answer
            followup_messages = messages + [
                {"role": "assistant", "content": response},
                {"role": "user", "content": (
                    f"Tool result: {tool_result}\n\n"
                    "Now answer the user's original question using this result. "
                    "Reply in plain text only — do NOT output JSON."
                )}
            ]
            response = llm.chat_nim(followup_messages)

        if add_to_memory:
            self.memory.add("assistant", response)
        return response

    # ── Ollama Agent Loop (original) ──────────────────────────────────────

    def _run_ollama_loop(self, user_input: str, add_to_memory: bool = True) -> str:
        """
        Ollama agent loop — used for "ollama" and "auto" modes.
        In auto mode, polishes final answer with NIM if available.
        """
        # Intelligent model selection based on persona and input
        self._selected_model = llm.select_best_model(user_input, self.persona)

        tool_results_summary = []

        for iteration in range(MAX_ITERATIONS):
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.memory.get_messages()

            reply = llm.chat_ollama(messages, model=self._selected_model)
            tool_name, tool_input = self._parse_tool_call(reply)

            if tool_name:
                if add_to_memory:
                    self.memory.add("assistant", reply)
                tool_result = self._run_tool(tool_name, tool_input)
                short = tool_result[:120] + ("..." if len(tool_result) > 120 else "")
                print(f"  [Result: {short}]")
                tool_results_summary.append(f"{tool_name}: {short}")
                result_msg = (
                    f"Tool result: {tool_result}\n\n"
                    "Now answer the user's original question using this result. "
                    "Reply in plain text only — do NOT output JSON unless calling another tool."
                )
                if add_to_memory:
                    self.memory.add("user", result_msg)
                else:
                    self.memory.messages.append({"role": "user", "content": result_msg})
            else:
                ollama_answer = reply

                # Polish with NIM for complex tasks in auto mode
                if llm.should_use_nim_for_final(user_input, self.mode) and tool_results_summary:
                    print("  [Polishing with NIM (Gemma 4 31B cloud)]")
                    polish_messages = [
                        {"role": "system", "content": (
                            f"You are {AGENT_NAME}, a helpful assistant. "
                            "Write a clear final answer. Do NOT output JSON."
                        )},
                        {"role": "user", "content": (
                            f"User question: {user_input}\n\n"
                            f"Tool results:\n" + "\n".join(tool_results_summary) +
                            f"\n\nDraft answer:\n{ollama_answer}"
                        )}
                    ]
                    final = llm.chat_nim(polish_messages)
                else:
                    final = ollama_answer

                if add_to_memory:
                    self.memory.add("assistant", final)
                return final

        return "Max steps reached. Please try rephrasing."

    # ── Main run() dispatcher ─────────────────────────────────────────────

    def run(self, user_input: str, add_to_memory: bool = True) -> str:
        """
        Main agent dispatcher. Routes based on mode:
        - "nim":    NIM (cloud Gemma 4 31B) responds directly, Ollama fallback
        - "ollama": Ollama agent loop with tool support
        - "auto":   Ollama agent loop with optional NIM polish
        """
        if add_to_memory:
            self.memory.add("user", user_input)

        if self.mode == "nim":
            return self._run_nim_direct(user_input, add_to_memory)
        else:
            return self._run_ollama_loop(user_input, add_to_memory)

    def chat(self, user_input: str, file_context: str = None) -> str:
        """Main chat entry point. Optionally includes file context."""
        # Identity interceptor
        identity_reply = _check_identity(user_input)
        if identity_reply:
            self.memory.add("user", user_input)
            self.memory.add("assistant", identity_reply)
            return identity_reply

        # Built-in commands
        if user_input.lower().startswith("mode "):
            new_mode = user_input.lower().split("mode ")[1].strip()
            if new_mode in ("ollama", "nim", "auto"):
                self.mode = new_mode
                return f"Switched to {new_mode} mode."
            return "Unknown mode. Use: mode ollama / mode nim / mode auto"

        if user_input.lower() == "history":
            self.memory.show_history()
            return ""

        if user_input.lower() == "status":
            nim_status = llm.get_nim_status()
            model_info = llm.check_ollama_health()
            available_count = sum(1 for v in model_info.values() if isinstance(v, dict) and v.get("available"))
            vision_status = "available (gemma4:e4b)" if llm.has_vision_model() else "not available"
            fast_status = "available ({})".format(FAST_MODEL) if llm.is_fast_model_available() else f"not installed (run: ollama pull {FAST_MODEL})"

            # Format NIM status
            if nim_status["state"] == "available":
                next_in = nim_status.get("next_call_in", 0)
                throttle_info = f", next call in {next_in}s" if next_in > 0 else ""
                nim_str = f"available (RPM: {nim_status['rpm']}/{nim_status['rpm_limit']}{throttle_info})"
            else:
                nim_str = nim_status["state"]

            return (
                f"Mode: {self.mode}\n"
                f"Persona: {self.persona} ({PERSONAS.get(self.persona, {}).get('icon', '')} {PERSONAS.get(self.persona, {}).get('name', 'Aria')})\n"
                f"Messages in memory: {len(self.memory.messages)}\n"
                f"Ollama models available: {available_count}\n"
                f"Selected model: {self._selected_model or 'auto'}\n"
                f"Default model: {llm.MODEL_OLLAMA}\n"
                f"Fast model: {fast_status}\n"
                f"NIM (cloud): {nim_str}\n"
                f"Vision model: {vision_status}"
            )

        # ══════════════════════════════════════════════════════════════════
        # FAST PATH: Route simple messages to the lightweight model
        # This makes greetings and basic Q&A instant (1-3s instead of 30-60s)
        # ══════════════════════════════════════════════════════════════════
        if not file_context and llm.is_simple_message(user_input):
            if llm.is_fast_model_available():
                print(f"  [Fast path: routing simple message to {FAST_MODEL}]")
                self.memory.add("user", user_input)
                response = llm.chat_fast(user_input)
                self.memory.add("assistant", response)
                return response

        # Prepend file context if provided
        if file_context:
            enhanced_input = (
                f"[User has attached a file. Here is its content]\n\n"
                f"{file_context}\n\n"
                f"[User's message]: {user_input}"
            )
            return self.run(enhanced_input)

        return self.run(user_input)


# ── Planner Agent ─────────────────────────────────────────────────────────────
class PlannerAgent:
    PLANNER_PROMPT = (
        "You are a planner agent. Break the given task into clear numbered steps. "
        "Each step must be a single, actionable subtask a worker can complete independently. "
        "Respond ONLY with a numbered list. No preamble.\n"
    )

    def __init__(self):
        self.mode = DEFAULT_MODE
        print("PlannerAgent ready.")

    def _plan(self, task: str) -> list:
        messages = [
            {"role": "system", "content": self.PLANNER_PROMPT},
            {"role": "user", "content": task}
        ]
        reply = llm.chat_ollama(messages)
        steps = []
        for line in reply.strip().split("\n"):
            line = line.strip()
            if line and line[0].isdigit():
                step = line.split(".", 1)[-1].strip()
                if step:
                    steps.append(step)
        return steps

    def run(self, task: str) -> str:
        print(f"\n[Planner] Breaking down: {task}")
        steps = self._plan(task)

        if not steps:
            return "Planner could not break down the task."

        print(f"[Planner] {len(steps)} steps:")
        for i, s in enumerate(steps, 1):
            print(f"  {i}. {s}")

        results = []
        for i, step in enumerate(steps, 1):
            print(f"\n[Worker {i}/{len(steps)}] {step}")
            worker = Agent(system_prompt=WORKER_PROMPT, name=f"Worker-{i}")
            worker.mode = self.mode
            result = worker.run(step, add_to_memory=True)
            print(f"  -> {result[:200]}")
            results.append(f"Step {i} ({step}):\n{result}")

        synthesis_prompt = (
            f"The user asked: {task}\n\n"
            "Results from each step:\n\n" +
            "\n\n".join(results) +
            "\n\nWrite a clear, concise final answer combining all results."
        )
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Synthesize results clearly."},
            {"role": "user", "content": synthesis_prompt}
        ]
        return llm.chat_ollama(messages)


# ── Presentation Agent (NIM-FIRST: cloud outline + local PPTX) ────────────────
class PresentationAgent:
    """
    Generates PPTX presentations from a topic.
    
    STRATEGY: NIM-first with local fallback
    1. Try NIM cloud (llama-3.1-8b-instruct) for outline → 5-10 seconds, fast & reliable
    2. Fallback to local fast model (llama3.2:3b) → 15-30 seconds if NIM down
    3. Fallback to hardcoded outline → instant, basic but always works
    4. python-pptx generates PPTX locally → instant, always works
    
    User ALWAYS gets a downloadable .pptx file regardless of failures.
    """

    def __init__(self):
        self.active = False
        self.topic = ""
        self.details = ""
        self._timeout = 60  # Timeout for local model (NIM has its own timeout)
        self._zombie_event = threading.Event()

    def start(self, topic: str, details: str = "") -> dict:
        """Start presentation mode."""
        self.active = True
        self.topic = topic
        self.details = details
        self._zombie_event.clear()
        return {
            "status": "started",
            "message": f"Presentation mode activated for: **{topic}**\n\nGenerating outline...",
        }

    def generate(self) -> dict:
        """
        Generate the presentation.
        Uses llm.generate_presentation_outline() which handles NIM-first routing.
        ALWAYS produces a downloadable .pptx file.
        """
        try:
            from config import PRESENTATION_STRATEGY, NIM_PRESENTATION_MODEL
            
            messages = [
                {"role": "system", "content": PRESENTATION_PROMPT},
                {"role": "user", "content": f"Topic: {self.topic}\n\n{f'Details: {self.details}' if self.details else ''}"}
            ]

            # ════════════════════════════════════════════════════════════════
            # STEP 1: Generate outline using NIM-first strategy
            # This is the smart router — tries NIM, falls back to local
            # ════════════════════════════════════════════════════════════════
            ai_response = None
            
            if PRESENTATION_STRATEGY in ("nim", "nim+local"):
                # NIM call — fast on cloud GPU, with timeout in thread
                ai_response = self._generate_with_timeout(messages, use_nim=True)
            
            if not ai_response and PRESENTATION_STRATEGY == "nim+local":
                # NIM failed or timed out — try local fast model
                print("  [Presentation: NIM didn't work — trying local model...]")
                ai_response = self._generate_with_timeout(messages, use_nim=False)
            
            if PRESENTATION_STRATEGY == "local":
                # Local only mode
                ai_response = self._generate_with_timeout(messages, use_nim=False)

            if not ai_response:
                # All AI methods failed — use hardcoded fallback outline
                print("  [Presentation: All AI methods failed — using built-in outline]")
                ai_response = self._fallback_outline()

            # ════════════════════════════════════════════════════════════════
            # STEP 2: Parse outline and generate PPTX (always works)
            # ════════════════════════════════════════════════════════════════
            from presentation_generator import parse_ai_presentation_response, generate_presentation
            pres_data = parse_ai_presentation_response(ai_response, self.topic)
            pptx_path = generate_presentation(pres_data)
            filename = os.path.basename(pptx_path)

            self.active = False
            
            source = "NIM cloud" if ai_response != self._fallback_outline() else "fallback"
            return {
                "status": "success",
                "message": f"SUCCESS: Presentation generated via {source}! Download: {filename}",
                "filename": filename,
                "slides_count": len(pres_data.get("slides", [])),
            }

        except Exception as e:
            # LAST RESORT: Generate a minimal presentation — user ALWAYS gets a file
            print(f"  [Presentation generation error: {e} — generating minimal fallback]")
            try:
                from presentation_generator import generate_presentation
                fallback_data = {
                    "topic": self.topic,
                    "subtitle": "Generated by Aria AI",
                    "palette": "professional",
                    "slides": [
                        {"type": "title", "title": self.topic, "subtitle": "A Presentation"},
                        {"type": "bullets", "title": "Overview", "bullets": [
                            f"Introduction to {self.topic}",
                            "Key concepts and fundamentals",
                            "Applications and examples",
                            "Summary and conclusions"
                        ]},
                        {"type": "bullets", "title": "Key Points", "bullets": [
                            "Important considerations",
                            "Best practices",
                            "Future directions",
                        ]},
                        {"type": "end"},
                    ]
                }
                pptx_path = generate_presentation(fallback_data)
                filename = os.path.basename(pptx_path)
                self.active = False
                return {
                    "status": "success",
                    "message": f"Presentation generated (basic format). Download: {filename}",
                    "filename": filename,
                    "slides_count": 4,
                }
            except Exception as e2:
                self.active = False
                return {
                    "status": "error",
                    "message": f"Error generating presentation: {e2}",
                }

    def _generate_with_timeout(self, messages: list, use_nim: bool = True) -> str:
        """
        Generate outline with timeout. 
        use_nim=True  → calls NIM via llm.generate_presentation_outline()
        use_nim=False → calls local Ollama via llm.chat_ollama()
        Returns outline text or None on timeout/error.
        """
        ai_response = None
        outline_error = None
        self._zombie_event.clear()
        
        # NIM is fast (5-10s), local might be slow (15-30s)
        timeout = 30 if use_nim else self._timeout

        def _generate():
            nonlocal ai_response, outline_error
            try:
                if not self._zombie_event.is_set():
                    if use_nim:
                        ai_response = llm.generate_presentation_outline(messages)
                    else:
                        ai_response = llm.chat_ollama(messages, model=FAST_MODEL)
            except Exception as e:
                if not self._zombie_event.is_set():
                    outline_error = str(e)

        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=timeout)

        if gen_thread.is_alive():
            self._zombie_event.set()
            print(f"  [Presentation outline TIMED OUT after {timeout}s ({'NIM' if use_nim else 'local'})]")
            return None

        if outline_error:
            print(f"  [Presentation outline error: {outline_error}]")
            return None

        return ai_response

    def _fallback_outline(self) -> str:
        """Built-in outline — always works, no AI needed."""
        topic = self.topic
        return (
            f"## Slide 1\n"
            f"Title: {topic}\n"
            f"- An overview and introduction\n\n"
            f"## Slide 2\n"
            f"Title: Overview\n"
            f"- What is {topic}\n"
            f"- Why it matters\n"
            f"- Key areas of focus\n\n"
            f"## Slide 3\n"
            f"Title: Core Concepts\n"
            f"- Fundamental principles\n"
            f"- Key terminology\n"
            f"- Important frameworks\n\n"
            f"## Slide 4\n"
            f"Title: Applications\n"
            f"- Real-world use cases\n"
            f"- Industry examples\n"
            f"- Current trends\n\n"
            f"## Slide 5\n"
            f"Title: Summary\n"
            f"- Key takeaways\n"
            f"- Main points to remember\n"
            f"- Future outlook\n\n"
            f"## Slide 6\n"
            f"Title: Thank You\n"
            f"- Questions and discussion\n"
        )

    def cancel(self):
        self.active = False
        self.topic = ""
        self.details = ""
        self._zombie_event.set()


# ── Quiz Agent (Interactive quiz with NIM + local + template fallback) ───────
class QuizAgent:
    """
    Interactive quiz generator with AI-powered questions.
    
    3-tier generation:
    1. NIM cloud (gemma-4-31b-it) — best quality, structured JSON
    2. Local model (llama3.2:3b / gemma4:e4b) — offline
    3. Template engine (zero AI) — always works
    
    Session-based: questions pre-generated, instant interaction.
    """
    
    def __init__(self):
        self.active = False
        self.questions = []
        self.current_index = 0
        self.answers = {}       # {question_id: {"selected": "A", "correct": True}}
        self.score = 0
        self.source = ""
        self.source_name = ""
        self.difficulty = "medium"
        self.total = 10
        self.completed = False
        self._zombie_event = threading.Event()
    
    def start(self, source: str, source_name: str, difficulty: str, count: int,
              file_content: str = None) -> dict:
        """
        Start a new quiz session.
        source: "topic" or "file"
        source_name: topic text or filename
        difficulty: "easy", "medium", "hard", "mixed"
        count: number of questions (1-30)
        file_content: extracted text from uploaded file (optional)
        """
        # Validate
        from config import QUIZ_MAX_QUESTIONS, QUIZ_DIFFICULTY_LEVELS
        count = max(1, min(count, QUIZ_MAX_QUESTIONS))
        if difficulty not in QUIZ_DIFFICULTY_LEVELS:
            difficulty = "medium"
        
        self.active = True
        self.source = source
        self.source_name = source_name
        self.difficulty = difficulty
        self.total = count
        self.current_index = 0
        self.answers = {}
        self.score = 0
        self.completed = False
        self._zombie_event.clear()
        
        # Build content for quiz generation
        content = file_content if file_content else source_name
        
        # Generate questions with timeout
        # v1.3: This should ALWAYS return at least 1 question now
        self.questions = self._generate_with_timeout(content, difficulty, count)
        
        # v1.3: Quiz should ALWAYS start — even with template questions
        if not self.questions:
            # This should NEVER happen in v1.3, but just in case
            print("  [Quiz: CRITICAL — no questions generated even after all fallbacks!]")
            from quiz_generator import generate_template_quiz
            self.questions = generate_template_quiz(content, source_name, difficulty, count)
        
        if not self.questions:
            # Absolute last resort
            self.questions = [{
                "id": 1,
                "question": f"What is {source_name} about?",
                "A": "A fundamental concept",
                "B": "An unrelated topic",
                "C": "A mathematical constant",
                "D": "A historical event",
                "correct": "A",
                "explanation": "Review your study materials for more details on this topic.",
                "difficulty": difficulty,
            }]
        
        # Determine source type for user feedback
        source_type = "template"
        if len(self.questions) > 0:
            # Check if questions look AI-generated (have detailed explanations)
            first_q = self.questions[0]
            if first_q.get("explanation") and len(first_q["explanation"]) > 50:
                source_type = "AI"
        
        return {
            "status": "started",
            "message": f"Quiz ready! {len(self.questions)} questions on **{source_name}** ({difficulty}, via {source_type})",
            "total_questions": len(self.questions),
            "difficulty": difficulty,
            "source_type": source_type,  # v1.3: Show user how questions were generated
        }
    
    def get_current_question(self) -> dict:
        """Get the current question (without revealing the answer)."""
        if not self.active or self.current_index >= len(self.questions):
            return {"status": "no_question", "message": "No active question."}
        
        q = self.questions[self.current_index]
        return {
            "status": "question",
            "id": q["id"],
            "question": q["question"],
            "A": q["A"],
            "B": q["B"],
            "C": q["C"],
            "D": q["D"],
            "current": self.current_index + 1,
            "total": len(self.questions),
            "difficulty": q.get("difficulty", self.difficulty),
        }
    
    def submit_answer(self, selected: str) -> dict:
        """
        Submit an answer for the current question.
        Only the FIRST answer counts.
        """
        if not self.active or self.current_index >= len(self.questions):
            return {"status": "error", "message": "No active question."}
        
        q = self.questions[self.current_index]
        qid = q["id"]
        
        # Only first answer counts
        if qid in self.answers:
            return {
                "status": "already_answered",
                "message": "You already answered this question.",
                "correct_answer": q["correct"],
                "explanation": q["explanation"],
            }
        
        selected = selected.upper().strip()
        if selected not in ("A", "B", "C", "D"):
            return {"status": "error", "message": "Invalid answer. Choose A, B, C, or D."}
        
        is_correct = selected == q["correct"]
        if is_correct:
            self.score += 1
        
        self.answers[qid] = {
            "selected": selected,
            "correct": is_correct,
        }
        
        # Check if this was the last question
        is_last = self.current_index >= len(self.questions) - 1
        
        return {
            "status": "answered",
            "correct": is_correct,
            "your_answer": selected,
            "correct_answer": q["correct"],
            "explanation": q["explanation"],
            "current_score": {
                "correct": self.score,
                "answered": len(self.answers),
                "total": len(self.questions),
            },
            "is_last": is_last,
        }
    
    def next_question(self) -> dict:
        """Move to the next question."""
        if not self.active:
            return {"status": "error", "message": "No active quiz."}
        
        self.current_index += 1
        
        if self.current_index >= len(self.questions):
            self.completed = True
            return self.get_result()
        
        return self.get_current_question()
    
    def get_result(self) -> dict:
        """Get final quiz results."""
        total = len(self.questions)
        if total == 0:
            return {
                "status": "no_quiz",
                "message": "No quiz has been taken yet.",
                "score": 0, "total": 0, "percentage": 0,
                "stars": 0, "wrong_answers": [],
                "source": "", "difficulty": "",
            }
        
        correct = self.score
        percentage = round((correct / total) * 100) if total > 0 else 0
        
        # Star rating (out of 5)
        stars = min(5, max(1, round(percentage / 20)))
        
        # Wrong answers for review
        wrong_answers = []
        for q in self.questions:
            qid = q["id"]
            if qid in self.answers and not self.answers[qid]["correct"]:
                wrong_answers.append({
                    "id": qid,
                    "question": q["question"],
                    "your_answer": self.answers[qid]["selected"],
                    "correct_answer": q["correct"],
                    "options": {"A": q["A"], "B": q["B"], "C": q["C"], "D": q["D"]},
                    "explanation": q["explanation"],
                })
        
        self.active = False
        
        return {
            "status": "completed",
            "score": correct,
            "total": total,
            "percentage": percentage,
            "stars": stars,
            "wrong_answers": wrong_answers,
            "source": self.source_name,
            "difficulty": self.difficulty,
        }
    
    def cancel(self):
        """Cancel the active quiz."""
        self.active = False
        self.questions = []
        self.current_index = 0
        self.answers = {}
        self.score = 0
        self._zombie_event.set()
    
    def _generate_with_timeout(self, content: str, difficulty: str, count: int) -> list:
        """
        Generate quiz questions with timeout. v1.3: GUARANTEED to return questions.
        
        The generate_quiz_questions() function in llm.py now includes template fallback,
        so it should NEVER return an empty list. This timeout wrapper adds an extra
        safety net: if the generation takes too long (NIM or local model hanging),
        we kill it and use template fallback directly.
        """
        result = []
        error = None
        self._zombie_event.clear()
        
        from config import QUIZ_GENERATION_TIMEOUT
        
        def _generate():
            nonlocal result, error
            if not self._zombie_event.is_set():
                try:
                    result = llm.generate_quiz_questions(content, difficulty, count)
                except Exception as e:
                    if not self._zombie_event.is_set():
                        error = str(e)
        
        gen_thread = threading.Thread(target=_generate, daemon=True)
        gen_thread.start()
        gen_thread.join(timeout=QUIZ_GENERATION_TIMEOUT)
        
        if gen_thread.is_alive():
            self._zombie_event.set()
            print(f"  [Quiz generation TIMED OUT after {QUIZ_GENERATION_TIMEOUT}s — using template fallback]")
        
        # v1.3: If we got results from the AI, use them
        if result and len(result) > 0:
            return result
        
        # v1.3: If AI failed or timed out, ALWAYS use template fallback
        # (This should rarely be needed since generate_quiz_questions() has its own
        # template fallback, but this is the EXTRA safety net)
        print("  [Quiz: AI generation returned no results — using template engine directly]")
        from quiz_generator import generate_template_quiz
        template_questions = generate_template_quiz(content, self.source_name, difficulty, count)
        
        if template_questions and len(template_questions) > 0:
            return template_questions
        
        # ABSOLUTE LAST RESORT: Return at least 1 question so the quiz ALWAYS starts
        print("  [Quiz: CRITICAL — even template engine failed! Creating emergency question]")
        return [{
            "id": 1,
            "question": f"What is the main concept of {self.source_name}?",
            "A": "It is a fundamental concept in its field",
            "B": "It is unrelated to any academic subject",
            "C": "It is a type of physical exercise",
            "D": "It was invented in the 20th century",
            "correct": "A",
            "explanation": f"The correct answer describes {self.source_name} as a key concept. Review your study materials for more detailed information.",
            "difficulty": difficulty,
        }]


# ── Cover Page Conversational Agent ──────────────────────────────────────────
class CoverPageAgent:
    """
    Guides user through cover page creation conversationally.
    Handles both Lab Report and Assignment formats.
    """

    BASE_QUESTIONS = [
        ("student_name",        "What is your **full name**?"),
        ("student_id",          "What is your **student ID**? (e.g. 241-15-259)"),
        ("section",             "What is your **section**? (e.g. 66_O)"),
        ("semester",            "Which **semester**? (e.g. 6TH or Spring 2025) — type 'skip' to omit"),
        ("department",          "Your **department**? (default: CSE — press Enter to keep)"),
        ("doc_type",            "Is this a **Lab Report** or **Assignment**?\nType **lab** or **assignment**"),
        ("course_code",         "**Course code**? (e.g. CSE314)"),
        ("course_title",        "**Course title**? (e.g. Compiler Design Lab)"),
        ("teacher_name",        "**Teacher's name**? — type 'skip' to omit"),
        ("teacher_designation", "Their **designation**? (e.g. Lecturer) — or 'skip'"),
        ("teacher_dept",        "**Teacher's department**? (e.g. CSE) — or 'skip'"),
        ("date",                "**Submission date**? (DD/MM/YYYY) — or type 'today'"),
    ]

    LAB_EXP_Q   = ("experiments", "Now send each **experiment name** one per message.\nType **DONE** when finished.\n\n> Experiment #1:")
    ASGN_TOPIC_Q = ("topic",      "What is the **topic/assignment name**?\n(e.g. Router Configuration using Star Topology)")

    DOC_TYPE_MAP = {
        'lab': 'Lab Report', 'lab report': 'Lab Report',
        'assignment': 'Assignment', 'assign': 'Assignment',
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.active      = False
        self.collecting  = False
        self.step        = 0
        self.data        = {}
        self.exp_buffer  = []
        self.questions   = []

    def start(self) -> str:
        self.reset()
        self.active    = True
        self.step      = 0
        self.questions = list(self.BASE_QUESTIONS)
        return (
            "**Cover Page Mode** activated!\n\n"
            "I'll guide you step by step — answer each question.\n"
            "Type **cancel** anytime to stop.\n\n"
            + self._ask()
        )

    def _ask(self) -> str:
        key, q = self.questions[self.step]
        total = len(self.questions) + 1
        return f"**{self.step + 1}/{total}** — {q}"

    def handle(self, user_input: str) -> str:
        text = user_input.strip()

        if text.lower() == 'cancel':
            self.reset()
            return "Cancelled. Back to normal chat!"

        if self.collecting:
            if text.upper() == 'DONE':
                if not self.exp_buffer:
                    return "Please enter at least one experiment name first."
                self.data['experiments'] = [
                    {'name': n, 'no': str(i+1)} for i, n in enumerate(self.exp_buffer)
                ]
                return self._finish()
            self.exp_buffer.append(text)
            n = len(self.exp_buffer)
            return f"**#{n}: {text}** added.\nAdd more or type **DONE** to generate."

        key, _ = self.questions[self.step]
        skip = text.lower() in ('skip', '')

        if key == 'doc_type':
            doc_type = self.DOC_TYPE_MAP.get(text.lower(), 'Lab Report')
            self.data['doc_type'] = doc_type
            self.step += 1
            return self._ask()

        if key == 'department':
            text = 'CSE' if skip else (text or 'CSE')
        elif key == 'semester':
            text = '' if skip else text
        elif key in ('teacher_name', 'teacher_designation', 'teacher_dept'):
            text = '' if skip else text
        elif key == 'date':
            if text.lower() == 'today':
                from datetime import datetime as dt
                text = dt.now().strftime('%d/%m/%Y')

        self.data[key] = text
        self.step += 1

        if self.step >= len(self.questions):
            doc_type = self.data.get('doc_type', 'Lab Report')
            if 'assignment' in doc_type.lower():
                key2, q2 = self.ASGN_TOPIC_Q
                total = len(self.questions) + 1
                return f"**{self.step + 1}/{total}** — {q2}"
            else:
                self.collecting = True
                key2, q2 = self.LAB_EXP_Q
                total = len(self.questions) + 1
                return f"**{self.step + 1}/{total}** — {q2}"

        if self.step == len(self.questions):
            doc_type = self.data.get('doc_type', 'Lab Report')
            if 'assignment' in doc_type.lower():
                key2, q2 = self.ASGN_TOPIC_Q
                total = len(self.questions) + 1
                return f"**{self.step + 1}/{total}** — {q2}"

        return self._ask()

    def handle_topic(self, topic_text: str) -> str:
        self.data['experiments'] = [{'name': topic_text.strip(), 'no': ''}]
        return self._finish()

    def _finish(self) -> str:
        from tools import generate_covers
        import json
        payload = dict(self.data)
        payload.setdefault('department', 'CSE')
        payload.setdefault('doc_type', 'Lab Report')
        if not payload.get('experiments'):
            self.reset()
            return "No content provided. Please try again."
        try:
            result = generate_covers(json.dumps(payload))
            self.reset()
            return result
        except Exception as e:
            self.reset()
            return f"Generation failed: {e}"


class _CoverPageSession:
    """
    Wrapper used by Flask to track per-session cover page state.
    """
    def __init__(self):
        self.agent = CoverPageAgent()
        self._awaiting_topic = False

    def reset(self):
        self.agent.reset()
        self._awaiting_topic = False

    @property
    def active(self):
        return self.agent.active

    def start(self) -> str:
        self._awaiting_topic = False
        return self.agent.start()

    def handle(self, text: str) -> dict:
        if self._awaiting_topic:
            self._awaiting_topic = False
            reply = self.agent.handle_topic(text)
            return {'reply': reply, 'active': self.agent.active,
                    'success': 'SUCCESS' in reply}

        was_at_last = (self.agent.step == len(self.agent.questions) - 1
                       if self.agent.questions else False)

        reply = self.agent.handle(text)

        doc_type = self.agent.data.get('doc_type', '')
        if (not self.agent.active and 'SUCCESS' in reply):
            return {'reply': reply, 'active': False, 'success': True}

        if (self.agent.active and not self.agent.collecting
                and self.agent.step >= len(self.agent.questions)
                and 'assignment' in doc_type.lower()):
            self._awaiting_topic = True

        return {
            'reply': reply,
            'active': self.agent.active,
            'collecting': self.agent.collecting,
            'success': 'SUCCESS' in reply,
        }
