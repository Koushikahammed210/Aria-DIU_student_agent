# run.py
# OPTIMIZED: Model pre-warming on startup, fast model support

import sys
import os
from agent import Agent, PlannerAgent, _CoverPageSession, PresentationAgent, QuizAgent
from config import AGENT_NAME, UPLOAD_FOLDER, ALLOWED_FILE_EXTENSIONS, PERSONAS, OLLAMA_MODELS, FAST_MODEL, ARIA_VERSION


# ── Ensure upload folder exists ───────────────────────────────────────────────
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def run_cli():
    print("=" * 52)
    print(f"   {AGENT_NAME} — Local AI Agent  [CLI]")
    print("=" * 52)
    agent = Agent()
    planner = PlannerAgent()
    cover = _CoverPageSession()
    pres_agent = PresentationAgent()
    quiz_agent = QuizAgent()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except KeyboardInterrupt:
            print("\nGoodbye!"); break
        if not user_input: continue
        if user_input.lower() in ("quit","exit","bye"): print("Goodbye!"); break
        if user_input.lower() == "clear": agent.memory.clear(); continue
        if user_input.lower().startswith("planner "):
            print(f"\n{AGENT_NAME}:", planner.run(user_input[8:].strip())); continue
        if user_input.lower() in ("cover","cover page"):
            print(f"\n{AGENT_NAME}:", cover.start()); continue
        if cover.active:
            r = cover.handle(user_input)
            print(f"\n{AGENT_NAME}:", r['reply']); continue
        if user_input.lower().startswith("persona "):
            p = user_input.lower().split("persona ")[1].strip()
            if agent.set_persona(p):
                info = PERSONAS[p]
                print(f"\n{AGENT_NAME}: Switched to {info['icon']} {info['name']} persona!")
            else:
                print(f"\n{AGENT_NAME}: Unknown persona. Available: {', '.join(PERSONAS.keys())}")
            continue
        if user_input.lower().startswith("present "):
            topic = user_input[8:].strip()
            result = pres_agent.start(topic)
            print(f"\n{AGENT_NAME}:", result['message'])
            if pres_agent.active:
                gen = pres_agent.generate()
                if gen['status'] == 'success':
                    print(f"\n{AGENT_NAME}: {gen['message']}")
                else:
                    print(f"\n{AGENT_NAME}: {gen['message']}")
            continue
        if user_input.lower().startswith("quiz "):
            parts = user_input[5:].strip()
            if not parts:
                print(f"\n{AGENT_NAME}: Usage: quiz <topic> [easy|medium|hard|mixed] [count]")
                continue
            args = parts.split()
            topic = args[0]
            diff = "medium"
            count = 10
            for a in args[1:]:
                if a.lower() in ("easy", "medium", "hard", "mixed"):
                    diff = a.lower()
                elif a.isdigit():
                    count = min(int(a), 30)
            print(f"\n{AGENT_NAME}: Generating {count} {diff} questions about {topic}...")
            result = quiz_agent.start("topic", topic, diff, count)
            if result["status"] == "error":
                print(f"\n{AGENT_NAME}: {result['message']}")
                continue
            print(f"\n{AGENT_NAME}: {result['message']}")
            while quiz_agent.active:
                q = quiz_agent.get_current_question()
                if q.get("status") != "question":
                    break
                print(f"\n  Q{q['current']}/{q['total']} ({q.get('difficulty','')}): {q['question']}")
                print(f"    A) {q['A']}")
                print(f"    B) {q['B']}")
                print(f"    C) {q['C']}")
                print(f"    D) {q['D']}")
                ans = input("  Your answer: ").strip().upper()
                if ans in ("QUIT", "EXIT", "CANCEL"):
                    quiz_agent.cancel()
                    print(f"\n{AGENT_NAME}: Quiz cancelled.")
                    break
                if ans not in ("A", "B", "C", "D"):
                    print("  Please enter A, B, C, or D.")
                    continue
                res = quiz_agent.submit_answer(ans)
                if res.get("correct"):
                    print(f"  ✅ Correct!")
                else:
                    print(f"  ❌ Wrong! Correct answer: {res['correct_answer']}")
                print(f"  📖 {res['explanation']}")
                if res.get("is_last"):
                    r = quiz_agent.get_result()
                    print(f"\n{AGENT_NAME}: Quiz Complete! Score: {r['score']}/{r['total']} ({r['percentage']}%) {'⭐' * r['stars']}")
                else:
                    quiz_agent.next_question()
            continue
        if user_input.lower().startswith("model "):
            model_name = user_input.lower().split("model ")[1].strip()
            # Map short names to full model names
            model_map = {
                "gemma4": "gemma4:e4b",
                "gemma": "gemma4:e4b",
                "coder": "qwen2.5-coder:7b",
                "qwen-coder": "qwen2.5-coder:7b",
                "deepseek": "deepseek-r1:8b",
                "r1": "deepseek-r1:8b",
                "fast": FAST_MODEL,
                "llama": FAST_MODEL,
                "llama3": FAST_MODEL,
            }
            full_model = model_map.get(model_name, model_name)
            if full_model in OLLAMA_MODELS:
                agent._selected_model = full_model
                print(f"\n{AGENT_NAME}: Model set to {full_model}")
            else:
                print(f"\n{AGENT_NAME}: Unknown model. Available: {', '.join(OLLAMA_MODELS.keys())}")
            continue
        print(f"\n{AGENT_NAME}:", agent.chat(user_input))


def run_web():
    from flask import Flask, request, jsonify, send_from_directory, send_file
    import llm as llm_mod
    from tools import extract_file_content

    app = Flask(__name__, static_folder="static")
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload

    agent  = Agent()
    planner = PlannerAgent()
    cover  = _CoverPageSession()
    pres_agent = PresentationAgent()
    quiz_agent = QuizAgent()

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    # ── Chat Endpoint (enhanced with file + vision + mode-aware routing) ────
    @app.route("/chat", methods=["POST"])
    def chat():
        data        = request.json
        user_input  = data.get("message", "").strip()
        use_planner = data.get("planner", False)
        images      = data.get("images", [])   # [{base64, mime}]
        file_context = data.get("file_context", None)
        persona     = data.get("persona", None)

        # Switch persona if requested
        if persona and persona != agent.persona:
            agent.set_persona(persona)

        if not user_input and not images and not file_context:
            return jsonify({"reply": "Please enter a message."})

        # If images are attached WITHOUT file context, use vision analysis
        if images and not file_context:
            vision_prompt = user_input or "Describe this image in detail."
            if persona == "tutor":
                vision_prompt = (
                    (user_input + "\n\n" if user_input else "") +
                    "Analyze this image in an educational way. If it contains diagrams, charts, "
                    "or educational content, explain them step by step. If it's code, explain what it does. "
                    "Ask if the student understands and offer to explain further."
                )
            elif persona == "coder":
                vision_prompt = (
                    (user_input + "\n\n" if user_input else "") +
                    "Analyze this image with a focus on technical content. If it contains code, "
                    "explain what it does, identify any issues, and suggest improvements."
                )
            elif persona == "analyst":
                vision_prompt = (
                    (user_input + "\n\n" if user_input else "") +
                    "Analyze this image systematically. If it contains data, charts, or graphs, "
                    "extract the key insights and provide a structured analysis."
                )
            # Mode-aware vision routing
            reply = llm_mod.chat_with_images(vision_prompt, images, mode=agent.mode)
            return jsonify({"reply": reply})

        if use_planner:
            reply = planner.run(user_input)
        else:
            reply = agent.chat(user_input, file_context=file_context)
        return jsonify({"reply": reply})

    # ── File Upload Endpoint ─────────────────────────────────────────────────
    @app.route("/upload", methods=["POST"])
    def upload_file():
        """Handle file uploads and extract text content."""
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if not file.filename:
            return jsonify({"error": "No filename"}), 400

        ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
        if ext not in ALLOWED_FILE_EXTENSIONS:
            return jsonify({"error": f"File type .{ext} not allowed. Allowed: {', '.join(sorted(ALLOWED_FILE_EXTENSIONS))}"}), 400

        safe_name = os.path.basename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, safe_name)

        counter = 1
        base, ext_with_dot = os.path.splitext(filepath)
        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{ext_with_dot}"
            counter += 1

        try:
            file.save(filepath)
        except Exception as e:
            return jsonify({"error": f"Failed to save file: {e}"}), 500

        extracted_text = ""
        if ext in ("pdf", "txt", "pptx", "docx", "csv", "json", "py", "js", "html", "css", "java", "cpp", "c", "md"):
            extracted_text = extract_file_content(filepath)

        file_size = os.path.getsize(filepath)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024*1024):.1f} MB"

        return jsonify({
            "success": True,
            "filename": os.path.basename(filepath),
            "size": size_str,
            "type": ext,
            "extracted_text": extracted_text[:8000] if extracted_text else None,
            "has_content": bool(extracted_text),
        })

    # ── Persona Endpoints ────────────────────────────────────────────────────
    @app.route("/personas", methods=["GET"])
    def get_personas():
        """Return available personas."""
        return jsonify({
            "personas": {k: v for k, v in PERSONAS.items()},
            "current": agent.persona,
        })

    @app.route("/persona", methods=["POST"])
    def set_persona():
        """Switch to a different persona."""
        persona = request.json.get("persona", "default")
        if agent.set_persona(persona):
            info = PERSONAS[persona]
            return jsonify({
                "status": "success",
                "persona": persona,
                "name": info["name"],
                "icon": info["icon"],
                "description": info["description"],
            })
        return jsonify({"status": "error", "message": f"Unknown persona: {persona}"}), 400

    # ── Presentation Endpoints ───────────────────────────────────────────────
    @app.route("/presentation", methods=["POST"])
    def create_presentation():
        """Generate a PPTX presentation from a topic."""
        data = request.json
        topic = data.get("topic", "").strip()
        details = data.get("details", "").strip()

        if not topic:
            return jsonify({"reply": "Please provide a topic for the presentation."})

        pres_agent.start(topic, details)
        result = pres_agent.generate()

        if result["status"] == "success":
            filename = result.get("filename", "")
            reply = result["message"]
            return jsonify({
                "reply": reply,
                "filename": filename,
                "slides_count": result.get("slides_count", 0),
                "download_url": f"/download/{filename}",
            })
        else:
            return jsonify({"reply": result["message"]})

    # ── Quiz Endpoints ───────────────────────────────────────────────────────
    @app.route("/quiz/start", methods=["POST"])
    def start_quiz():
        """Start a new quiz session. v1.3: ALWAYS returns questions (never errors)."""
        data = request.json
        source = data.get("source", "topic")  # "topic" or "file"
        topic = data.get("topic", "").strip()
        difficulty = data.get("difficulty", "medium")
        count = data.get("count", 10)
        file_content = data.get("file_content", None)
        
        if not topic and not file_content:
            return jsonify({"status": "error", "message": "Please provide a topic or upload a file."})
        
        source_name = topic if topic else "Uploaded File"
        
        # v1.3: Log what's happening
        print(f"  [Quiz /start] source={source}, topic={source_name}, difficulty={difficulty}, count={count}")
        print(f"  [Quiz /start] file_content={'provided' if file_content else 'none'} ({len(file_content) if file_content else 0} chars)")
        
        result = quiz_agent.start(source, source_name, difficulty, count, file_content)
        
        # v1.3: Log result
        if result.get("status") == "started":
            print(f"  [Quiz /start] SUCCESS: {result['total_questions']} questions via {result.get('source_type', 'unknown')}")
        else:
            print(f"  [Quiz /start] Result: {result.get('status')} - {result.get('message', '')}")
        
        return jsonify(result)
    
    @app.route("/quiz/question", methods=["GET"])
    def get_quiz_question():
        """Get current quiz question."""
        result = quiz_agent.get_current_question()
        return jsonify(result)
    
    @app.route("/quiz/answer", methods=["POST"])
    def submit_quiz_answer():
        """Submit answer for current question."""
        data = request.json
        selected = data.get("answer", "").strip().upper()
        result = quiz_agent.submit_answer(selected)
        return jsonify(result)
    
    @app.route("/quiz/next", methods=["POST"])
    def next_quiz_question():
        """Move to next question."""
        result = quiz_agent.next_question()
        return jsonify(result)
    
    @app.route("/quiz/result", methods=["GET"])
    def get_quiz_result():
        """Get quiz results."""
        result = quiz_agent.get_result()
        return jsonify(result)
    
    @app.route("/quiz/cancel", methods=["POST"])
    def cancel_quiz():
        """Cancel active quiz."""
        quiz_agent.cancel()
        return jsonify({"status": "cancelled", "message": "Quiz cancelled."})

    # ── Cover Page Endpoints ─────────────────────────────────────────────────
    @app.route("/cover-chat", methods=["POST"])
    def cover_chat():
        data   = request.json
        action = data.get("action", "message")
        if action == "start":
            cover.reset()
            return jsonify({"reply": cover.start(), "active": cover.active})
        if action == "cancel":
            cover.reset()
            return jsonify({"reply": "Cover page creation cancelled.", "active": False})
        msg = data.get("message", "").strip()
        if not msg:
            return jsonify({"reply": "Please enter a response.", "active": cover.active})
        result = cover.handle(msg)
        return jsonify(result)

    @app.route("/cover", methods=["POST"])
    def cover_direct():
        from tools import generate_covers
        import json
        data = request.json
        if not data:
            return jsonify({"reply": "No data received."})
        try:
            result = generate_covers(json.dumps(data))
            return jsonify({"reply": result})
        except Exception as e:
            return jsonify({"reply": f"Error: {e}"})

    # ── Model/Status Endpoints ───────────────────────────────────────────────
    @app.route("/clear", methods=["POST"])
    def clear():
        agent.memory.clear()
        return jsonify({"status": "Memory cleared."})

    @app.route("/status", methods=["GET"])
    def status():
        """Enhanced status with NIM + fast model details."""
        nim_status = llm_mod.get_nim_status()
        model_info = llm_mod.check_ollama_health()
        available_models = [k for k, v in model_info.items() if isinstance(v, dict) and v.get("available")]
        vision_available = llm_mod.has_vision_model()
        fast_available = llm_mod.is_fast_model_available()

        return jsonify({
            "mode": agent.mode,
            "persona": agent.persona,
            "persona_info": PERSONAS.get(agent.persona, {}),
            "messages": len(agent.memory.messages),
            "models_available": available_models,
            "selected_model": agent._selected_model or "auto",
            "fast_model_available": fast_available,
            "fast_model": FAST_MODEL,
            "vision_available": vision_available,
            "nim": nim_status,
        })

    @app.route("/mode", methods=["POST"])
    def set_mode():
        mode = request.json.get("mode", "auto")
        if mode in ("ollama", "nim", "auto"):
            agent.mode = mode
            planner.mode = mode
            return jsonify({"status": f"Switched to {mode} mode."})
        return jsonify({"status": "Unknown mode. Use: ollama, nim, or auto"}), 400

    @app.route("/models", methods=["GET"])
    def get_models():
        """Return available model information."""
        health = llm_mod.check_ollama_health()
        nim_status = llm_mod.get_nim_status()
        return jsonify({
            "models": health,
            "nim": nim_status,
            "fast_model": FAST_MODEL,
            "fast_model_available": llm_mod.is_fast_model_available(),
            "vision_available": llm_mod.has_vision_model(),
            "current_model": agent._selected_model or "auto",
        })

    @app.route("/models/select", methods=["POST"])
    def select_model():
        """Let user manually select an Ollama model."""
        model_name = request.json.get("model", "").strip()
        # Map short names to full model names
        model_map = {
            "gemma4": "gemma4:e4b",
            "gemma": "gemma4:e4b",
            "coder": "qwen2.5-coder:7b",
            "qwen-coder": "qwen2.5-coder:7b",
            "deepseek": "deepseek-r1:8b",
            "r1": "deepseek-r1:8b",
            "fast": FAST_MODEL,
            "llama": FAST_MODEL,
            "llama3": FAST_MODEL,
        }
        full_model = model_map.get(model_name, model_name)
        if full_model in OLLAMA_MODELS:
            agent._selected_model = full_model
            return jsonify({
                "status": "success",
                "model": full_model,
                "info": OLLAMA_MODELS[full_model],
            })
        return jsonify({
            "status": "error",
            "message": f"Unknown model: {model_name}. Available: {', '.join(OLLAMA_MODELS.keys())}"
        }), 400

    # ── Download Endpoint ────────────────────────────────────────────────────
    @app.route("/download/<filename>")
    def download(filename):
        safe = os.path.basename(filename)
        for search_dir in [os.getcwd(), UPLOAD_FOLDER]:
            fp = os.path.join(search_dir, safe)
            if os.path.exists(fp):
                try:
                    return send_file(fp, as_attachment=True, download_name=safe)
                except Exception as e:
                    return jsonify({"error": str(e)}), 500
        return jsonify({"error": f"File not found: {safe}"}), 404

    # ── YouTube Info Endpoint ────────────────────────────────────────────────
    @app.route("/youtube-info", methods=["POST"])
    def youtube_info():
        """Get YouTube video info for tutor mode."""
        from tools import get_youtube_info
        url = request.json.get("url", "").strip()
        if not url or "youtube" not in url.lower():
            return jsonify({"error": "Please provide a valid YouTube URL"})
        info = get_youtube_info(url)
        return jsonify({"info": info})

    # ── Vision Diagnostic Endpoint ──────────────────────────────────────────
    @app.route("/vision-test", methods=["GET"])
    def vision_test():
        """Full diagnostic of vision model detection."""
        diag = {
            "models": llm_mod.get_available_models(),
            "vision_available": llm_mod.has_vision_model(),
            "fast_model_available": llm_mod.is_fast_model_available(),
            "nim_status": llm_mod.get_nim_status(),
        }
        return jsonify(diag)

    # ── Startup Diagnostics ─────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"   {AGENT_NAME} v{ARIA_VERSION} — AI Agent")
    print(f"{'='*52}")
    print(f"   URL: http://localhost:5000")
    print(f"   Mode: {agent.mode}")

    # Ollama models
    try:
        health = llm_mod.check_ollama_health()
        available = [k for k, v in health.items() if isinstance(v, dict) and v.get("available")]
        print(f"   Ollama models: {', '.join(available) if available else 'none detected'}")
    except:
        print("   Ollama models: error detecting")

    # Fast model check
    try:
        fast_available = llm_mod.is_fast_model_available()
        if fast_available:
            print(f"   Fast model: {FAST_MODEL} — ready for instant responses!")
        else:
            print(f"   Fast model: NOT INSTALLED — run 'ollama pull {FAST_MODEL}' for instant responses")
    except Exception as e:
        print(f"   Fast model check error: {e}")

    # Vision model (Gemma 4 E4B is multimodal)
    try:
        vision_available = llm_mod.has_vision_model()
        if vision_available:
            print("   Vision: gemma4:e4b (multimodal) — image analysis ready!")
        else:
            print("   Vision: gemma4:e4b not found — run 'ollama pull gemma4:e4b'")
    except Exception as e:
        print(f"   Vision check error: {e}")

    # NIM (cloud)
    from config import PRESENTATION_STRATEGY, NIM_PRESENTATION_MODEL, QUIZ_GENERATION_MODEL_NIM
    nim_status = llm_mod.get_nim_status()
    if nim_status["state"] == "available":
        failed_models_info = ""
        if nim_status.get("failed_models"):
            failed = [f"{m} ({c}x)" for m, c in nim_status["failed_models"].items() if c > 0]
            if failed:
                failed_models_info = f"\n     - Failed models: {', '.join(failed)}"
        print(f"   NIM (cloud): available (RPM: {nim_status['rpm']}/{nim_status['rpm_limit']})")
        print(f"     - Default:  {nim_status['model']} (vision/complex tasks)")
        print(f"     - Presentation: {NIM_PRESENTATION_MODEL} (slide outlines)")
        print(f"     - Quiz: {QUIZ_GENERATION_MODEL_NIM} (question generation)")
        if failed_models_info:
            print(failed_models_info)
    else:
        print(f"   NIM (cloud): {nim_status['state']}")
        if nim_status.get("last_error"):
            print(f"     Error: {nim_status['last_error']}")
        if nim_status.get("failed_models"):
            failed = [f"{m} ({c}x)" for m, c in nim_status["failed_models"].items() if c > 0]
            if failed:
                print(f"     Failed models: {', '.join(failed)}")

    # Presentation strategy
    print(f"   Presentation: {PRESENTATION_STRATEGY} mode")
    print(f"   Quiz: v1.4 GUARANTEED (1-30 questions, AI + template fallback)")

    # Pre-warm the fast model for instant first response
    print(f"\n   [Pre-warming models for faster first response...]")
    try:
        llm_mod.prewarm_model(FAST_MODEL)
    except Exception as e:
        print(f"   Pre-warm failed: {e}")

    print(f"{'='*52}\n")
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    if "--web" in sys.argv:
        run_web()
    else:
        run_cli()
