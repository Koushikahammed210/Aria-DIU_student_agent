# run.py

import sys
import os
from agent import Agent, PlannerAgent, _CoverPageSession, PresentationAgent
from config import AGENT_NAME, UPLOAD_FOLDER, ALLOWED_FILE_EXTENSIONS, PERSONAS, OLLAMA_MODELS


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
        """Enhanced status with NIM details."""
        nim_status = llm_mod.get_nim_status()
        model_info = llm_mod.check_ollama_health()
        available_models = [k for k, v in model_info.items() if isinstance(v, dict) and v.get("available")]
        vision_available = llm_mod.has_vision_model()

        return jsonify({
            "mode": agent.mode,
            "persona": agent.persona,
            "persona_info": PERSONAS.get(agent.persona, {}),
            "messages": len(agent.memory.messages),
            "models_available": available_models,
            "selected_model": agent._selected_model or "auto",
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
            "nim_status": llm_mod.get_nim_status(),
        }
        return jsonify(diag)

    # ── Startup Diagnostics ─────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"   {AGENT_NAME} — AI Agent")
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
    nim_status = llm_mod.get_nim_status()
    if nim_status["state"] == "available":
        print(f"   NIM (cloud): available (RPM: {nim_status['rpm']}/{nim_status['rpm_limit']}, model: {nim_status['model']})")
    else:
        print(f"   NIM (cloud): {nim_status['state']}")
        if nim_status.get("last_error"):
            print(f"     Error: {nim_status['last_error']}")

    print(f"{'='*52}\n")
    app.run(debug=False, port=5000)


if __name__ == "__main__":
    if "--web" in sys.argv:
        run_web()
    else:
        run_cli()
