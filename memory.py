# memory.py

import json
import os
from datetime import datetime
from config import MEMORY_FILE, MAX_HISTORY


class Memory:
    def __init__(self):
        self.messages = []
        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._load()

    def _load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.messages = data.get("messages", [])[-MAX_HISTORY:]
                print(f"Memory loaded — {len(self.messages)} messages from previous sessions.")
            except Exception:
                self.messages = []
        else:
            print("No previous memory. Starting fresh.")

    def _save(self):
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump({"messages": self.messages}, f, indent=2)
        except Exception as e:
            print(f"Warning: could not save memory: {e}")

    def add(self, role: str, content: str):
        self.messages.append({
            "role": role,
            "content": content,
            "time": datetime.now().strftime("%H:%M")
        })
        if len(self.messages) > MAX_HISTORY:
            self.messages = self.messages[-MAX_HISTORY:]
        self._save()

    def get_messages(self) -> list:
        return [{"role": m["role"], "content": m["content"]} for m in self.messages]

    def clear(self):
        self.messages = []
        if os.path.exists(MEMORY_FILE):
            os.remove(MEMORY_FILE)
        print("Memory cleared.")

    def show_history(self):
        if not self.messages:
            print("No history yet.")
            return
        print("\n--- Last 10 messages ---")
        for m in self.messages[-10:]:
            time = m.get("time", "")
            role = "You" if m["role"] == "user" else "Aria"
            print(f"[{time}] {role}: {m['content'][:100]}{'...' if len(m['content']) > 100 else ''}")
        print("------------------------\n")
