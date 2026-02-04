from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import streamlit as st
from filelock import BaseFileLock, FileLock

from ollama_core import OllamaConfig
from spanish_ai_tutor import SpanishAITutor


class SpanishTutorUI:
    HISTORY_PATH: Path = Path(__file__).with_name("chat_history.json")
    HISTORY_LOCK_PATH: Path = HISTORY_PATH.with_suffix(".lock")
    ALLOWED_ROLES: set[str] = {"system", "developer", "user", "assistant"}

    def __init__(self) -> None:
        self._config = OllamaConfig()

    def run(self) -> None:
        st.set_page_config(page_title="Spanish AI Tutor", page_icon="💬")

        st.title("Spanish AI Tutor")
        st.caption("Chat con tu tutor de español impulsado por Ollama.")

        if "messages" not in st.session_state:
            st.session_state.messages = self._load_history()

        if "tutor" not in st.session_state:
            st.session_state.tutor = SpanishAITutor(self._config)

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt: str | None = st.chat_input("Escribe tu pregunta en español...")

        if prompt:
            self._handle_user_prompt(prompt)

        self._render_sidebar()

    def _handle_user_prompt(self, prompt: str) -> None:
        st.session_state.messages.append({"role": "user", "content": prompt})
        self._save_history(st.session_state.messages)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                reply: str = st.session_state.tutor.create_completion(prompt)
            st.markdown(reply)

        if reply.strip():
            st.session_state.messages.append({"role": "assistant", "content": reply})
            self._save_history(st.session_state.messages)

    def _render_sidebar(self) -> None:
        with st.sidebar:
            st.subheader("Opciones")
            if st.button("Reiniciar conversación"):
                st.session_state.messages = []
                self._save_history(st.session_state.messages)
                st.session_state.tutor = SpanishAITutor(self._config)
                st.rerun()

    @classmethod
    def _load_history(cls) -> list[dict[str, Any]]:
        if not cls.HISTORY_PATH.exists():
            return []
        lock: BaseFileLock = FileLock(str(cls.HISTORY_LOCK_PATH))
        try:
            with lock:
                data: Any = json.loads(cls.HISTORY_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, list):
            return [
                item
                for item in data
                if isinstance(item, dict)
                and item.get("role") in cls.ALLOWED_ROLES
                and isinstance(item.get("content"), str)
                and item.get("content", "").strip()
            ]
        return []

    @classmethod
    def _save_history(cls, messages: list[dict[str, Any]]) -> None:
        lock: BaseFileLock = FileLock(str(cls.HISTORY_LOCK_PATH))
        with lock:
            cls.HISTORY_PATH.write_text(
                json.dumps(messages, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


SpanishTutorUI().run()
