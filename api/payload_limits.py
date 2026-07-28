"""Лимиты размера запросов к LLM REST API."""

from __future__ import annotations

from typing import Any, Optional

# Ограничения против DoS на API / GPU.
MAX_PROMPT_CHARS = 100_000
MAX_SYSTEM_CHARS = 50_000
MAX_MESSAGES = 100
MAX_MESSAGE_CHARS = 50_000
MAX_EMBED_TEXTS = 64
MAX_EMBED_TEXT_CHARS = 20_000


def validate_prompt_payload(
    prompt: Any,
    *,
    system: Any = None,
) -> Optional[str]:
    if not isinstance(prompt, str):
        return 'prompt должен быть строкой'
    if len(prompt) > MAX_PROMPT_CHARS:
        return f'prompt слишком длинный (макс. {MAX_PROMPT_CHARS} символов)'
    if system is not None:
        if not isinstance(system, str):
            return 'system должен быть строкой'
        if len(system) > MAX_SYSTEM_CHARS:
            return f'system слишком длинный (макс. {MAX_SYSTEM_CHARS} символов)'
    return None


def validate_chat_messages(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return 'messages должен быть списком'
    if len(messages) > MAX_MESSAGES:
        return f'слишком много messages (макс. {MAX_MESSAGES})'
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return f'messages[{i}] должен быть объектом'
        content = msg.get('content', '')
        if content is None:
            content = ''
        if not isinstance(content, str):
            return f'messages[{i}].content должен быть строкой'
        if len(content) > MAX_MESSAGE_CHARS:
            return (
                f'messages[{i}].content слишком длинный '
                f'(макс. {MAX_MESSAGE_CHARS} символов)'
            )
    return None


def validate_embed_texts(texts: Any) -> Optional[str]:
    if not isinstance(texts, list):
        return 'texts должен быть списком'
    if len(texts) > MAX_EMBED_TEXTS:
        return f'слишком много texts (макс. {MAX_EMBED_TEXTS})'
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            return f'texts[{i}] должен быть строкой'
        if len(text) > MAX_EMBED_TEXT_CHARS:
            return (
                f'texts[{i}] слишком длинный '
                f'(макс. {MAX_EMBED_TEXT_CHARS} символов)'
            )
    return None
