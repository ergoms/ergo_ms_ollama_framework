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
# Vision: Ollama /api/chat images — base64 без data-URI.
MAX_IMAGES_PER_MESSAGE = 8
MAX_IMAGE_B64_CHARS = 15_000_000
MAX_IMAGES_TOTAL_B64_CHARS = 40_000_000


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
    total_images_b64 = 0
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
        images = msg.get('images')
        if images is None:
            continue
        if not isinstance(images, list):
            return f'messages[{i}].images должен быть списком'
        if len(images) > MAX_IMAGES_PER_MESSAGE:
            return (
                f'messages[{i}].images слишком много '
                f'(макс. {MAX_IMAGES_PER_MESSAGE})'
            )
        for j, image in enumerate(images):
            if not isinstance(image, str):
                return f'messages[{i}].images[{j}] должен быть строкой (base64)'
            if not image:
                return f'messages[{i}].images[{j}] пустой'
            if len(image) > MAX_IMAGE_B64_CHARS:
                return (
                    f'messages[{i}].images[{j}] слишком большой '
                    f'(макс. {MAX_IMAGE_B64_CHARS} символов base64)'
                )
            total_images_b64 += len(image)
            if total_images_b64 > MAX_IMAGES_TOTAL_B64_CHARS:
                return (
                    f'суммарный размер images слишком большой '
                    f'(макс. {MAX_IMAGES_TOTAL_B64_CHARS} символов base64)'
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
