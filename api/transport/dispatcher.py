"""
Единая точка вызова Ollama для потребителей модулей.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .. import settings as of_settings
from ..client.base import LLMClientError
from .http import invoke_http
from .local import invoke_local

SUPPORTED_OPERATIONS = frozenset({'health', 'list_models', 'generate', 'chat', 'embed'})


def resolve_transport(transport: Optional[str] = None) -> str:
    mode = (transport or of_settings.OLLAMA_FRAMEWORK_TRANSPORT or 'local').strip().lower()
    if mode not in ('local', 'http'):
        raise LLMClientError(
            f'Недопустимый OLLAMA_FRAMEWORK_TRANSPORT={mode!r}. Допустимо: local, http.'
        )
    return mode


def ollama_invoke(
    operation: str,
    *,
    transport: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    api_base: Optional[str] = None,
    auth_token: Optional[str] = None,
    **payload: Any,
) -> Any:
    """
    Вызов операции Ollama через ollama_framework.

    Args:
        operation: health | list_models | generate | chat | embed
        transport: local | http (по умолчанию из env)
        config: переопределения model, base_url и др.
        skip_env_injection: только config + fallback
        api_base: базовый URL API (режим http)
        auth_token: JWT для REST (режим http)
        **payload: аргументы операции (messages, prompt, texts, ...)
    """
    if operation not in SUPPORTED_OPERATIONS:
        raise LLMClientError(f'Неизвестная операция: {operation}')

    mode = resolve_transport(transport)

    if mode == 'local':
        kwargs: Dict[str, Any] = dict(payload)
        if config is not None:
            kwargs['config'] = config
        kwargs['skip_env_injection'] = skip_env_injection
        return invoke_local(operation, **kwargs)

    http_payload: Dict[str, Any] = dict(payload)
    if config is not None:
        http_payload['config'] = config
    http_payload['skip_env_injection'] = skip_env_injection
    return invoke_http(
        operation,
        api_base=api_base or os.getenv('OLLAMA_FRAMEWORK_API_BASE') or of_settings.OLLAMA_FRAMEWORK_API_BASE,
        auth_token=auth_token,
        payload=http_payload,
    )


def ollama_chat(
    messages: List[Dict[str, str]],
    *,
    config: Optional[Dict[str, Any]] = None,
    transport: Optional[str] = None,
    skip_env_injection: bool = False,
    **kwargs: Any,
) -> str:
    return ollama_invoke(
        'chat',
        transport=transport,
        config=config,
        skip_env_injection=skip_env_injection,
        messages=messages,
        **kwargs,
    )


def ollama_generate(
    prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    transport: Optional[str] = None,
    skip_env_injection: bool = False,
    **kwargs: Any,
) -> str:
    return ollama_invoke(
        'generate',
        transport=transport,
        config=config,
        skip_env_injection=skip_env_injection,
        prompt=prompt,
        **kwargs,
    )
