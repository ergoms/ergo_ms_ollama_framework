"""
Локальный транспорт — вызов операций через ModuleBridge in-process.
"""
from __future__ import annotations

from typing import Any

from src.core.integrations import bridge

from ..client.base import LLMClientError

_OPERATION_MAP = {
    'health': 'ollama_framework.health',
    'list_models': 'ollama_framework.list_models',
    'generate': 'ollama_framework.generate',
    'chat': 'ollama_framework.chat',
    'embed': 'ollama_framework.embed',
}


def invoke_local(operation: str, **kwargs: Any) -> Any:
    op_name = _OPERATION_MAP.get(operation)
    if not op_name:
        raise LLMClientError(f'Неизвестная операция ollama_framework: {operation}')

    if not bridge.has(op_name):
        raise LLMClientError(f'Операция {op_name} не зарегистрирована в ModuleBridge')

    result = bridge.call(op_name, **kwargs)
    if result is None and operation != 'health':
        raise LLMClientError(f'Операция {op_name} вернула пустой результат')
    return result
