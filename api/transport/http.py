"""
HTTP-транспорт — вызов REST API ollama_framework.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

from .. import settings as of_settings
from ..client.base import LLMClientError

logger = logging.getLogger(__name__)

_OPERATION_PATHS = {
    'health': 'status/',
    'list_models': 'models/',
    'generate': 'generate/',
    'chat': 'chat/',
    'embed': 'embed/',
}


def _api_base(api_base: Optional[str] = None) -> str:
    base = (api_base or of_settings.OLLAMA_FRAMEWORK_API_BASE or '').strip()
    if not base:
        raise LLMClientError(
            'OLLAMA_FRAMEWORK_API_BASE не задан. Укажите базовый URL API ERGO MS для режима http.'
        )
    base = base.rstrip('/') + '/'
    if 'ollama_framework' not in base:
        base = f'{base}ollama_framework/'
    return base


def invoke_http(
    operation: str,
    *,
    api_base: Optional[str] = None,
    auth_token: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    timeout: float = 300.0,
) -> Any:
    path = _OPERATION_PATHS.get(operation)
    if not path:
        raise LLMClientError(f'Неизвестная операция ollama_framework: {operation}')

    url = f'{_api_base(api_base)}{path}'
    headers: Dict[str, str] = {'Content-Type': 'application/json'}
    if auth_token:
        headers['Authorization'] = f'Bearer {auth_token}'

    body = payload or {}
    try:
        if operation == 'health':
            response = httpx.get(url, headers=headers, timeout=min(timeout, 30.0))
        else:
            response = httpx.post(url, json=body, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and 'result' in data:
            return data['result']
        return data
    except httpx.HTTPError as exc:
        logger.warning('HTTP-транспорт ollama_framework (%s): %s', operation, exc)
        raise LLMClientError(f'HTTP ollama_framework/{operation} недоступен: {exc}') from exc
