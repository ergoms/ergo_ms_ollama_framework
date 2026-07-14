"""Безопасная обработка config из REST-запросов ollama_framework."""
from __future__ import annotations

import ipaddress
import json
from typing import Any, Optional
from urllib.parse import urlparse

from rest_framework.exceptions import ValidationError

from src.core.cms.adp.services.permissions import PermissionService

from . import settings as of_settings

_URL_KEYS = ('base_url',)


def parse_config_param(raw: Any, field_name: str = 'config') -> Optional[dict]:
    if raw is None or raw == '':
        return None
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ValidationError({field_name: 'Некорректный формат config.'})
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValidationError({field_name: 'Некорректный JSON в config.'}) from exc
    if parsed is not None and not isinstance(parsed, dict):
        raise ValidationError({field_name: 'config должен быть объектом JSON.'})
    return parsed


def _allowed_hosts() -> set[str]:
    hosts = {'localhost', '127.0.0.1', '::1'}
    configured = (of_settings.OLLAMA_BASE_URL or '').strip()
    if configured:
        parsed = urlparse(configured if '://' in configured else f'http://{configured}')
        if parsed.hostname:
            hosts.add(parsed.hostname.lower())
    return hosts


def _is_private_or_local_host(hostname: str) -> bool:
    if not hostname:
        return False
    lowered = hostname.lower()
    if lowered in _allowed_hosts():
        return True
    try:
        ip = ipaddress.ip_address(lowered)
        return ip.is_loopback or ip.is_private
    except ValueError:
        return False


def _validate_base_url(url: str) -> None:
    if not url or not isinstance(url, str):
        raise ValidationError({'config': 'base_url должен быть непустой строкой.'})
    parsed = urlparse(url.strip())
    if parsed.scheme not in ('http', 'https'):
        raise ValidationError({'config': 'base_url: допустимы только http и https.'})
    if not _is_private_or_local_host(parsed.hostname or ''):
        raise ValidationError({'config': 'base_url: разрешены только локальные адреса Ollama.'})


def _strip_url_overrides(config: dict) -> dict:
    sanitized = dict(config)
    for key in _URL_KEYS:
        sanitized.pop(key, None)
    provider_config = sanitized.get('provider_config')
    if isinstance(provider_config, dict):
        pc = dict(provider_config)
        for key in _URL_KEYS:
            pc.pop(key, None)
        sanitized['provider_config'] = pc
    return sanitized


def sanitize_client_config(config: Optional[dict], user) -> Optional[dict]:
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValidationError({'config': 'config должен быть объектом.'})

    if PermissionService.is_admin(user):
        result = dict(config)
        base_url = result.get('base_url')
        if base_url:
            _validate_base_url(str(base_url))
        provider_config = result.get('provider_config')
        if isinstance(provider_config, dict):
            pc_base = provider_config.get('base_url')
            if pc_base:
                _validate_base_url(str(pc_base))
        return result

    return _strip_url_overrides(config)


def gateway_error_message() -> str:
    return 'Не удалось выполнить запрос к Ollama. Проверьте ergoms ollama_framework:ollama-status.'
