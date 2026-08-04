from __future__ import annotations

from typing import Any, Dict, Optional

from ..config import RuntimeLLMConfig, build_runtime_config
from .base import BaseLLMClient, LLMClientError
from .httpx_client import HttpxOllamaClient

OLLAMA_OPTION_KEYS = ('top_p', 'top_k', 'repeat_penalty', 'num_ctx', 'think')


def build_llm_client(
    *,
    provider: str,
    model: str,
    base_url: str,
    request_timeout: float,
    stream_timeout: float,
    concurrency_limit: int,
    max_retries: int,
    keep_alive: str = '10m',
    provider_config: Optional[Dict[str, Any]] = None,
    device_config: Optional[Dict[str, Any]] = None,
) -> BaseLLMClient:
    provider_config = provider_config or {}
    device_config = device_config or {}

    normalized_provider = (provider or 'ollama').lower()
    if normalized_provider == 'llama_cpp':
        raise LLMClientError(
            'llama.cpp удалён. Используйте LLM_PROVIDER=ollama и Ollama для работы с LLM.'
        )

    base_api_url = provider_config.get('base_url', base_url)

    if normalized_provider in ('auto', 'ollama'):
        return HttpxOllamaClient(
            model=model,
            base_url=base_api_url,
            request_timeout=request_timeout,
            stream_timeout=stream_timeout,
            concurrency_limit=concurrency_limit,
            max_retries=max_retries,
            keep_alive=keep_alive,
            device_config=device_config,
        )

    raise LLMClientError(f'Неподдерживаемый провайдер LLM: {provider}')


def create_client(
    config_overrides: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
) -> Tuple[RuntimeLLMConfig, BaseLLMClient]:
    """
    Создаёт клиент Ollama и runtime-конфигурацию.

    Args:
        config_overrides: переопределения model, base_url, temperature и др.
        skip_env_injection: только overrides + fallback (для модулей с TP_LLM_* и т.п.).
    """
    overrides = dict(config_overrides or {})
    if 'provider' not in overrides:
        overrides['provider'] = 'ollama'
    if 'num_gpu' in overrides:
        if 'device_config' not in overrides or not isinstance(overrides.get('device_config'), dict):
            overrides['device_config'] = {}
        overrides['device_config']['num_gpu'] = overrides['num_gpu']

    runtime_config = build_runtime_config(overrides, skip_env_injection=skip_env_injection)

    for key in OLLAMA_OPTION_KEYS:
        if key in runtime_config.provider_config:
            runtime_config.device_config[key] = runtime_config.provider_config[key]

    provider_name = (
        runtime_config.provider.value
        if hasattr(runtime_config.provider, 'value')
        else str(runtime_config.provider)
    )
    base_url = runtime_config.provider_config.get(
        'base_url',
        runtime_config.base_url or 'http://127.0.0.1:11434',
    )

    client = build_llm_client(
        provider=provider_name,
        model=runtime_config.model or 'mistral:latest',
        base_url=base_url,
        request_timeout=runtime_config.request_timeout,
        stream_timeout=runtime_config.stream_timeout,
        concurrency_limit=runtime_config.concurrency_limit,
        max_retries=runtime_config.max_retries,
        keep_alive=runtime_config.keep_alive,
        provider_config=runtime_config.provider_config,
        device_config=runtime_config.device_config,
    )
    return runtime_config, client
