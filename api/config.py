from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from . import settings as of_settings


class ComputeDevice(str, Enum):
    AUTO = 'auto'
    GPU = 'gpu'
    CPU = 'cpu'


class LLMProvider(str, Enum):
    AUTO = 'auto'
    OLLAMA = 'ollama'


@dataclass
class RuntimeLLMConfig:
    provider: LLMProvider = LLMProvider.AUTO
    model: Optional[str] = of_settings.OLLAMA_DEFAULT_MODEL
    base_url: Optional[str] = of_settings.OLLAMA_BASE_URL
    request_timeout: float = of_settings.OLLAMA_REQUEST_TIMEOUT
    stream_timeout: float = of_settings.OLLAMA_STREAM_TIMEOUT
    compute_device: ComputeDevice = ComputeDevice.AUTO
    concurrency_limit: int = of_settings.OLLAMA_CONCURRENCY_LIMIT
    max_retries: int = of_settings.OLLAMA_MAX_RETRIES
    keep_alive: str = of_settings.OLLAMA_KEEP_ALIVE
    provider_config: Dict[str, Any] = field(default_factory=dict)
    device_config: Dict[str, Any] = field(default_factory=dict)

    def copy_with_overrides(self, overrides: Optional[Dict[str, Any]] = None) -> 'RuntimeLLMConfig':
        if not overrides:
            return self

        data: Dict[str, Any] = {
            'provider': self.provider,
            'model': self.model,
            'base_url': self.base_url,
            'request_timeout': self.request_timeout,
            'stream_timeout': self.stream_timeout,
            'compute_device': self.compute_device,
            'concurrency_limit': self.concurrency_limit,
            'max_retries': self.max_retries,
            'keep_alive': self.keep_alive,
            'provider_config': dict(self.provider_config),
            'device_config': dict(self.device_config),
        }

        provider_overrides: Dict[str, Any] = {}
        device_overrides: Dict[str, Any] = {}

        for key, value in overrides.items():
            if key in data:
                data[key] = value
            elif key.startswith('provider__'):
                provider_overrides[key.split('__', 1)[1]] = value
            elif key.startswith('device__'):
                device_overrides[key.split('__', 1)[1]] = value
            else:
                provider_overrides[key] = value

        if isinstance(data['provider'], str):
            p = data['provider'].lower()
            data['provider'] = LLMProvider.OLLAMA if p == 'llama_cpp' else LLMProvider(p)
        if isinstance(data['compute_device'], str):
            parsed = parse_compute_device(data['compute_device'])
            data['compute_device'] = parsed if parsed is not None else ComputeDevice.AUTO

        data['provider_config'].update(provider_overrides)
        data['device_config'].update(device_overrides)

        return RuntimeLLMConfig(**data)


def parse_compute_device(value: Any) -> Optional[ComputeDevice]:
    if value is None:
        return None
    if isinstance(value, ComputeDevice):
        return value
    normalized = str(value).strip().lower()
    if normalized in ('auto', 'gpu', 'cpu'):
        return ComputeDevice(normalized)
    return None


def _sync_num_gpu(config: RuntimeLLMConfig) -> None:
    """auto — не трогаем num_gpu (Ollama сам выберет GPU/CPU)."""
    if config.compute_device == ComputeDevice.CPU:
        config.device_config.setdefault('num_gpu', 0)
    elif config.compute_device == ComputeDevice.GPU:
        config.device_config.setdefault('num_gpu', -1)


def _inject_env_defaults(
    config: RuntimeLLMConfig,
    *,
    apply_compute_device: bool = True,
) -> RuntimeLLMConfig:
    from src.config.env import env

    env_provider = env.str('LLM_PROVIDER', default='').lower()
    if env_provider in ('llama_cpp', 'ollama'):
        config.provider = LLMProvider.OLLAMA

    if config.provider in (LLMProvider.AUTO, LLMProvider.OLLAMA):
        env_base_url = env.str('OLLAMA_BASE_URL', default='') or env.str('OLLAMA_API_BASE', default='')
        if env_base_url:
            config.base_url = env_base_url

        env_model = env.str('OLLAMA_DEFAULT_MODEL', default='')
        if env_model:
            config.model = env_model

    if config.base_url and not config.provider_config.get('base_url'):
        config.provider_config['base_url'] = config.base_url

    if apply_compute_device:
        parsed = parse_compute_device(of_settings.OLLAMA_COMPUTE_DEVICE)
        config.compute_device = parsed if parsed is not None else ComputeDevice.AUTO

    _sync_num_gpu(config)

    return config


_FALLBACK_CONFIG = {
    'provider': LLMProvider.OLLAMA,
    'model': 'mistral:latest',
    'base_url': 'http://127.0.0.1:11434',
    'request_timeout': 180.0,
    'stream_timeout': 300.0,
    'compute_device': ComputeDevice.AUTO,
    'concurrency_limit': 8,
    'max_retries': 2,
    'keep_alive': '10m',
    'provider_config': {},
    'device_config': {},
}


def build_runtime_config(
    overrides: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
) -> RuntimeLLMConfig:
    override_values = dict(overrides or {})
    explicit_compute = 'compute_device' in override_values or 'num_gpu' in override_values

    if skip_env_injection:
        base_config = RuntimeLLMConfig(**_FALLBACK_CONFIG)
        config = base_config.copy_with_overrides(override_values)
        if config.base_url and not config.provider_config.get('base_url'):
            config.provider_config['base_url'] = config.base_url
        _sync_num_gpu(config)
    else:
        base_config = RuntimeLLMConfig()
        config = base_config.copy_with_overrides(override_values)
        config = _inject_env_defaults(config, apply_compute_device=not explicit_compute)

    if config.provider == LLMProvider.AUTO:
        config.provider = LLMProvider.OLLAMA

    return config
