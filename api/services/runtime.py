"""
Runtime-операции Ollama — общая логика для ModuleBridge и REST API.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..client.factory import create_client
from ..ollama_process import find_ollama


def _resolve_client(config: Optional[Dict[str, Any]] = None, skip_env_injection: bool = False):
    runtime_config, client = create_client(config, skip_env_injection=skip_env_injection)
    return runtime_config, client


def run_health(config: Optional[Dict[str, Any]] = None, skip_env_injection: bool = False) -> Dict[str, Any]:
    _, client = _resolve_client(config, skip_env_injection=skip_env_injection)
    result = client.check_health()
    process = find_ollama()
    result['process_running'] = process is not None
    if process is not None:
        result['process_pid'] = process.pid
    if result.get('available'):
        result['message'] = 'Ollama доступен'
    else:
        result['message'] = result.get('error', 'Ollama недоступен')
    return result


def run_list_models(config: Optional[Dict[str, Any]] = None, skip_env_injection: bool = False) -> List[str]:
    _, client = _resolve_client(config, skip_env_injection=skip_env_injection)
    return client.list_models()


def run_generate(
    prompt: str,
    *,
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    system: Optional[str] = None,
    num_predict: Optional[int] = None,
    temperature: float = 0.7,
    stream: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
    format: Optional[Any] = None,
    return_stats: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    _, client = _resolve_client(config, skip_env_injection=skip_env_injection)
    return client.complete(
        prompt,
        system=system,
        num_predict=num_predict,
        temperature=temperature,
        stream=stream,
        stream_callback=stream_callback,
        format=format,
        return_stats=return_stats,
    )


def run_chat(
    messages: List[Dict[str, str]],
    *,
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    num_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = False,
    stream_callback: Optional[Callable[[str], None]] = None,
    format: Optional[Any] = None,
    return_stats: bool = False,
) -> str | tuple[str, Dict[str, Any]]:
    _, client = _resolve_client(config, skip_env_injection=skip_env_injection)
    return client.chat(
        messages,
        num_predict=num_predict,
        temperature=temperature,
        seed=seed,
        stream=stream,
        stream_callback=stream_callback,
        format=format,
        return_stats=return_stats,
    )


def run_embed(
    texts: List[str],
    *,
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    model: Optional[str] = None,
) -> List[List[float]]:
    _, client = _resolve_client(config, skip_env_injection=skip_env_injection)
    return client.embed(texts, model=model)
