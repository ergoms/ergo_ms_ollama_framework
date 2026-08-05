"""
ModuleBridge — публичные операции ollama_framework.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.core.integrations import bridge

from .client.base import LLMClientError
from .payload_limits import (
    validate_chat_messages,
    validate_embed_texts,
    validate_prompt_payload,
)
from .services.runtime import run_chat, run_chat_stream, run_embed, run_generate, run_health, run_list_models


def _raise_if_invalid(error: Optional[str]) -> None:
    if error:
        raise LLMClientError(error)


@bridge.provide_op('ollama_framework.health')
def _health(config: Optional[Dict[str, Any]] = None, skip_env_injection: bool = False) -> Dict[str, Any]:
    return run_health(config=config, skip_env_injection=skip_env_injection)


@bridge.provide_op('ollama_framework.list_models')
def _list_models(config: Optional[Dict[str, Any]] = None, skip_env_injection: bool = False) -> List[str]:
    return run_list_models(config=config, skip_env_injection=skip_env_injection)


@bridge.provide_op('ollama_framework.generate')
def _generate(
    prompt: str,
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
    _raise_if_invalid(validate_prompt_payload(prompt, system=system))
    return run_generate(
        prompt,
        config=config,
        skip_env_injection=skip_env_injection,
        system=system,
        num_predict=num_predict,
        temperature=temperature,
        stream=stream,
        stream_callback=stream_callback,
        format=format,
        return_stats=return_stats,
    )


@bridge.provide_op('ollama_framework.chat')
def _chat(
    messages: List[Dict[str, Any]],
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
    _raise_if_invalid(validate_chat_messages(messages))
    return run_chat(
        messages,
        config=config,
        skip_env_injection=skip_env_injection,
        num_predict=num_predict,
        temperature=temperature,
        seed=seed,
        stream=stream,
        stream_callback=stream_callback,
        format=format,
        return_stats=return_stats,
    )


@bridge.provide_op('ollama_framework.chat_stream')
def _chat_stream(
    messages: List[Dict[str, Any]],
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    num_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
):
    _raise_if_invalid(validate_chat_messages(messages))
    return run_chat_stream(
        messages,
        config=config,
        skip_env_injection=skip_env_injection,
        num_predict=num_predict,
        temperature=temperature,
        seed=seed,
    )


@bridge.provide_op('ollama_framework.embed')
def _embed(
    texts: List[str],
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    model: Optional[str] = None,
) -> List[List[float]]:
    _raise_if_invalid(validate_embed_texts(texts))
    return run_embed(
        texts,
        config=config,
        skip_env_injection=skip_env_injection,
        model=model,
    )
