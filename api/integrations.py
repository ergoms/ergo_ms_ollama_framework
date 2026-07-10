"""
ModuleBridge — публичные операции ollama_framework.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from src.core.integrations import bridge

from .services.runtime import run_chat, run_embed, run_generate, run_health, run_list_models


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
) -> str:
    return run_generate(
        prompt,
        config=config,
        skip_env_injection=skip_env_injection,
        system=system,
        num_predict=num_predict,
        temperature=temperature,
        stream=stream,
    )


@bridge.provide_op('ollama_framework.chat')
def _chat(
    messages: List[Dict[str, str]],
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    num_predict: Optional[int] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    stream: bool = False,
    format: Optional[Any] = None,
) -> str:
    return run_chat(
        messages,
        config=config,
        skip_env_injection=skip_env_injection,
        num_predict=num_predict,
        temperature=temperature,
        seed=seed,
        stream=stream,
        format=format,
    )


@bridge.provide_op('ollama_framework.embed')
def _embed(
    texts: List[str],
    config: Optional[Dict[str, Any]] = None,
    skip_env_injection: bool = False,
    model: Optional[str] = None,
) -> List[List[float]]:
    return run_embed(
        texts,
        config=config,
        skip_env_injection=skip_env_injection,
        model=model,
    )
