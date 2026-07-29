"""Пути локальной Ollama — re-export без Django settings."""

from ..deployment.paths import (  # noqa: F401
    build_ollama_env,
    ensure_ollama_models_dir,
    get_ollama_dir,
    get_ollama_executable,
    get_ollama_models_dir,
    resolve_ollama_command,
)

__all__ = [
    'build_ollama_env',
    'ensure_ollama_models_dir',
    'get_ollama_dir',
    'get_ollama_executable',
    'get_ollama_models_dir',
    'resolve_ollama_command',
]
