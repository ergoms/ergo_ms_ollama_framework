"""Утилиты процесса Ollama — re-export django-free слоя deployment."""

from ..deployment.paths import get_ollama_base_url  # noqa: F401
from ..deployment.process import (  # noqa: F401
    find_ollama,
    is_ollama_server_available,
    start_ollama_background,
)

__all__ = [
    'find_ollama',
    'get_ollama_base_url',
    'is_ollama_server_available',
    'start_ollama_background',
]
