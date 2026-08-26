"""Имена моделей библиотеки Ollama для pull и сверки со списком сервера."""

from __future__ import annotations


def _norm(name: str) -> str:
    return (name or '').strip()


def _base(name: str) -> str:
    return _norm(name).split(':', 1)[0]


def is_ollama_library_name(name: str) -> bool:
    """False для Hugging Face org/name (снимок, не манифест ollama.com)."""
    raw = _norm(name)
    if not raw:
        return False
    lowered = raw.lower()
    if lowered.startswith('hf.co/') or lowered.startswith('huggingface.co/'):
        return True
    if '/' in raw and ':' not in raw.split('/', 1)[0]:
        parts = raw.split('/')
        return not (len(parts) == 2 and all(parts))
    return True


def pull_names(requested: str) -> list[str]:
    """Имена для ollama pull."""
    name = _norm(requested)
    if not name or not is_ollama_library_name(name):
        return []
    return [name]


def matches_installed(requested: str, installed: list[str] | tuple[str, ...]) -> bool:
    """True, если requested уже есть у Ollama."""
    models = [item for item in installed if isinstance(item, str) and item]
    for candidate in pull_names(requested):
        if candidate in models:
            return True
        base = _base(candidate)
        if any(item == base or item.startswith(f'{base}:') for item in models):
            return True
        if any(candidate in item for item in models):
            return True
    return False
