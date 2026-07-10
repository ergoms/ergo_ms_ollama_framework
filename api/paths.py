"""Пути и окружение локальной установки Ollama в virtual_env/packages."""

import os
import platform
import shutil
from pathlib import Path
from typing import Mapping, Optional

from django.conf import settings


def get_ollama_dir() -> Path:
    return Path(settings.PACKAGES_PATH) / 'ollama'


def get_ollama_models_dir() -> Path:
    return Path(settings.PACKAGES_PATH) / 'ollama_models'


def ensure_ollama_models_dir() -> Path:
    models_dir = get_ollama_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_ollama_executable() -> Optional[Path]:
    ollama_dir = get_ollama_dir()
    if platform.system().lower() == 'windows':
        exe = ollama_dir / 'ollama.exe'
        if exe.is_file():
            return exe
    else:
        exe = ollama_dir / 'ollama'
        if exe.is_file():
            return exe

    found = shutil.which('ollama')
    return Path(found) if found else None


def build_ollama_env(base: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    env = dict(os.environ)
    if base:
        env.update(base)
    env['OLLAMA_MODELS'] = str(ensure_ollama_models_dir())

    ollama_dir = get_ollama_dir()
    if ollama_dir.is_dir():
        path_sep = ';' if platform.system().lower() == 'windows' else ':'
        current_path = env.get('PATH', '')
        ollama_dir_str = str(ollama_dir)
        if ollama_dir_str not in current_path.split(path_sep):
            env['PATH'] = (
                f'{ollama_dir_str}{path_sep}{current_path}'
                if current_path
                else ollama_dir_str
            )

    return env


def resolve_ollama_command(*args: str) -> list[str]:
    exe = get_ollama_executable()
    if exe is None:
        return ['ollama', *args]
    return [str(exe), *args]
