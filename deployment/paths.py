"""Пути и окружение локальной Ollama без Django."""

from __future__ import annotations

import os
import platform
import shutil
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def read_env(name: str, default: str = '') -> str:
    value = os.environ.get(name)
    if value is not None and str(value).strip() != '':
        return str(value).strip()
    env_path = PROJECT_ROOT / '.env'
    if not env_path.is_file():
        return default
    for line in env_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, raw = line.partition('=')
        if key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return default


def packages_dir(root: Optional[Path] = None) -> Path:
    return (root or PROJECT_ROOT) / 'virtual_env' / 'packages'


def get_ollama_dir(root: Optional[Path] = None) -> Path:
    return packages_dir(root) / 'ollama'


def get_ollama_models_dir(root: Optional[Path] = None) -> Path:
    return packages_dir(root) / 'ollama_models'


def get_trained_models_dir(root: Optional[Path] = None) -> Path:
    return (root or PROJECT_ROOT) / 'virtual_env' / 'trained_models'


def ensure_ollama_models_dir(root: Optional[Path] = None) -> Path:
    models_dir = get_ollama_models_dir(root)
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_ollama_base_url() -> str:
    return (read_env('OLLAMA_BASE_URL', 'http://127.0.0.1:11434') or 'http://127.0.0.1:11434').rstrip('/')


def get_default_model() -> str:
    return read_env('OLLAMA_DEFAULT_MODEL', 'mistral:7b') or 'mistral:7b'


def get_ollama_executable(root: Optional[Path] = None) -> Optional[Path]:
    ollama_dir = get_ollama_dir(root)
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


def build_ollama_env(
    base: Optional[Mapping[str, str]] = None,
    root: Optional[Path] = None,
) -> dict[str, str]:
    env = dict(os.environ)
    if base:
        env.update(base)
    env['OLLAMA_MODELS'] = str(ensure_ollama_models_dir(root))

    ollama_dir = get_ollama_dir(root)
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


def resolve_ollama_command(*args: str, root: Optional[Path] = None) -> list[str]:
    exe = get_ollama_executable(root)
    if exe is None:
        return ['ollama', *args]
    return [str(exe), *args]
