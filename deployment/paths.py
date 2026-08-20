"""Пути и окружение локальной Ollama без Django."""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _read_env_file_value(env_path: Path, name: str) -> Optional[str]:
    if not env_path.is_file():
        return None
    try:
        lines = env_path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, _, raw = line.partition('=')
        if key.strip() == name:
            return raw.strip().strip('"').strip("'")
    return None


def read_env(name: str, default: str = '') -> str:
    """Читает переменную: os.environ → modules/ollama_framework/.env → корневой .env."""
    value = os.environ.get(name)
    if value is not None and str(value).strip() != '':
        return str(value).strip()
    module_val = _read_env_file_value(
        PROJECT_ROOT / 'modules' / 'ollama_framework' / '.env',
        name,
    )
    if module_val is not None and str(module_val).strip() != '':
        return str(module_val).strip()
    root_val = _read_env_file_value(PROJECT_ROOT / '.env', name)
    if root_val is not None and str(root_val).strip() != '':
        return str(root_val).strip()
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


DEFAULT_OLLAMA_HOST = '127.0.0.1'
DEFAULT_OLLAMA_PORT = 11434
_LOOPBACK_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1'})


def _is_loopback_host(host: str) -> bool:
    return (host or '').strip().lower() in _LOOPBACK_HOSTS


def _normalize_bind_host(host: str) -> str:
    """localhost → 127.0.0.1; остальные адреса без изменения."""
    stripped = (host or '').strip() or DEFAULT_OLLAMA_HOST
    if stripped.lower() == 'localhost':
        return DEFAULT_OLLAMA_HOST
    return stripped


def parse_ollama_host_port(value: str) -> tuple[str, int]:
    """Разбирает OLLAMA_HOST (`host:port`) или URL в пару хост/порт."""
    raw = (value or '').strip()
    if not raw:
        return DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_PORT
    if '://' not in raw:
        raw = f'http://{raw}'
    parsed = urlparse(raw)
    host = (parsed.hostname or DEFAULT_OLLAMA_HOST).strip() or DEFAULT_OLLAMA_HOST
    port = int(parsed.port or DEFAULT_OLLAMA_PORT)
    return host, port


def resolve_ollama_listen_addr(
    *,
    override_host: Optional[str] = None,
    override_port: Optional[str | int] = None,
) -> tuple[str, int]:
    """
    Адрес прослушивания ``ollama serve``.

    По умолчанию loopback. LAN (`0.0.0.0` / внешний IP) — только явным
    ``OLLAMA_HOST`` или ``--host``. Хост из ``OLLAMA_BASE_URL`` берётся,
    только если это loopback; иначе клиентский URL не открывает serve наружу.
    """
    port_override: int | None = None
    if override_port is not None and str(override_port).strip() != '':
        port_override = int(override_port)

    if override_host and override_host.strip():
        host = _normalize_bind_host(override_host)
        if port_override is not None:
            return host, port_override
        _, fallback_port = parse_ollama_host_port(
            read_env('OLLAMA_HOST', '') or get_ollama_base_url()
        )
        return host, fallback_port

    explicit = read_env('OLLAMA_HOST', '')
    if explicit:
        host, port = parse_ollama_host_port(explicit)
        if port_override is not None:
            port = port_override
        return _normalize_bind_host(host), port

    base_host, base_port = parse_ollama_host_port(get_ollama_base_url())
    if port_override is not None:
        base_port = port_override
    if _is_loopback_host(base_host):
        return _normalize_bind_host(base_host), base_port
    return DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_PORT


def format_ollama_host(host: str, port: int) -> str:
    return f'{host}:{port}'


def get_default_model() -> str:
    return read_env('OLLAMA_DEFAULT_MODEL', 'mistral:latest') or 'mistral:latest'


def get_ollama_executable(root: Optional[Path] = None) -> Optional[Path]:
    ollama_dir = get_ollama_dir(root)
    if platform.system().lower() == 'windows':
        candidates = (
            ollama_dir / 'ollama.exe',
            ollama_dir / 'bin' / 'ollama.exe',
        )
    else:
        candidates = (
            ollama_dir / 'ollama',
            ollama_dir / 'bin' / 'ollama',
        )
    for exe in candidates:
        if exe.is_file():
            return exe

    found = shutil.which('ollama')
    return Path(found) if found else None


def get_ollama_runtime_home(root: Optional[Path] = None) -> Path:
    """Служебный HOME для ollama serve (кэш, не в git)."""
    return (root or PROJECT_ROOT) / 'virtual_env' / 'cache' / 'ollama' / 'home'


OLLAMA_SERVE_LOG_NAME = 'ollama-serve.log'


def get_ollama_serve_log_path(root: Optional[Path] = None) -> Path:
    """Файл логов serve: ``logs/ollama-serve.log`` или ``ERGO_LOG_FILE_OLLAMA``."""
    project_root = root or PROJECT_ROOT
    scripts = project_root / 'core' / 'deployment' / 'scripts'
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    import log_env

    return log_env.log_file_path('OLLAMA', project_root)


def purge_ollama_identity_keys(home: Optional[Path] = None) -> None:
    """Удаляет id_ed25519*, которые ollama создаёт при старте (для cloud; локально не нужны)."""
    base = Path(home) if home is not None else get_ollama_runtime_home()
    ollama_dir = base / '.ollama'
    for name in ('id_ed25519', 'id_ed25519.pub'):
        path = ollama_dir / name
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def build_ollama_env(
    base: Optional[Mapping[str, str]] = None,
    root: Optional[Path] = None,
) -> dict[str, str]:
    project_root = root or PROJECT_ROOT
    env = dict(os.environ)
    if base:
        env.update(base)
    env['OLLAMA_MODELS'] = str(ensure_ollama_models_dir(project_root))

    # systemd часто без HOME; бинарь ollama требует переменную.
    # Держим в cache/ (gitignore), не рядом с packages.
    ollama_home = get_ollama_runtime_home(project_root)
    ollama_home.mkdir(parents=True, exist_ok=True)
    if not (env.get('HOME') or '').strip():
        env['HOME'] = str(ollama_home)

    # Локальный режим только через env (без server.json / ключей в дереве проекта на виду)
    # Переопределение: OLLAMA_NO_CLOUD=0 в .env
    if not (env.get('OLLAMA_NO_CLOUD') or '').strip():
        env['OLLAMA_NO_CLOUD'] = '1'

    # Явный bind: не полагаться на дефолт бинаря (исторически бывал 0.0.0.0).
    if not (env.get('OLLAMA_HOST') or '').strip():
        listen_host, listen_port = resolve_ollama_listen_addr()
        env['OLLAMA_HOST'] = format_ollama_host(listen_host, listen_port)

    # Локальный API не через корпоративный http_proxy
    local_hosts = '127.0.0.1,localhost,::1'
    for key in ('NO_PROXY', 'no_proxy'):
        current = (env.get(key) or '').strip()
        if not current:
            env[key] = local_hosts
            continue
        parts = [p.strip() for p in current.split(',') if p.strip()]
        for host in local_hosts.split(','):
            if host not in parts:
                parts.append(host)
        env[key] = ','.join(parts)

    ollama_dir = get_ollama_dir(project_root)
    if ollama_dir.is_dir():
        path_sep = ';' if platform.system().lower() == 'windows' else ':'
        parts = env.get('PATH', '').split(path_sep) if env.get('PATH') else []
        for prefix in (ollama_dir / 'bin', ollama_dir):
            prefix_str = str(prefix)
            if prefix.is_dir() and prefix_str not in parts:
                parts.insert(0, prefix_str)
        env['PATH'] = path_sep.join(parts)

    return env


def resolve_ollama_command(*args: str, root: Optional[Path] = None) -> list[str]:
    exe = get_ollama_executable(root)
    if exe is None:
        return ['ollama', *args]
    return [str(exe), *args]
