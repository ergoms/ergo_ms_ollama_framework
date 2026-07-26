"""Поиск и управление процессом Ollama без Django."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import psutil

from .paths import (
    build_ollama_env,
    get_ollama_base_url,
    get_ollama_dir,
    resolve_ollama_command,
)

logger = logging.getLogger('modules.ollama_framework.process')


def is_ollama_server_available(timeout: float = 2.0) -> bool:
    """Проверяет доступность Ollama по HTTP, не только по процессу в ОС."""
    try:
        import httpx

        response = httpx.get(f'{get_ollama_base_url()}/api/tags', timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def find_ollama(include_wrapper: bool = False) -> Optional[psutil.Process]:
    """
    Ищет запущенный процесс Ollama.

    Args:
        include_wrapper: учитывать обёртку ``start_ollama`` (для stop)

    Returns:
        Объект процесса или None
    """
    wrapper_candidate: Optional[psutil.Process] = None

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline') or []
            cmdline_lower = [part.lower() for part in cmdline]

            if 'ollama' in cmdline_lower and 'serve' in cmdline_lower:
                logger.debug('Найден процесс Ollama: PID=%s', proc.pid)
                return proc

            if include_wrapper and 'start_ollama' in cmdline_lower:
                wrapper_candidate = proc
                logger.debug(
                    'Найден процесс Ollama (обёртка): PID=%s, CMDLINE=%s',
                    proc.pid,
                    ' '.join(cmdline),
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            logger.error('Ошибка при поиске процесса: %s', exc)
            continue

    if wrapper_candidate is not None:
        return wrapper_candidate

    logger.debug('Процесс Ollama не найден')
    return None


def start_ollama_background(
    api_dir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    Запускает Ollama serve в фоновом режиме.

    Args:
        api_dir: рабочая директория (legacy); по умолчанию каталог пакета Ollama
        extra_args: дополнительные аргументы после ``serve`` (host, port)

    Returns:
        True если сервер доступен после попытки запуска
    """
    if is_ollama_server_available():
        logger.info('Ollama API уже доступен')
        return True

    cmd = resolve_ollama_command('serve')
    if extra_args:
        cmd.extend(extra_args)

    ollama_dir = get_ollama_dir()
    if ollama_dir.is_dir():
        working_dir = ollama_dir
    elif api_dir is not None:
        working_dir = api_dir
    else:
        working_dir = ollama_dir.parent

    try:
        popen_kwargs = {
            'cwd': str(working_dir),
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.PIPE,
            'env': build_ollama_env(),
        }
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs['start_new_session'] = True

        process = subprocess.Popen(cmd, **popen_kwargs)

        for _ in range(15):
            time.sleep(1)
            if is_ollama_server_available():
                logger.info('Ollama API доступен после фонового запуска')
                return True
            if process.poll() is not None:
                stderr = ''
                if process.stderr is not None:
                    stderr = process.stderr.read().decode('utf-8', errors='replace').strip()
                if is_ollama_server_available():
                    return True
                if stderr:
                    logger.error('Ollama завершился при запуске: %s', stderr)
                else:
                    logger.error('Ollama процесс завершился сразу после запуска')
                return False

        if process.poll() is None:
            logger.info('Ollama запущен в фоне (PID: %s)', process.pid)
            return is_ollama_server_available()

        logger.error('Ollama не ответил по API после фонового запуска')
        return False
    except FileNotFoundError:
        logger.error(
            'Ollama не найден в virtual_env/packages/ollama. '
            'Установите: ergoms ollama_framework:install-ollama'
        )
        return False
    except Exception as exc:
        logger.error('Ошибка при запуске Ollama: %s', exc)
        return False
