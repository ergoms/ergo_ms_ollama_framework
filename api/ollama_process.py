"""Утилиты для поиска и управления процессами Ollama в ОС."""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import psutil

logger = logging.getLogger('modules.ollama_framework.process')


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


def start_ollama_background(api_dir: Path, extra_args: Optional[List[str]] = None) -> bool:
    """
    Запускает Ollama serve в фоновом режиме.

    Args:
        api_dir: рабочая директория API
        extra_args: дополнительные аргументы после ``serve`` (host, port)

    Returns:
        True если процесс запущен и не завершился сразу
    """
    cmd: List[str] = ['ollama', 'serve']
    if extra_args:
        cmd.extend(extra_args)

    try:
        popen_kwargs = {
            'cwd': str(api_dir),
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
        }
        if sys.platform == 'win32':
            popen_kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:
            popen_kwargs['start_new_session'] = True

        process = subprocess.Popen(cmd, **popen_kwargs)
        time.sleep(2)

        if process.poll() is None:
            logger.info('Ollama запущен в фоне (PID: %s)', process.pid)
            return True

        logger.error('Ollama процесс завершился сразу после запуска')
        return False
    except FileNotFoundError:
        logger.error('Ollama не найден. Убедитесь, что Ollama установлен и доступен в PATH.')
        return False
    except Exception as exc:
        logger.error('Ошибка при запуске Ollama: %s', exc)
        return False
