"""Поиск и управление процессом Ollama без Django."""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional
from urllib.parse import urlparse

import psutil

from .paths import (
    build_ollama_env,
    get_ollama_base_url,
    get_ollama_dir,
    resolve_ollama_command,
)

logger = logging.getLogger('modules.ollama_framework.process')

_DEFAULT_OLLAMA_PORT = 11434


def _parse_base_url_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url or get_ollama_base_url())
    host = parsed.hostname or '127.0.0.1'
    port = parsed.port or _DEFAULT_OLLAMA_PORT
    return host, port


def is_tcp_port_in_use(host: str, port: int) -> bool:
    """True, если порт уже занят (слушает другой процесс)."""
    return find_listener_process(port, host=host) is not None


def find_listener_process(port: int, host: str | None = None) -> Optional[psutil.Process]:
    """Возвращает процесс, слушающий TCP-порт, или None."""
    probe_hosts = {host.lower()} if host else set()
    probe_hosts.update({'127.0.0.1', 'localhost', '0.0.0.0', '::', '::1'})

    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            if conn.laddr.port != port:
                continue
            laddr_host = (conn.laddr.ip or '').lower()
            if laddr_host not in probe_hosts and host is not None:
                continue
            try:
                return psutil.Process(conn.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.AccessDenied, OSError) as exc:
        logger.debug('Не удалось перечислить сетевые соединения: %s', exc)
    return None


def is_ollama_server_available(timeout: float = 2.0) -> bool:
    """Проверяет доступность Ollama по HTTP, не только по процессу в ОС."""
    try:
        import httpx

        response = httpx.get(
            f'{get_ollama_base_url()}/api/tags',
            timeout=timeout,
            trust_env=False,
        )
        return response.status_code == 200
    except Exception:
        return False


def _process_cmdline_text(proc: psutil.Process) -> str:
    cmdline = proc.info.get('cmdline') if proc.info else None
    if not cmdline:
        try:
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return ''
    return ' '.join(str(part) for part in (cmdline or [])).lower()


def _process_name(proc: psutil.Process) -> str:
    name = proc.info.get('name') if proc.info else None
    if not name:
        try:
            name = proc.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return ''
    return str(name).lower()


def _process_listens_on_port(proc: psutil.Process, port: int) -> bool:
    try:
        connections_fn = getattr(proc, 'net_connections', proc.connections)
        for conn in connections_fn(kind='inet'):
            if conn.status == psutil.CONN_LISTEN and conn.laddr and conn.laddr.port == port:
                return True
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    return False


def _looks_like_ollama_serve(proc: psutil.Process, *, port: int = _DEFAULT_OLLAMA_PORT) -> bool:
    name = _process_name(proc)
    cmdline_text = _process_cmdline_text(proc)
    if 'ollama' not in name and 'ollama' not in cmdline_text:
        return False
    if 'serve' in cmdline_text:
        return True
    return _process_listens_on_port(proc, port)


def find_ollama(include_wrapper: bool = False) -> Optional[psutil.Process]:
    """
    Ищет запущенный процесс Ollama.

    Args:
        include_wrapper: учитывать обёртку ``start_ollama`` (для stop)

    Returns:
        Объект процесса или None
    """
    wrapper_candidate: Optional[psutil.Process] = None

    base_url = get_ollama_base_url()
    _, default_port = _parse_base_url_host_port(base_url)

    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline_text = _process_cmdline_text(proc)

            if _looks_like_ollama_serve(proc, port=default_port):
                logger.debug('Найден процесс Ollama: PID=%s', proc.pid)
                return proc

            if include_wrapper and 'start_ollama' in cmdline_text:
                wrapper_candidate = proc
                logger.debug(
                    'Найден процесс Ollama (обёртка): PID=%s, CMDLINE=%s',
                    proc.pid,
                    cmdline_text,
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            logger.error('Ошибка при поиске процесса: %s', exc)
            continue

    if wrapper_candidate is not None:
        return wrapper_candidate

    logger.debug('Процесс Ollama не найден')
    return None


def terminate_ollama_process(
    process: psutil.Process | subprocess.Popen[Any],
    *,
    timeout: float = 5.0,
) -> None:
    """Останавливает процесс Ollama и его дочерние процессы."""
    try:
        proc = (
            psutil.Process(process.pid)
            if isinstance(process, subprocess.Popen)
            else process
        )
    except psutil.NoSuchProcess:
        return

    children = proc.children(recursive=True)
    targets = children + [proc]
    for target in targets:
        try:
            target.terminate()
        except psutil.NoSuchProcess:
            continue

    _gone, alive = psutil.wait_procs(targets, timeout=timeout)
    for target in alive:
        try:
            target.kill()
        except psutil.NoSuchProcess:
            continue
    if alive:
        psutil.wait_procs(alive, timeout=timeout)


def _linux_child_preexec() -> None:
    import ctypes

    libc = ctypes.CDLL('libc.so.6', use_errno=True)
    libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _attach_windows_kill_job(process: subprocess.Popen[Any]) -> Any | None:
    import ctypes
    from ctypes import wintypes

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ('ReadOperationCount', ctypes.c_ulonglong),
            ('WriteOperationCount', ctypes.c_ulonglong),
            ('OtherOperationCount', ctypes.c_ulonglong),
            ('ReadTransferCount', ctypes.c_ulonglong),
            ('WriteTransferCount', ctypes.c_ulonglong),
            ('OtherTransferCount', ctypes.c_ulonglong),
        ]

    class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('PerProcessUserTimeLimit', wintypes.LARGE_INTEGER),
            ('PerJobUserTimeLimit', wintypes.LARGE_INTEGER),
            ('LimitFlags', wintypes.DWORD),
            ('MinimumWorkingSetSize', ctypes.c_size_t),
            ('MaximumWorkingSetSize', ctypes.c_size_t),
            ('ActiveProcessLimit', wintypes.DWORD),
            ('Affinity', ctypes.c_size_t),
            ('PriorityClass', wintypes.DWORD),
            ('SchedulingClass', wintypes.DWORD),
        ]

    class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
        _fields_ = [
            ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
            ('IoInfo', IO_COUNTERS),
            ('ProcessMemoryLimit', ctypes.c_size_t),
            ('JobMemoryLimit', ctypes.c_size_t),
            ('PeakProcessMemoryUsed', ctypes.c_size_t),
            ('PeakJobMemoryUsed', ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(
        job,
        9,  # JobObjectExtendedLimitInformation
        ctypes.byref(info),
        ctypes.sizeof(info),
    ):
        kernel32.CloseHandle(job)
        return None

    access = 0x000F0000 | 0x00100000 | 0xFFFF
    process_handle = kernel32.OpenProcess(access, False, process.pid)
    if not process_handle:
        kernel32.CloseHandle(job)
        return None

    try:
        if not kernel32.AssignProcessToJobObject(job, process_handle):
            kernel32.CloseHandle(job)
            return None
    finally:
        kernel32.CloseHandle(process_handle)

    return job


def build_foreground_popen_kwargs() -> dict[str, Any]:
    """Параметры Popen: дочерний процесс завершается вместе с обёрткой."""
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NO_WINDOW}
    return {'preexec_fn': _linux_child_preexec}


@contextmanager
def foreground_child_lifecycle(
    process: subprocess.Popen[Any],
) -> Iterator[subprocess.Popen[Any]]:
    """
    Держит дочерний процесс привязанным к жизненному циклу Python.

    На Windows — Job Object KILL_ON_JOB_CLOSE; на Linux — PR_SET_PDEATHSIG.
  """
    job_handle = None
    if sys.platform == 'win32':
        job_handle = _attach_windows_kill_job(process)
    try:
        yield process
    finally:
        if job_handle is not None:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(job_handle)


def start_ollama_background(
    api_dir: Optional[Path] = None,
    extra_args: Optional[List[str]] = None,
) -> bool:
    """
    Запускает Ollama serve в фоновом режиме.

    Args:
        api_dir: рабочая директория процесса; по умолчанию каталог пакета Ollama
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
