"""
Запуск Ollama serve в foreground без Django.

Вызов: ergoms ollama_framework:start-ollama [--host ...] [--port ...]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from modules.ollama_framework.deployment.paths import (  # noqa: E402
    build_ollama_env,
    get_ollama_dir,
    resolve_ollama_command,
)
from modules.ollama_framework.deployment.process import (  # noqa: E402
    find_ollama,
    is_ollama_server_available,
)


def parse_ollama_log(line: str) -> Optional[Dict[str, Any]]:
    pattern = r'time=([^\s]+)\s+level=(\w+)\s+source=([^\s]+)\s+msg="([^"]*)"'
    match = re.match(pattern, line)
    if not match:
        return None

    time_str, level, source, msg = match.groups()
    try:
        time_obj = datetime.fromisoformat(time_str.replace('+', '+').replace('Z', '+00:00'))
        time_formatted = time_obj.strftime('%H:%M:%S')
    except Exception:
        time_formatted = time_str.split('T')[1].split('+')[0] if 'T' in time_str else time_str

    extra_params: Dict[str, str] = {}
    for param_match in re.finditer(r'(\w+)=([^\s"]+|"[^"]*")', line):
        key, value = param_match.groups()
        if key not in ('time', 'level', 'source', 'msg'):
            extra_params[key] = value.strip('"')

    return {
        'time': time_formatted,
        'level': level,
        'source': source,
        'message': msg,
        'extra': extra_params,
        'raw': line,
    }


def format_log_line(log_data: Dict[str, Any]) -> str:
    time_str = log_data['time']
    level = log_data['level']
    source = Path(log_data['source']).name
    msg = log_data['message']
    extra = log_data['extra']

    formatted_msg = msg
    if 'Listening on' in msg:
        addr_match = re.search(r'Listening on ([^\s]+)', msg)
        if addr_match:
            formatted_msg = f"Сервер запущен на {addr_match.group(1)}"
    elif 'discovering available GPUs' in msg:
        formatted_msg = 'Поиск доступных GPU...'
    elif 'inference compute' in msg:
        gpu_name = extra.get('description', 'Unknown GPU')
        total_vram = extra.get('total', 'N/A')
        available_vram = extra.get('available', 'N/A')
        formatted_msg = f'GPU: {gpu_name} | VRAM: {available_vram} / {total_vram}'
    elif 'entering low vram mode' in msg:
        formatted_msg = f"Режим низкой VRAM (всего: {extra.get('total vram', 'N/A')})"
    elif 'total blobs' in msg:
        formatted_msg = f"Всего blob'ов: {extra.get('total', '0')}"
    elif 'total unused blobs removed' in msg:
        formatted_msg = (
            f"Удалено неиспользуемых blob'ов: "
            f"{extra.get('total unused blobs removed', '0')}"
        )

    return f'{time_str} [{level}] {source} {formatted_msg}'


def track_generation_metrics(log_data: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    extra = log_data['extra']
    raw_line = log_data.get('raw', '')
    eval_count = None
    eval_duration = None

    if 'eval_count' in extra or 'eval_duration' in extra:
        try:
            eval_count = int(extra.get('eval_count', 0))
            eval_duration_str = extra.get('eval_duration', '0')
            if 'ns' in str(eval_duration_str):
                eval_duration = float(str(eval_duration_str).replace('ns', '')) / 1e9
            elif 'ms' in str(eval_duration_str):
                eval_duration = float(str(eval_duration_str).replace('ms', '')) / 1000
            else:
                eval_duration = float(eval_duration_str) / 1e9
        except (ValueError, TypeError):
            pass

    if eval_count is None or eval_duration is None:
        eval_count_match = re.search(r'eval_count=(\d+)', raw_line)
        eval_duration_match = re.search(r'eval_duration=([\d.]+)(ns|ms|s)?', raw_line)
        if eval_count_match and eval_duration_match:
            try:
                eval_count = int(eval_count_match.group(1))
                duration_val = float(eval_duration_match.group(1))
                duration_unit = eval_duration_match.group(2) or 'ns'
                if duration_unit == 'ns':
                    eval_duration = duration_val / 1e9
                elif duration_unit == 'ms':
                    eval_duration = duration_val / 1000
                else:
                    eval_duration = duration_val
            except (ValueError, TypeError):
                return

    if not eval_count or not eval_duration or eval_duration <= 0:
        return

    tokens_per_sec = eval_count / eval_duration
    metrics['total_tokens'] = metrics.get('total_tokens', 0) + eval_count
    metrics['total_duration'] = metrics.get('total_duration', 0) + eval_duration
    avg = (
        metrics['total_tokens'] / metrics['total_duration']
        if metrics.get('total_duration', 0) > 0
        else 0
    )
    print(
        f"\nГенерация: {tokens_per_sec:.2f} токенов/сек "
        f"(среднее: {avg:.2f}) | Всего токенов: {metrics['total_tokens']}"
    )


def _read_output(pipe, queue: Queue) -> None:
    try:
        for line in iter(pipe.readline, ''):
            if line:
                queue.put(line.rstrip())
    finally:
        pipe.close()
        queue.put(None)


def start_ollama(host: Optional[str] = None, port: Optional[str] = None) -> int:
    if find_ollama() or is_ollama_server_available():
        print(format_console('warning', 'Ollama уже запущен'))
        return 0

    cmd = resolve_ollama_command('serve')
    if host:
        cmd.extend(['--host', str(host)])
    if port:
        cmd.extend(['--port', str(port)])

    ollama_dir = get_ollama_dir()
    working_dir = ollama_dir if ollama_dir.is_dir() else PROJECT_ROOT
    process = None

    try:
        print(format_console('info', 'Запуск Ollama Server'))
        process = subprocess.Popen(
            cmd,
            cwd=str(working_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
            env=build_ollama_env(),
        )

        output_queue: Queue = Queue()
        output_thread = Thread(target=_read_output, args=(process.stdout, output_queue))
        output_thread.daemon = True
        output_thread.start()

        metrics: Dict[str, Any] = {'total_tokens': 0, 'total_duration': 0}

        while True:
            try:
                line = output_queue.get(timeout=0.1)
                if line is None:
                    break
                log_data = parse_ollama_log(line)
                if log_data:
                    track_generation_metrics(log_data, metrics)
                    print(format_log_line(log_data))
                else:
                    print(line)
            except Empty:
                if process.poll() is not None:
                    while True:
                        try:
                            line = output_queue.get_nowait()
                            if line is None:
                                break
                            log_data = parse_ollama_log(line)
                            print(format_log_line(log_data) if log_data else line)
                        except Empty:
                            break
                    break

        output_thread.join(timeout=1)
        return_code = process.wait()
        if return_code != 0:
            print(format_console('error', f'Ollama завершился с кодом: {return_code}'))
        return return_code

    except KeyboardInterrupt:
        print(format_console('warning', 'Получен сигнал прерывания, завершение работы...'))
        if process is not None:
            process.terminate()
            process.wait(timeout=5)
        return 0
    except FileNotFoundError:
        print(
            format_console(
                'error',
                'Ollama не найден в virtual_env/packages/ollama. '
                'Установите: ergoms ollama_framework:install-ollama',
            )
        )
        return 1
    except Exception as exc:
        print(format_console('error', f'Ошибка: {exc}'))
        if process is not None:
            process.terminate()
        return 1


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description='Запуск Ollama serve')
    parser.add_argument('--host', default=None, help='Хост для запуска Ollama')
    parser.add_argument('--port', default=None, help='Порт для запуска Ollama')
    args = parser.parse_args(argv)
    return start_ollama(host=args.host, port=args.port)


if __name__ == '__main__':
    raise SystemExit(main())
