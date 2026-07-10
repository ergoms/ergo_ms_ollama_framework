"""
Django-команда для запуска Ollama сервера с форматированием логов и метриками.

Пример: ergoms ollama_framework:start-ollama
"""

import logging
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any, Dict, Optional

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from modules.ollama_framework.api.ollama_process import (
    find_ollama,
    is_ollama_server_available,
)
from modules.ollama_framework.api.paths import build_ollama_env, resolve_ollama_command
from src.core.utils.os_abstraction import get_os_abstraction

logger = logging.getLogger('modules.ollama_framework.commands')


class Command(BaseCommand):
    help = 'Запускает Ollama сервер'

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            '--host',
            default=None,
            help='Хост для запуска Ollama',
        )
        parser.add_argument(
            '--port',
            default=None,
            help='Порт для запуска Ollama',
        )

    def parse_ollama_log(self, line: str) -> Optional[Dict[str, Any]]:
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
        param_pattern = r'(\w+)=([^\s"]+|"[^"]*")'
        for param_match in re.finditer(param_pattern, line):
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

    def format_log_line(self, log_data: Dict[str, Any]) -> str:
        time_str = log_data['time']
        level = log_data['level']
        source = log_data['source']
        msg = log_data['message']
        extra = log_data['extra']

        level_colors = {
            'INFO': 'CYAN',
            'WARN': 'YELLOW',
            'ERROR': 'RED',
            'DEBUG': 'MAGENTA',
        }

        source_short = get_os_abstraction().basename_from_path(source)
        formatted_msg = msg

        if 'Listening on' in msg:
            addr_match = re.search(r'Listening on ([^\s]+)', msg)
            if addr_match:
                addr = addr_match.group(1)
                addr_styled = self.style.SUCCESS(addr)  # type: ignore[attr-defined]
                formatted_msg = f"Сервер запущен на {addr_styled}"
        elif 'discovering available GPUs' in msg:
            formatted_msg = 'Поиск доступных GPU...'
        elif 'inference compute' in msg:
            gpu_name = extra.get('description', 'Unknown GPU')
            total_vram = extra.get('total', 'N/A')
            available_vram = extra.get('available', 'N/A')
            formatted_msg = f"GPU: {gpu_name} | VRAM: {available_vram} / {total_vram}"
        elif 'entering low vram mode' in msg:
            total_vram = extra.get('total vram', 'N/A')
            formatted_msg = f"Режим низкой VRAM (всего: {total_vram})"
        elif 'total blobs' in msg:
            formatted_msg = f"Всего blob'ов: {extra.get('total', '0')}"
        elif 'total unused blobs removed' in msg:
            formatted_msg = f"Удалено неиспользуемых blob'ов: {extra.get('total unused blobs removed', '0')}"

        level_color = level_colors.get(level, 'SUCCESS')
        level_styled = getattr(self.style, level_color, self.style.SUCCESS)(f"[{level}]")  # type: ignore[attr-defined]
        time_styled = self.style.HTTP_INFO(time_str)  # type: ignore[attr-defined]
        source_styled = self.style.HTTP_INFO(source_short)  # type: ignore[attr-defined]
        return f"{time_styled} {level_styled} {source_styled} {formatted_msg}"

    def track_generation_metrics(self, log_data: Dict[str, Any], metrics: Dict[str, Any]) -> None:
        extra = log_data['extra']
        raw_line = log_data.get('raw', '')

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

                if eval_duration > 0 and eval_count > 0:
                    self._emit_metrics(metrics, eval_count, eval_duration)
                    return
            except (ValueError, TypeError):
                pass

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

                if eval_duration > 0 and eval_count > 0:
                    self._emit_metrics(metrics, eval_count, eval_duration)
            except (ValueError, TypeError):
                pass

    def _emit_metrics(self, metrics: Dict[str, Any], eval_count: int, eval_duration: float) -> None:
        tokens_per_sec = eval_count / eval_duration
        metrics['total_tokens'] = metrics.get('total_tokens', 0) + eval_count
        metrics['total_duration'] = metrics.get('total_duration', 0) + eval_duration
        avg_tokens_per_sec = (
            metrics['total_tokens'] / metrics['total_duration']
            if metrics.get('total_duration', 0) > 0
            else 0
        )
        self.stdout.write(
            f"\n{self.style.SUCCESS('Генерация:')} "  # type: ignore[attr-defined]
            f"{self.style.SUCCESS(f'{tokens_per_sec:.2f}')} токенов/сек "  # type: ignore[attr-defined]
            f"(среднее: {self.style.SUCCESS(f'{avg_tokens_per_sec:.2f}')} токенов/сек) | "  # type: ignore[attr-defined]
            f"Всего токенов: {self.style.SUCCESS(str(metrics['total_tokens']))}\n"  # type: ignore[attr-defined]
        )

    def read_output(self, pipe, queue: Queue) -> None:
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    queue.put(line.rstrip())
        finally:
            pipe.close()
            queue.put(None)

    def handle(self, *args: tuple, **options: dict) -> None:
        logger.info('Запуск команды start_ollama')

        if find_ollama() or is_ollama_server_available():
            msg = 'Ollama уже запущен'
            logger.warning(msg)
            self.stdout.write(self.style.WARNING(msg))
            return

        api_dir = Path(settings.API_DIR)
        cmd = resolve_ollama_command('serve')
        ollama_env = build_ollama_env()

        if options.get('host'):
            cmd.extend(['--host', str(options['host'])])
        if options.get('port'):
            cmd.extend(['--port', str(options['port'])])

        process = None
        try:
            logger.info('Запуск Ollama с командой: %s', ' '.join(cmd))
            self.stdout.write(self.style.SUCCESS('\n=== Запуск Ollama Server ===\n'))  # type: ignore[attr-defined]

            process = subprocess.Popen(
                cmd,
                cwd=str(api_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=ollama_env,
            )

            output_queue: Queue = Queue()
            output_thread = Thread(target=self.read_output, args=(process.stdout, output_queue))
            output_thread.daemon = True
            output_thread.start()

            metrics: Dict[str, Any] = {
                'total_tokens': 0,
                'total_duration': 0,
            }

            while True:
                try:
                    line = output_queue.get(timeout=0.1)
                    if line is None:
                        break

                    log_data = self.parse_ollama_log(line)
                    if log_data:
                        self.track_generation_metrics(log_data, metrics)
                        self.stdout.write(self.format_log_line(log_data))
                    else:
                        self.stdout.write(line)
                    self.stdout.flush()
                except Empty:
                    if process.poll() is not None:
                        while True:
                            try:
                                line = output_queue.get_nowait()
                                if line is None:
                                    break
                                log_data = self.parse_ollama_log(line)
                                if log_data:
                                    self.stdout.write(self.format_log_line(log_data))
                                else:
                                    self.stdout.write(line)
                            except Empty:
                                break
                        break

            output_thread.join(timeout=1)
            return_code = process.wait()

            if return_code != 0:
                self.stdout.write(self.style.ERROR(f'\nOllama завершился с кодом: {return_code}\n'))  # type: ignore[attr-defined]
                sys.exit(return_code)

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nПолучен сигнал прерывания, завершение работы...\n'))  # type: ignore[attr-defined]
            if process is not None:
                process.terminate()
                process.wait(timeout=5)
            sys.exit(0)
        except FileNotFoundError:
            msg = (
                'Ollama не найден в virtual_env/packages/ollama. '
                'Установите: ergoms ollama_framework:install-ollama'
            )
            logger.error(msg)
            self.stdout.write(self.style.ERROR(f'\n{msg}\n'))  # type: ignore[attr-defined]
            sys.exit(1)
        except Exception as exc:
            logger.error('Ошибка при запуске Ollama: %s', exc)
            self.stdout.write(self.style.ERROR(f'\nОшибка: {exc}\n'))  # type: ignore[attr-defined]
            if process is not None:
                process.terminate()
            raise
