"""
Хвост логов Ollama (файл в logs/ или journalctl службы ergo-ollama).

Вызов: ergoms ollama_framework:logs-ollama [строки]
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
SCRIPTS_DIR = DEPLOYMENT_DIR / 'scripts'
for _path in (DEPLOYMENT_DIR, SCRIPTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from modules.ollama_framework.deployment.paths import (  # noqa: E402
    OLLAMA_SERVE_LOG_NAME,
    get_ollama_serve_log_path,
)

OLLAMA_LOG_NAME = OLLAMA_SERVE_LOG_NAME
OLLAMA_UNIT = 'ergo-ollama'


def _print_last_lines(path: Path, lines: int) -> None:
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError as exc:
        print(format_console('error', f'Не удалось прочитать {path}: {exc}'), file=sys.stderr)
        return
    chunk = text.splitlines()[-lines:]
    for line in chunk:
        print(line)


def _follow_file(path: Path, lines: int) -> int:
    print(format_console('info', f'Лог Ollama: {path}'))
    print()
    _print_last_lines(path, lines)
    try:
        with path.open('r', encoding='utf-8', errors='replace') as handle:
            handle.seek(0, 2)
            while True:
                line = handle.readline()
                if line:
                    print(line, end='')
                    continue
                time.sleep(0.25)
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        print(format_console('error', f'Ошибка чтения {path}: {exc}'), file=sys.stderr)
        return 1


def _tail_journal(lines: int) -> int:
    journalctl = shutil.which('journalctl')
    if not journalctl:
        print(format_console('error', 'journalctl не найден'), file=sys.stderr)
        return 1
    print(format_console('info', f'Лог службы {OLLAMA_UNIT} (journalctl)'))
    print()
    try:
        subprocess.run(
            [journalctl, '-u', OLLAMA_UNIT, '-n', str(lines), '-f', '--no-pager'],
            check=False,
        )
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(description='Хвост логов Ollama')
    parser.add_argument(
        'lines',
        nargs='?',
        type=int,
        default=500,
        help='Число строк (по умолчанию 500)',
    )
    args = parser.parse_args(argv)
    lines = max(1, int(args.lines or 500))

    log_path = get_ollama_serve_log_path(PROJECT_ROOT)
    if log_path.is_file():
        return _follow_file(log_path, lines)

    if sys.platform.startswith('linux'):
        return _tail_journal(lines)

    print(
        format_console(
            'error',
            f'Файл логов не найден: {log_path}. '
            'Запустите Ollama (ergoms ollama_framework:start-ollama) '
            'или установите службу ergo-ollama.',
        ),
        file=sys.stderr,
    )
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
