"""
Остановка процесса Ollama без Django.

Вызов: ergoms ollama_framework:stop-ollama
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from modules.ollama_framework.deployment.process import find_ollama, terminate_ollama_process  # noqa: E402


def stop_ollama() -> int:
    process = find_ollama(include_wrapper=True)
    if not process:
        print(format_console('warning', 'Ollama процесс не найден'))
        return 0

    try:
        pid = process.pid
        terminate_ollama_process(process)
        print(format_console('ok', f'Ollama процесс (PID: {pid}) успешно остановлен'))
        return 0
    except Exception as exc:
        print(format_console('error', f'Ошибка при остановке Ollama: {exc}'))
        return 1


def main(argv: list[str] | None = None) -> int:
    del argv  # нет аргументов
    configure_stdio_utf8()
    return stop_ollama()


if __name__ == '__main__':
    raise SystemExit(main())
