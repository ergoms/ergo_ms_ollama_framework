"""
Статус процесса и API Ollama без Django.

Вызов: ergoms ollama_framework:ollama-status
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

from console_tags import configure_stdio_utf8  # noqa: E402
from modules.ollama_framework.deployment.ollama_ops import print_status  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    del argv
    configure_stdio_utf8()
    return print_status()


if __name__ == '__main__':
    raise SystemExit(main())
