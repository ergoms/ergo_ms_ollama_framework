"""
Скачивание моделей Ollama, объявленных в modules/*/ollama_models.yaml.

Вызов: ergoms ollama_framework:pull-setup-models
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
SCRIPTS_DIR = DEPLOYMENT_DIR / 'scripts'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from ollama_models_loader import load_resolved_models  # noqa: E402

from modules.ollama_framework.deployment.install_ollama import is_installed  # noqa: E402
from modules.ollama_framework.deployment.ollama_ops import OllamaOps  # noqa: E402


def pull_setup_models(root: Path) -> int:
    root = root.resolve()
    models = load_resolved_models(str(root))
    if not models:
        print(format_console('skip', 'Нет моделей для установки (ollama_models.yaml)'))
        return 0

    if not is_installed(root):
        print(format_console('skip', 'Ollama не установлен — пропуск pull-setup-models'))
        return 0

    ops = OllamaOps()
    try:
        ops.ensure_ollama_running()
    except RuntimeError as exc:
        print(format_console('error', str(exc)))
        return 1

    failed_required: list[str] = []
    for model in models:
        if ops.is_model_installed(model.name):
            print(format_console('skip', f'Модель {model.name} уже установлена'))
            continue

        source_hint = model.source_module or 'ollama_framework'
        print(format_console('info', f'Скачиваю {model.name} ({source_hint})…'))
        if ops.pull_model(model.name):
            continue
        if model.required:
            failed_required.append(model.name)
        else:
            print(format_console('warning', f'Не удалось скачать необязательную модель {model.name}'))

    if failed_required:
        names = ', '.join(failed_required)
        print(format_console('error', f'Не удалось скачать обязательные модели: {names}'))
        print(format_console('info', 'Повторите: ergoms ollama_framework:pull-setup-models'))
        return 1

    print(format_console('ok', 'Модели setup-full готовы'))
    return 0


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        prog='ergoms ollama_framework:pull-setup-models',
        description='Скачать модели Ollama из modules/*/ollama_models.yaml',
    )
    parser.add_argument(
        '--root',
        type=Path,
        default=PROJECT_ROOT,
        help='Корень проекта ERGO MS',
    )
    args = parser.parse_args(argv)
    try:
        return pull_setup_models(args.root.resolve())
    except Exception as exc:
        print(format_console('error', str(exc)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
