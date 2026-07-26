"""
Удаление Ollama из virtual_env/packages без Django.

Вызов: ergoms ollama_framework:uninstall-ollama [--remove-models]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from project_layout import packages_dir  # noqa: E402


def ollama_dir(root: Path) -> Path:
    return packages_dir(root) / 'ollama'


def ollama_models_dir(root: Path) -> Path:
    return packages_dir(root) / 'ollama_models'


def uninstall_ollama(root: Path, *, remove_models: bool = False) -> int:
    configure_stdio_utf8()
    target = ollama_dir(root)
    models = ollama_models_dir(root)
    legacy_models = packages_dir(root) / 'models'

    try:
        if target.exists():
            print(f'Удаление Ollama из {target}...')
            shutil.rmtree(target)
            print(format_console('ok', 'Ollama удалён'))
        else:
            print(format_console(
                'warning',
                'Ollama не найден в virtual_env/packages/ollama',
            ))

        if remove_models and models.exists():
            print(f'Удаление моделей из {models}...')
            model_count = len(list(models.iterdir()))
            shutil.rmtree(models)
            models.mkdir(parents=True, exist_ok=True)
            print(format_console('ok', f'Удалено элементов: {model_count}'))
        elif remove_models:
            print(format_console('warning', 'Каталог ollama_models не найден'))

        if remove_models and legacy_models.exists():
            print(f'Удаление устаревшего каталога {legacy_models}...')
            shutil.rmtree(legacy_models, ignore_errors=True)
    except Exception as exc:
        print(format_console('error', f'Ошибка при удалении Ollama: {exc}'), file=sys.stderr)
        return 1

    print(format_console('ok', 'Удаление завершено'))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Удаление Ollama из virtual_env/packages')
    parser.add_argument(
        '--root',
        type=Path,
        default=PROJECT_ROOT,
        help='Корень проекта ERGO MS',
    )
    parser.add_argument(
        '--remove-models',
        action='store_true',
        help='Также удалить модели из virtual_env/packages/ollama_models',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    return uninstall_ollama(args.root.resolve(), remove_models=args.remove_models)


if __name__ == '__main__':
    raise SystemExit(main())
