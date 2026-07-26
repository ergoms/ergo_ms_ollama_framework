"""
CLI управления моделями Ollama без Django.

Вызов: ergoms ollama_framework:ollama [--list|--pull ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from modules.ollama_framework.deployment.ollama_ops import OllamaOps  # noqa: E402
from modules.ollama_framework.deployment.paths import get_default_model  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='ergoms ollama_framework:ollama',
        description='Менеджер моделей Ollama (без Django)',
    )
    parser.add_argument('--list', action='store_true', help='Показать список установленных моделей')
    parser.add_argument('--pull', type=str, help='Скачать модель (например: mistral:latest)')
    parser.add_argument('--remove', type=str, help='Удалить модель')
    parser.add_argument('--test', type=str, help='Протестировать модель')
    parser.add_argument('--info', action='store_true', help='Информация о системе Ollama')
    parser.add_argument('--train', type=str, help='Создать модель из Modelfile (базовая модель)')
    parser.add_argument(
        '--data',
        type=str,
        help='Путь к файлу данных в virtual_env/trained_models',
    )
    parser.add_argument('--chat', nargs='+', help='Отправить сообщение к модели')
    parser.add_argument('--prompt', nargs='+', help='Алиас для --chat')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Имя модели (по умолчанию OLLAMA_DEFAULT_MODEL)',
    )
    parser.add_argument('--interactive', action='store_true', help='Интерактивный чат')
    parser.add_argument('--system-prompt', type=str, help='Системный промпт')
    parser.add_argument('--temperature', type=float, default=0.7, help='Температура генерации')
    parser.add_argument('--max-tokens', type=int, default=2048, help='Лимит токенов ответа')
    return parser


def _send_single_message(ops: OllamaOps, model_name: str, message: str, options) -> None:
    ops.stdout.write(f'Отправляю сообщение к модели {model_name}...')
    ops.stdout.write(f'Сообщение: {message}')
    ops.stdout.write('-' * 50)
    answer = ops.send_message(
        model_name,
        message,
        options.system_prompt,
        options.temperature,
        options.max_tokens,
    )
    if answer:
        ops.stdout.write(f'Ответ: {answer}')
    else:
        ops.stdout.write('Не удалось получить ответ от модели')


def _run_interactive(ops: OllamaOps, model_name: str, options) -> None:
    if not ops._check_model_availability(model_name):
        return

    ops.stdout.write(f'Интерактивный режим с моделью {model_name}')
    ops.stdout.write('Введите сообщения (Ctrl+C для выхода):')
    ops.stdout.write('-' * 50)
    if options.system_prompt:
        ops.stdout.write(f'Системный промпт: {options.system_prompt}')
        ops.stdout.write('-' * 50)

    while True:
        try:
            user_input = input('\nВы: ').strip()
            if not user_input:
                continue
            ops.stdout.write('\nМодель думает...')
            answer = ops.send_message(
                model_name,
                user_input,
                options.system_prompt,
                options.temperature,
                options.max_tokens,
            )
            if answer:
                ops.stdout.write(f'Модель: {answer}')
            else:
                ops.stdout.write('Не удалось получить ответ от модели')
        except (KeyboardInterrupt, EOFError):
            ops.stdout.write('\n\nДо свидания!')
            break
        except Exception as exc:
            ops.stdout.write(f'\nОшибка: {exc}')


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    ops = OllamaOps()
    model_name = args.model or get_default_model()

    if args.prompt:
        chat_message = ' '.join(args.prompt)
    elif args.chat:
        chat_message = ' '.join(args.chat)
    else:
        chat_message = None

    try:
        needs_ollama = bool(
            chat_message
            or args.interactive
            or args.test
            or args.train
            or args.pull
            or args.remove
            or args.list
        )
        if needs_ollama:
            ops.ensure_ollama_running()

        if args.info:
            ops.show_info()
        elif args.list:
            ops.list_models()
        elif args.pull:
            ops.pull_model(args.pull)
        elif args.remove:
            ops.remove_model(args.remove)
        elif args.test:
            ops.test_model(args.test)
        elif args.train:
            if not args.data:
                print(format_console('error', 'Укажите файл данных: --data путь/к/файлу.jsonl'))
                return 1
            ops.train_model(args.train, args.data)
        elif chat_message:
            _send_single_message(ops, model_name, chat_message, args)
        elif args.interactive:
            _run_interactive(ops, model_name, args)
        else:
            ops.show_help()
        return 0
    except Exception as exc:
        print(format_console('error', str(exc)), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
