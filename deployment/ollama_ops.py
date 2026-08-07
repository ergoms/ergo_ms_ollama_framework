"""Ops-методы Ollama для CLI без Django."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TextIO

import httpx

from .paths import (
    get_ollama_base_url,
    get_trained_models_dir,
    resolve_ollama_command,
)
from .process import (
    find_ollama,
    is_ollama_server_available,
    start_ollama_background,
)


class StreamWriter:
    """Совместим с Django BaseCommand.stdout.write(ending=...)."""

    def __init__(self, stream: Optional[TextIO] = None):
        self._stream = stream or sys.stdout

    def write(self, msg: str = '', ending: str = '\n') -> None:
        self._stream.write(f'{msg}{ending}')

    def flush(self) -> None:
        self._stream.flush()


class OllamaOps:
    """Администрирование моделей и простые запросы к Ollama API."""

    def __init__(self, stdout: Optional[Any] = None):
        self.stdout = stdout or StreamWriter()

    def get_ollama_client(self):
        try:
            import ollama
            return ollama
        except ImportError as exc:
            raise ImportError('Возникла ошибка при импорте ollama') from exc

    def show_info(self) -> None:
        try:
            result = subprocess.run(
                resolve_ollama_command('--version'),
                capture_output=True,
                text=True,
                check=True,
            )
            self.stdout.write('Информация о системе Ollama:')
            self.stdout.write(f'CLI версия: {result.stdout.strip()}')

            if is_ollama_server_available():
                models = self._list_model_names()
                self.stdout.write('Сервер: работает')
                self.stdout.write(f'Установлено моделей: {len(models)}')
            else:
                self.stdout.write('Сервер: недоступен')

            try:
                from importlib.metadata import version
                self.stdout.write(f'Python клиент: {version("ollama")}')
            except Exception:
                self.stdout.write('Python клиент: установлен')
        except subprocess.CalledProcessError:
            self.stdout.write('Ollama CLI не найден')
        except Exception as exc:
            self.stdout.write(f'Ошибка: {exc}')

    def list_models(self) -> None:
        try:
            models_list = self._list_model_names()
            if not models_list:
                self.stdout.write('Модели не найдены.')
                self.stdout.write(
                    'Для скачивания: ergoms ollama_framework:ollama --pull <model_name>'
                )
                return
            self.stdout.write(f'Найдено моделей: {len(models_list)}')
            self.stdout.write('-' * 40)
            for i, name in enumerate(models_list, 1):
                self.stdout.write(f'{i:2d}. {name}')
        except Exception as exc:
            self.stdout.write(f'Ошибка получения списка моделей: {exc}')

    def is_model_installed(self, model_name: str) -> bool:
        try:
            models = self._list_model_names()
        except Exception:
            return False
        if model_name in models:
            return True
        base = model_name.split(':', 1)[0]
        for installed in models:
            if installed == model_name:
                return True
            if installed.startswith(f'{base}:'):
                return True
            if model_name in installed:
                return True
        return False

    def pull_model(self, model_name: str) -> bool:
        try:
            client = self.get_ollama_client()
        except ImportError:
            self.stdout.write('Ollama Python client не установлен')
            return False

        self.stdout.write(f'Скачиваю модель {model_name}...')
        last_key = None
        try:
            for progress in client.pull(model_name, stream=True):
                key = self._pull_progress_key(progress)
                line = self._format_pull_progress(progress)
                if not line or key == last_key:
                    continue
                self.stdout.write(f'\r{line}', ending='')
                self.stdout.flush()
                last_key = key
            self.stdout.write('')
            self.stdout.write(f'Модель {model_name} успешно скачана')
            return True
        except Exception as exc:
            self.stdout.write('')
            self.stdout.write(f'Ошибка при скачивании модели: {exc}')
            return False

    def remove_model(self, model_name: str) -> None:
        try:
            client = self.get_ollama_client()
            available = self._sdk_model_names(client)
            if model_name not in available:
                self.stdout.write(f'Модель {model_name} не найдена')
                return
            client.delete(model_name)
            self.stdout.write(f'Модель {model_name} удалена')
        except ImportError:
            self.stdout.write('Ollama Python client не установлен')
        except Exception as exc:
            self.stdout.write(f'Ошибка при удалении модели: {exc}')

    def show_help(self) -> None:
        self.stdout.write('Менеджер моделей Ollama (ollama_framework)')
        self.stdout.write('=' * 30)
        self.stdout.write('  ergoms ollama_framework:ollama --list')
        self.stdout.write('  ergoms ollama_framework:ollama-status')
        self.stdout.write('  ergoms ollama_framework:start-ollama')

    def test_model(self, model_name: str) -> None:
        try:
            if not self._check_model_availability(model_name):
                return
            self.stdout.write(f'Тестирую модель {model_name}...')
            test_prompt = 'Привет! Как дела?'
            answer = self.send_message(model_name, test_prompt)
            self.stdout.write(f'Запрос: {test_prompt}')
            self.stdout.write(f'Ответ: {answer}')
            if answer:
                self.stdout.write('Модель работает корректно')
        except Exception as exc:
            self.stdout.write(f'Ошибка при тестировании: {exc}')

    def train_model(self, base_model: str, data_file_path: str) -> None:
        try:
            client = self.get_ollama_client()
            available = self._sdk_model_names(client)
            if base_model not in available:
                self.stdout.write(f'Базовая модель {base_model} не найдена')
                return

            trained_models_dir = get_trained_models_dir()
            data_file = trained_models_dir / data_file_path
            if not data_file.exists():
                self.stdout.write(f'Файл данных не найден: {data_file}')
                return

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            file_stem = Path(data_file_path).stem
            trained_model_name = f'{file_stem}-{timestamp}'

            self.stdout.write(f'Создаю модель {trained_model_name} из {base_model}...')
            modelfile_path = trained_models_dir / f'{trained_model_name}.modelfile'
            modelfile_content = f'FROM {base_model}\nPARAMETER temperature 0.3\n'
            modelfile_path.write_text(modelfile_content, encoding='utf-8')

            result = subprocess.run(
                resolve_ollama_command('create', trained_model_name, '-f', str(modelfile_path)),
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.stdout.write(f'Модель {trained_model_name} создана')
            else:
                self.stdout.write(f'Ошибка создания модели: {result.stderr}')
        except Exception as exc:
            self.stdout.write(f'Ошибка обучения: {exc}')

    def send_message(
        self,
        model_name: str,
        message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        try:
            if not self._check_model_availability(model_name):
                return None
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': message})
            payload = {
                'model': model_name,
                'messages': messages,
                'stream': False,
                'options': {
                    'temperature': temperature,
                    'num_predict': max_tokens,
                },
            }
            response = httpx.post(
                f'{get_ollama_base_url()}/api/chat',
                json=payload,
                timeout=180.0,
                trust_env=False,
            )
            response.raise_for_status()
            data = response.json()
            return (data.get('message') or {}).get('content') or data.get('response')
        except Exception as exc:
            self.stdout.write(f'Ошибка при отправке сообщения: {exc}')
            return None

    def ensure_ollama_running(self, wait_seconds: int = 30) -> None:
        if is_ollama_server_available():
            return

        if find_ollama():
            self.stdout.write('Ollama запускается, ожидание API...')
            self._wait_for_ollama_api(wait_seconds)
            return

        self.stdout.write('Ollama не запущен. Запускаю...')
        if not start_ollama_background():
            if is_ollama_server_available():
                return
            raise RuntimeError(
                'Не удалось запустить Ollama. '
                'Запустите в отдельном терминале: ergoms ollama_framework:start-ollama'
            )
        self._wait_for_ollama_api(wait_seconds)

    def _wait_for_ollama_api(self, wait_seconds: int) -> None:
        import time

        self.stdout.write('Ожидание запуска Ollama...')
        for i in range(wait_seconds):
            if is_ollama_server_available():
                self.stdout.write('Ollama готов к работе')
                return
            time.sleep(1)
            if (i + 1) % 5 == 0:
                self.stdout.write(f'   ... еще {wait_seconds - i - 1} секунд')
        raise RuntimeError('Ollama не стал доступен за отведенное время')

    def _list_model_names(self) -> list[str]:
        response = httpx.get(
            f'{get_ollama_base_url()}/api/tags',
            timeout=10.0,
            trust_env=False,
        )
        response.raise_for_status()
        models = response.json().get('models') or []
        names = []
        for model in models:
            if isinstance(model, dict):
                name = model.get('name') or model.get('model')
                if name:
                    names.append(name)
        return names

    def _sdk_model_names(self, client) -> list[str]:
        models = client.list()
        available = []
        for model in models.get('models', []):
            if hasattr(model, 'model'):
                available.append(model.model)
            elif isinstance(model, dict):
                available.append(model.get('name', str(model)))
        return available

    def _check_model_availability(self, model_name: str) -> bool:
        try:
            models = self._list_model_names()
            if model_name not in models and not any(model_name in m for m in models):
                self.stdout.write(f'Модель {model_name} не найдена.')
                self.stdout.write(f'Доступные модели: {", ".join(models)}')
                return False
            return True
        except Exception as exc:
            self.stdout.write(f'Ошибка при проверке модели: {exc}')
            return False

    def _pull_progress_key(self, progress):
        if hasattr(progress, 'model_dump'):
            data = progress.model_dump()
        elif isinstance(progress, dict):
            data = progress
        else:
            return None
        return (
            data.get('status'),
            data.get('completed'),
            data.get('total'),
            data.get('digest'),
        )

    def _format_pull_progress(self, progress) -> str:
        if hasattr(progress, 'model_dump'):
            data = progress.model_dump()
        elif isinstance(progress, dict):
            data = progress
        else:
            return str(progress)

        status = data.get('status') or ''
        completed = data.get('completed')
        total = data.get('total')
        digest = data.get('digest') or ''
        digest_short = f' [{digest[:12]}]' if digest else ''

        if completed is not None and total and total > 0:
            percent = completed / total * 100
            completed_mb = completed / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            return (
                f'{status}{digest_short}: {percent:.1f}% '
                f'({completed_mb:.1f} / {total_mb:.1f} МБ)'
            )
        return f'{status}{digest_short}' if status else ''


def print_status() -> int:
    """Статус процесса и API. Код выхода 0 если API доступен."""
    from console_tags import format_console

    process = find_ollama()
    if process:
        print(format_console('ok', f'Процесс Ollama: PID {process.pid}'))
    else:
        print(format_console('warning', 'Процесс Ollama не найден'))

    base_url = get_ollama_base_url()
    if is_ollama_server_available():
        try:
            models = OllamaOps()._list_model_names()
        except Exception:
            models = []
        print(format_console('ok', f'API: доступен ({base_url})'))
        print(f'Моделей: {len(models)}')
        return 0

    print(format_console('error', f'API: недоступен ({base_url})'))
    return 1
