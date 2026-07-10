"""
Методы для работы с Ollama — CLI и администрирование.
Runtime-вызовы LLM — через modules.ollama_framework.api.client.
"""

import json
import subprocess
from pathlib import Path
from datetime import datetime

from django.conf import settings

from modules.ollama_framework.api.client.factory import create_client
from modules.ollama_framework.api.services.runtime import run_health


class OllamaMethods:
    """Класс с методами для работы с Ollama"""

    def __init__(self, stdout):
        self.stdout = stdout

    def get_ollama_client(self):
        """Получить Python SDK ollama (только для pull/remove CLI)."""
        try:
            import ollama
            return ollama
        except ImportError:
            raise ImportError('Возникла ошибка при импорте ollama')

    def show_info(self):
        """Показать информацию о системе ollama"""
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                check=True,
            )
            self.stdout.write('Информация о системе Ollama:')
            self.stdout.write(f'CLI версия: {result.stdout.strip()}')

            health = run_health()
            if health.get('available'):
                self.stdout.write('Сервер: работает')
                models = health.get('models') or []
                self.stdout.write(f'Установлено моделей: {len(models)}')
            else:
                self.stdout.write(f"Сервер: недоступен ({health.get('error', '')})")

            try:
                from importlib.metadata import version
                client_version = version('ollama')
                self.stdout.write(f'Python клиент: {client_version}')
            except Exception:
                self.stdout.write('Python клиент: установлен')
        except subprocess.CalledProcessError:
            self.stdout.write('Ollama CLI не найден в PATH')
        except Exception as e:
            self.stdout.write(f'Ошибка: {e}')

    def list_models(self):
        """Показать список доступных моделей"""
        try:
            _, client = create_client()
            models_list = client.list_models()
            if not models_list:
                self.stdout.write('Модели не найдены.')
                self.stdout.write('Для скачивания: ergoms ollama_framework:ollama --pull <model_name>')
                return
            self.stdout.write(f'Найдено моделей: {len(models_list)}')
            self.stdout.write('-' * 40)
            for i, name in enumerate(models_list, 1):
                self.stdout.write(f'{i:2d}. {name}')
        except Exception as e:
            self.stdout.write(f'Ошибка получения списка моделей: {e}')

    def pull_model(self, model_name):
        try:
            client = self.get_ollama_client()
        except ImportError:
            self.stdout.write('Ollama Python client не установлен')
            return

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
        except Exception as e:
            self.stdout.write('')
            self.stdout.write(f'Ошибка при скачивании модели: {e}')

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

    def remove_model(self, model_name):
        try:
            client = self.get_ollama_client()
            models = client.list()
            available_models = []
            for model in models.get('models', []):
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    available_models.append(model.get('name', str(model)))
            if model_name not in available_models:
                self.stdout.write(f'Модель {model_name} не найдена')
                return
            client.delete(model_name)
            self.stdout.write(f'Модель {model_name} удалена')
        except ImportError:
            self.stdout.write('Ollama Python client не установлен')
        except Exception as e:
            self.stdout.write(f'Ошибка при удалении модели: {e}')

    def show_help(self):
        self.stdout.write('Менеджер моделей Ollama (ollama_framework)')
        self.stdout.write('=' * 30)
        self.stdout.write('  ergoms ollama_framework:ollama --list')
        self.stdout.write('  ergoms ollama_framework:ollama-status')
        self.stdout.write('  ergoms ollama_framework:start-ollama')

    def test_model(self, model_name):
        try:
            if not self._check_model_availability(model_name):
                return
            from modules.ollama_framework.api.services.runtime import run_chat

            self.stdout.write(f'Тестирую модель {model_name}...')
            test_prompt = 'Привет! Как дела?'
            answer = run_chat(
                [{'role': 'user', 'content': test_prompt}],
                config={'model': model_name},
            )
            self.stdout.write(f'Запрос: {test_prompt}')
            self.stdout.write(f'Ответ: {answer}')
            self.stdout.write('Модель работает корректно')
        except Exception as e:
            self.stdout.write(f'Ошибка при тестировании: {e}')

    def generate_training_data(self):
        self.stdout.write(
            'Генерация тренировочных данных пока не реализована. '
            'Подготовьте JSONL вручную и используйте --train с --data.'
        )

    def train_model(self, base_model, data_file_path):
        try:
            client = self.get_ollama_client()
            models = client.list()
            available_models = []
            for model in models.get('models', []):
                if hasattr(model, 'model'):
                    available_models.append(model.model)
                elif isinstance(model, dict):
                    available_models.append(model.get('name', str(model)))
            if base_model not in available_models:
                self.stdout.write(f'Базовая модель {base_model} не найдена')
                return

            trained_models_dir = Path(settings.TRAINED_MODELS_PATH)
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
                ['ollama', 'create', trained_model_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                self.stdout.write(f'Модель {trained_model_name} создана')
            else:
                self.stdout.write(f'Ошибка создания модели: {result.stderr}')
        except Exception as e:
            self.stdout.write(f'Ошибка обучения: {e}')

    def send_message(self, model_name, message, system_prompt=None, temperature=0.7, max_tokens=2048):
        try:
            from modules.ollama_framework.api.services.runtime import run_chat

            if not self._check_model_availability(model_name):
                return None
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': message})
            return run_chat(
                messages,
                config={'model': model_name},
                temperature=temperature,
                num_predict=max_tokens,
            )
        except Exception as e:
            self.stdout.write(f'Ошибка при отправке сообщения: {e}')
            return None

    def _check_model_availability(self, model_name):
        try:
            _, client = create_client({'model': model_name})
            models = client.list_models()
            if model_name not in models and not any(model_name in m for m in models):
                self.stdout.write(f'Модель {model_name} не найдена.')
                self.stdout.write(f'Доступные модели: {", ".join(models)}')
                return False
            return True
        except Exception as e:
            self.stdout.write(f'Ошибка при проверке модели: {e}')
            return False

