"""
Django команда для удаления Ollama из virtual_env/packages
"""

import shutil
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from modules.ollama_framework.api.paths import get_ollama_dir, get_ollama_models_dir


class Command(BaseCommand):
    help = 'Удаляет Ollama из virtual_env/packages/ollama'

    def add_arguments(self, parser):
        parser.add_argument(
            '--remove-models',
            action='store_true',
            help='Также удалить все модели из virtual_env/packages/ollama_models',
        )

    def handle(self, *args, **options):
        remove_models = options.get('remove_models', False)

        ollama_path = get_ollama_dir()
        models_path = get_ollama_models_dir()
        legacy_models_path = Path(settings.PACKAGES_PATH) / 'models'

        if ollama_path.exists():
            self.stdout.write(f'Удаление Ollama из {ollama_path}...')
            try:
                shutil.rmtree(ollama_path)
                self.stdout.write(self.style.SUCCESS('Ollama удалён'))
            except Exception as e:
                raise CommandError(f'Ошибка при удалении Ollama: {e}') from e
        else:
            self.stdout.write(self.style.WARNING('Ollama не найден в virtual_env/packages/ollama'))

        if remove_models and models_path.exists():
            self.stdout.write(f'Удаление моделей из {models_path}...')
            try:
                model_count = len(list(models_path.iterdir()))
                shutil.rmtree(models_path)
                models_path.mkdir(parents=True, exist_ok=True)
                self.stdout.write(self.style.SUCCESS(f'Удалено элементов: {model_count}'))
            except Exception as e:
                raise CommandError(f'Ошибка при удалении моделей: {e}') from e
        elif remove_models:
            self.stdout.write(self.style.WARNING('Каталог ollama_models не найден'))

        if remove_models and legacy_models_path.exists():
            self.stdout.write(f'Удаление устаревшего каталога {legacy_models_path}...')
            shutil.rmtree(legacy_models_path, ignore_errors=True)

        self.stdout.write(self.style.SUCCESS('\nУдаление завершено!'))
