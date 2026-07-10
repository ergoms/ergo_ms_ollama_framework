"""
Проверка состояния Ollama: процесс и доступность API.

Пример: ergoms ollama_framework:ollama-status
"""

from django.core.management.base import BaseCommand

from modules.ollama_framework.api.ollama_process import find_ollama
from modules.ollama_framework.api.services.runtime import run_health


class Command(BaseCommand):
    help = 'Проверяет процесс Ollama и доступность API'

    def handle(self, *args, **options):
        process = find_ollama()
        if process:
            self.stdout.write(self.style.SUCCESS(f'Процесс Ollama: PID {process.pid}'))
        else:
            self.stdout.write(self.style.WARNING('Процесс Ollama не найден'))

        health = run_health()
        if health.get('available'):
            self.stdout.write(self.style.SUCCESS(f"API: доступен ({health.get('base_url')})"))
            models = health.get('models') or []
            self.stdout.write(f'Моделей: {len(models)}')
        else:
            self.stdout.write(self.style.ERROR(f"API: {health.get('message', health.get('error'))}"))
