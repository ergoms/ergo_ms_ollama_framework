"""
Совместимость: ergoms api stop_ollama → deployment-скрипт без Django.

Предпочтительный вызов: ergoms ollama_framework:stop-ollama
"""

from django.core.management.base import BaseCommand

from modules.ollama_framework.deployment.stop_ollama import stop_ollama


class Command(BaseCommand):
    help = 'Останавливает Ollama сервер (обёртка над deployment-скриптом)'

    def handle(self, *args, **options):
        raise SystemExit(stop_ollama())
