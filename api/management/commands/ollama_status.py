"""
Совместимость: ergoms api ollama_status → deployment-скрипт без Django.

Предпочтительный вызов: ergoms ollama_framework:ollama-status
"""

from django.core.management.base import BaseCommand

from modules.ollama_framework.deployment.ollama_status import main as status_main


class Command(BaseCommand):
    help = 'Проверяет процесс Ollama и доступность API (обёртка над deployment-скриптом)'

    def handle(self, *args, **options):
        raise SystemExit(status_main())
