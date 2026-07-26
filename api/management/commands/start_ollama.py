"""
Совместимость: ergoms api start_ollama → deployment-скрипт без Django.

Предпочтительный вызов: ergoms ollama_framework:start-ollama
"""

from django.core.management.base import BaseCommand

from modules.ollama_framework.deployment.start_ollama import main as start_main


class Command(BaseCommand):
    help = 'Запускает Ollama сервер (обёртка над deployment-скриптом)'

    def run_from_argv(self, argv):
        raise SystemExit(start_main(argv[2:]))

    def handle(self, *args, **options):
        raise SystemExit(start_main([]))
