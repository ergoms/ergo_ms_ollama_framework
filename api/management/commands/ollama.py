"""
Совместимость: ergoms api ollama → deployment-скрипт без Django.

Предпочтительный вызов: ergoms ollama_framework:ollama
"""

from django.core.management.base import BaseCommand

from modules.ollama_framework.deployment.ollama_cli import main as ollama_main


class Command(BaseCommand):
    help = 'Менеджер моделей Ollama (обёртка над deployment-скриптом)'

    def run_from_argv(self, argv):
        raise SystemExit(ollama_main(argv[2:]))

    def handle(self, *args, **options):
        raise SystemExit(ollama_main([]))
