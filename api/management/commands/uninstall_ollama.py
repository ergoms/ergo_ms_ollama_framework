"""
Совместимость: ergoms api uninstall_ollama → deployment-скрипт без Django-логики.

Предпочтительный вызов: ergoms ollama_framework:uninstall-ollama
"""

from django.core.management.base import BaseCommand, CommandError

from modules.ollama_framework.deployment.uninstall_ollama import (
    PROJECT_ROOT,
    uninstall_ollama,
)


class Command(BaseCommand):
    help = 'Удаляет Ollama из virtual_env/packages/ollama (обёртка над deployment-скриптом)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--remove-models',
            action='store_true',
            help='Также удалить модели из virtual_env/packages/ollama_models',
        )

    def handle(self, *args, **options):
        code = uninstall_ollama(
            PROJECT_ROOT,
            remove_models=options.get('remove_models', False),
        )
        if code != 0:
            raise CommandError('Удаление Ollama завершилось с ошибкой')
