"""
Совместимость: ergoms api install_ollama → тот же скрипт без полного Django-сценария.

Предпочтительный вызов: ergoms ollama_framework:install-ollama
"""

from django.core.management.base import BaseCommand, CommandError

from modules.ollama_framework.deployment.install_ollama import install_ollama, PROJECT_ROOT


class Command(BaseCommand):
    help = 'Устанавливает Ollama в virtual_env/packages/ollama (обёртка над deployment-скриптом)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Переустановить Ollama, даже если он уже установлен',
        )
        parser.add_argument(
            '--refresh',
            action='store_true',
            help='Скачать архив заново, игнорируя кэш',
        )

    def handle(self, *args, **options):
        code = install_ollama(
            PROJECT_ROOT,
            force=options.get('force', False),
            refresh=options.get('refresh', False),
        )
        if code != 0:
            raise CommandError('Установка Ollama завершилась с ошибкой')
