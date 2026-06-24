"""
Django-команда для остановки Ollama сервера.

Пример: ergoms ollama_framework:stop-ollama
"""

import logging

from django.core.management.base import BaseCommand

from modules.ollama_framework.api.ollama_process import find_ollama

logger = logging.getLogger('modules.ollama_framework.commands')


class Command(BaseCommand):
    help = 'Останавливает Ollama сервер'

    def handle(self, *args: tuple, **options: dict) -> None:
        logger.info('Запуск команды stop_ollama')
        process = find_ollama(include_wrapper=True)

        if not process:
            msg = 'Ollama процесс не найден'
            logger.warning(msg)
            self.stdout.write(self.style.WARNING(msg))
            return

        try:
            logger.info('Остановка Ollama (PID: %s)', process.pid)
            process.terminate()
            process.wait(timeout=5)
            msg = f'Ollama процесс (PID: {process.pid}) успешно остановлен'
            logger.info(msg)
            self.stdout.write(self.style.SUCCESS(msg))
        except Exception as exc:
            msg = f'Ошибка при остановке Ollama: {exc}'
            logger.error(msg)
            self.stdout.write(self.style.ERROR(msg))
