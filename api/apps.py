"""
Конфигурация Django приложения для Ollama Framework
"""

from django.apps import AppConfig


class OllamaFrameworkConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'modules.ollama_framework.api'
    label = 'ollama_framework'
    verbose_name = 'Ollama Framework'

    def ready(self):
        from django.conf import settings

        from . import integrations  # noqa: F401

        # ScopedRateThrottle для generate/chat/embed (см. api/views.py).
        rates = getattr(settings, 'REST_FRAMEWORK', {}).setdefault('DEFAULT_THROTTLE_RATES', {})
        rates.setdefault('ollama_llm', '30/minute')
