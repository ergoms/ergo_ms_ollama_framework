"""
Методы CLI Ollama — re-export django-free ops.

Runtime LLM — через modules.ollama_framework.api.client / transport.
"""

from modules.ollama_framework.deployment.ollama_ops import OllamaOps

# Совместимость со старым именем
OllamaMethods = OllamaOps

__all__ = ['OllamaMethods', 'OllamaOps']
