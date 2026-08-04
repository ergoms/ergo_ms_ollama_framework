"""
Настройки модуля ollama_framework — единый источник для Ollama и транспорта.
"""
from src.config.env import env

OLLAMA_BASE_URL = env.str('OLLAMA_BASE_URL', default='http://127.0.0.1:11434')
OLLAMA_DEFAULT_MODEL = env.str('OLLAMA_DEFAULT_MODEL', default='mistral:latest')
OLLAMA_EMBEDDINGS_MODEL = env.str('OLLAMA_EMBEDDINGS_MODEL', default='embeddinggemma')

OLLAMA_REQUEST_TIMEOUT = env.float('OLLAMA_REQUEST_TIMEOUT', default=180.0)
OLLAMA_STREAM_TIMEOUT = env.float('OLLAMA_STREAM_TIMEOUT', default=300.0)
OLLAMA_CONCURRENCY_LIMIT = env.int('OLLAMA_CONCURRENCY_LIMIT', default=8)
OLLAMA_MAX_RETRIES = env.int('OLLAMA_MAX_RETRIES', default=2)
OLLAMA_KEEP_ALIVE = env.str('OLLAMA_KEEP_ALIVE', default='10m')

# local — ModuleBridge in-process; http — REST API ollama_framework
OLLAMA_FRAMEWORK_TRANSPORT = env.str('OLLAMA_FRAMEWORK_TRANSPORT', default='local')
OLLAMA_FRAMEWORK_API_BASE = env.str('OLLAMA_FRAMEWORK_API_BASE', default='')
