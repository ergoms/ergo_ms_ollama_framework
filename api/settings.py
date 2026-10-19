"""
Настройки модуля ollama_framework — единый источник для Ollama и транспорта.
"""
from src.config.env import env

_OLLAMA_EMBEDDINGS_DEFAULT = 'embeddinggemma'


def _normalize_compute_device(value: str, *, default: str = 'auto') -> str:
    normalized = (value or '').strip().lower()
    if normalized in ('gpu', 'cpu', 'auto'):
        return normalized
    return default


def _configured_base_url() -> str:
    """Адрес serve из .env этого модуля.

    Общее слияние modules/**/.env оставляет последний файл. Чужой модуль
    переписывает адрес на 127.0.0.1, хотя serve слушает свой OLLAMA_BASE_URL.
    """
    from ..deployment.paths import module_env_value

    raw = module_env_value('OLLAMA_BASE_URL')
    if not raw:
        raw = env.str('OLLAMA_BASE_URL', default='http://127.0.0.1:11434')
    raw = (raw or '').strip()
    if raw and '://' not in raw:
        raw = f'http://{raw}'
    return raw.rstrip('/') or 'http://127.0.0.1:11434'


OLLAMA_BASE_URL = _configured_base_url()
OLLAMA_DEFAULT_MODEL = env.str('OLLAMA_DEFAULT_MODEL', default='ministral-3:8b')
# Имя библиотеки Ollama или снимок Hugging Face org/name.
# org/name считается sentence-transformers, без подмены на embeddinggemma.
OLLAMA_EMBEDDINGS_MODEL = env.str(
    'OLLAMA_EMBEDDINGS_MODEL',
    default=_OLLAMA_EMBEDDINGS_DEFAULT,
).strip() or _OLLAMA_EMBEDDINGS_DEFAULT

OLLAMA_REQUEST_TIMEOUT = env.float('OLLAMA_REQUEST_TIMEOUT', default=180.0)
OLLAMA_STREAM_TIMEOUT = env.float('OLLAMA_STREAM_TIMEOUT', default=300.0)
OLLAMA_CONCURRENCY_LIMIT = env.int('OLLAMA_CONCURRENCY_LIMIT', default=8)
OLLAMA_MAX_RETRIES = env.int('OLLAMA_MAX_RETRIES', default=2)
OLLAMA_KEEP_ALIVE = env.str('OLLAMA_KEEP_ALIVE', default='10m')

# auto — Ollama сам выбирает GPU/CPU; gpu — num_gpu=-1; cpu — num_gpu=0
OLLAMA_COMPUTE_DEVICE = _normalize_compute_device(
    env.str('OLLAMA_COMPUTE_DEVICE', default='auto'),
)

# local — ModuleBridge in-process; http — REST API ollama_framework
OLLAMA_FRAMEWORK_TRANSPORT = env.str('OLLAMA_FRAMEWORK_TRANSPORT', default='local')
OLLAMA_FRAMEWORK_API_BASE = env.str('OLLAMA_FRAMEWORK_API_BASE', default='')
