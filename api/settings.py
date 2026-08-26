"""
Настройки модуля ollama_framework — единый источник для Ollama и транспорта.
"""
import logging

from src.config.env import env
from src.core.utils.huggingface_snapshot import is_huggingface_repo_id

logger = logging.getLogger(__name__)

_OLLAMA_EMBEDDINGS_DEFAULT = 'embeddinggemma'


def _normalize_compute_device(value: str, *, default: str = 'auto') -> str:
    normalized = (value or '').strip().lower()
    if normalized in ('gpu', 'cpu', 'auto'):
        return normalized
    return default


OLLAMA_BASE_URL = env.str('OLLAMA_BASE_URL', default='http://127.0.0.1:11434')
OLLAMA_DEFAULT_MODEL = env.str('OLLAMA_DEFAULT_MODEL', default='mistral:latest')
_raw_embeddings_model = env.str(
    'OLLAMA_EMBEDDINGS_MODEL',
    default=_OLLAMA_EMBEDDINGS_DEFAULT,
)
if is_huggingface_repo_id(_raw_embeddings_model):
    logger.warning(
        'OLLAMA_EMBEDDINGS_MODEL=%s — снимок Hugging Face, для /api/embed используется %s. '
        'Исправьте modules/ollama_framework/.env (имя из библиотеки Ollama).',
        _raw_embeddings_model,
        _OLLAMA_EMBEDDINGS_DEFAULT,
    )
    OLLAMA_EMBEDDINGS_MODEL = _OLLAMA_EMBEDDINGS_DEFAULT
else:
    OLLAMA_EMBEDDINGS_MODEL = _raw_embeddings_model

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
