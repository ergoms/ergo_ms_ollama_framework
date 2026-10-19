"""Эмбеддинги снимка Hugging Face (org/name), без Ollama /api/embed."""
from __future__ import annotations

import logging
import threading
from typing import List

from src.core.utils.huggingface_snapshot import ensure_local_source

from .settings import OLLAMA_COMPUTE_DEVICE

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()
_ST_MODEL = None
_ST_MODEL_NAME: str | None = None
_ST_DEVICE: str | None = None


class SnapshotEmbeddingsError(Exception):
    """Снимок Hugging Face не загрузился или не посчитал векторы."""


def _resolve_device(requested: str) -> str:
    requested = (requested or 'auto').strip().lower()
    if requested == 'cpu':
        return 'cpu'
    if requested in ('cuda', 'gpu'):
        try:
            import torch
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return 'cpu'
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:
        return 'cpu'


def _sentence_transformer(model_name: str):
    global _ST_MODEL, _ST_MODEL_NAME, _ST_DEVICE

    name = (model_name or '').strip()
    if not name:
        raise SnapshotEmbeddingsError('Не задано имя снимка Hugging Face')
    resolved_device = _resolve_device(OLLAMA_COMPUTE_DEVICE)

    with _load_lock:
        if (
            _ST_MODEL is not None
            and _ST_MODEL_NAME == name
            and _ST_DEVICE == resolved_device
        ):
            return _ST_MODEL

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise SnapshotEmbeddingsError(
                'Пакет sentence-transformers не установлен. '
                'Выполните: ergoms python-install'
            ) from exc

        source = ensure_local_source(name)
        logger.info(
            'Эмбеддинги %s считаются снимком Hugging Face на %s, не через Ollama',
            name,
            resolved_device,
        )
        try:
            _ST_MODEL = SentenceTransformer(source, device=resolved_device)
        except Exception as exc:
            if resolved_device != 'cpu':
                logger.warning('Снимок embeddings на CPU: %s', exc)
                _ST_MODEL = SentenceTransformer(source, device='cpu')
                resolved_device = 'cpu'
            else:
                raise SnapshotEmbeddingsError(str(exc)) from exc

        _ST_MODEL_NAME = name
        _ST_DEVICE = resolved_device
        return _ST_MODEL


def embed_snapshot(texts: List[str], *, model_name: str) -> List[List[float]]:
    """Векторы deepvk/USER-bge-m3 и других снимков org/name."""
    cleaned = [str(item or '').strip() for item in texts]
    if not cleaned:
        return []

    model = _sentence_transformer(model_name)
    try:
        vectors = model.encode(
            cleaned,
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise SnapshotEmbeddingsError(str(exc)) from exc

    rows: List[List[float]] = []
    if getattr(vectors, 'ndim', None) == 1:
        vectors = [vectors]
    for item in vectors:
        raw = item.tolist() if hasattr(item, 'tolist') else item
        rows.append([float(value) for value in raw])
    return rows
