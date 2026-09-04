from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, Dict, Iterator, List, Optional

import httpx

from .base import BaseLLMClient, LLMClientError

logger = logging.getLogger(__name__)

_MODELS_CACHE_TTL_SECONDS = 60.0


def _pick_family(requested: str, models: List[str]) -> str:
    base = requested.split(':', 1)[0]
    same_family = [item for item in models if item == base or item.startswith(f'{base}:')]
    if not same_family:
        return ''
    for candidate in same_family:
        if candidate == base or candidate.endswith(':latest'):
            return candidate
    return same_family[0]


def _looks_like_json_object(text: str) -> bool:
    stripped = (text or '').strip()
    if stripped.startswith('```'):
        stripped = stripped.split('\n', 1)[-1].lstrip()
    return stripped.startswith('{') or stripped.startswith('[')


def _assistant_text(message: Any) -> str:
    """Текст ответа: content, иначе thinking (gpt-oss при format=json часто пустой content)."""
    if not isinstance(message, dict):
        return ''
    content = str(message.get('content') or '').strip()
    thinking = str(message.get('thinking') or '').strip()
    if content and _looks_like_json_object(content):
        return content
    if thinking and _looks_like_json_object(thinking):
        return thinking
    if content:
        return content
    return thinking


def resolve_ollama_model(requested: Optional[str], available: List[str]) -> str:
    """Подбирает установленную модель: точное имя или семейство (name / name:latest)."""
    models = [item for item in available if isinstance(item, str) and item]
    if not models:
        return requested or ''
    if not requested:
        return models[0]
    if requested in models:
        return requested
    family = _pick_family(requested, models)
    if family:
        return family
    return requested


class HttpxOllamaClient(BaseLLMClient):
    """HTTP-клиент Ollama API (httpx)."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        request_timeout: float,
        stream_timeout: float,
        concurrency_limit: int,
        max_retries: int,
        keep_alive: str,
        device_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(model=model)
        self._base_url = base_url.rstrip('/')
        self._max_retries = max_retries
        self._device_config = device_config or {}
        self._model_resolved = False
        self._resolved_embed_models: Dict[str, str] = {}
        self._models_cache: Optional[List[str]] = None
        self._models_cache_at = 0.0

        limits = httpx.Limits(
            max_connections=concurrency_limit,
            max_keepalive_connections=concurrency_limit,
        )
        timeout = httpx.Timeout(
            timeout=request_timeout,
            connect=min(10.0, request_timeout),
            write=request_timeout,
            read=max(stream_timeout, request_timeout * 2),
        )
        from src.core.utils.http_proxy import http_trust_env

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            limits=limits,
            trust_env=http_trust_env(),
        )
        self._keep_alive = keep_alive

    def get_provider_name(self) -> str:
        return 'ollama'

    def _invalidate_models_cache(self) -> None:
        self._models_cache = None
        self._models_cache_at = 0.0
        self._model_resolved = False
        self._resolved_embed_models.clear()

    def _fetch_model_names(self) -> List[str]:
        now = time.monotonic()
        if (
            self._models_cache is not None
            and (now - self._models_cache_at) < _MODELS_CACHE_TTL_SECONDS
        ):
            return self._models_cache
        try:
            resp = self._client.get('/api/tags', timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get('name', '') for m in data.get('models', []) if m.get('name')]
            self._models_cache = models
            self._models_cache_at = now
            return models
        except Exception:
            self._invalidate_models_cache()
            raise

    def _ensure_resolved_model(self) -> None:
        if self._model_resolved:
            return
        models = self.list_models()
        resolved = resolve_ollama_model(self.model, models)
        if resolved and resolved != self.model:
            logger.info(
                'Модель %r не найдена среди установленных, используем %r',
                self.model,
                resolved,
            )
            self.model = resolved
        self._model_resolved = True

    def _resolve_embed_model(self, requested: str) -> str:
        cached = self._resolved_embed_models.get(requested)
        if cached is not None:
            return cached
        available = self.list_models()
        resolved = resolve_ollama_model(requested, available) or requested
        if resolved != requested:
            logger.info(
                'Embed-модель %r не найдена среди установленных, используем %r',
                requested,
                resolved,
            )
        self._resolved_embed_models[requested] = resolved
        return resolved

    def check_health(self) -> Dict[str, Any]:
        try:
            models = self._fetch_model_names()
            resolved = resolve_ollama_model(self.model, models)
            model_loaded = bool(
                resolved
                and (
                    resolved in models
                    or any(resolved == m or m.startswith(f'{resolved}:') for m in models)
                )
            )
            return {
                'available': True,
                'models': models,
                'model': resolved or self.model,
                'model_loaded': model_loaded,
                'base_url': self._base_url,
            }
        except httpx.TimeoutException:
            return {'available': False, 'error': 'Таймаут подключения к Ollama', 'base_url': self._base_url}
        except httpx.ConnectError:
            return {'available': False, 'error': 'Не удалось подключиться к Ollama', 'base_url': self._base_url}
        except Exception as exc:
            return {'available': False, 'error': str(exc), 'base_url': self._base_url}

    def list_models(self) -> List[str]:
        try:
            return self._fetch_model_names()
        except Exception:
            return []

    def _build_generate_payload(
        self,
        prompt: str,
        *,
        num_predict: Optional[int],
        temperature: float,
        stream: bool,
        system: Optional[str],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            'model': self.model,
            'prompt': prompt,
            'stream': stream,
            'keep_alive': self._keep_alive,
            'options': {
                'temperature': temperature,
                'top_k': 40,
                'top_p': 0.9,
            },
        }
        if system:
            payload['system'] = system
        if num_predict is not None:
            payload['options']['num_predict'] = num_predict
        if self._device_config:
            payload['options'].update(self._device_config)
        return payload

    def complete(
        self,
        prompt: str,
        *,
        num_predict: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        system: Optional[str] = None,
        format: Optional[Any] = None,
        return_stats: bool = False,
    ) -> str | tuple[str, Dict[str, Any]]:
        payload = self._build_generate_payload(
            prompt,
            num_predict=num_predict,
            temperature=temperature,
            stream=stream,
            system=system,
        )
        if format is not None:
            payload['format'] = (
                {'type': 'object'} if format == 'json' else format
            )
        self._ensure_resolved_model()
        payload['model'] = self.model

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    text = self._stream_generate(payload, stream_callback)
                    return (text, {}) if return_stats else text
                return self._complete_generate(payload, return_stats=return_stats)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    'Ошибка Ollama generate (попытка %s/%s): %s',
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise LLMClientError('Ollama API недоступен') from last_error

    def _complete_generate(self, payload: Dict[str, Any], return_stats: bool = False) -> str | tuple[str, Dict[str, Any]]:
        try:
            response = self._client.post('/api/generate', json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get('response', '')
            if return_stats:
                return content, {
                    'prompt_eval_count': data.get('prompt_eval_count'),
                    'eval_count': data.get('eval_count'),
                }
            return content
        except httpx.HTTPStatusError as exc:
            raise self._http_error('generate', exc) from exc

    def _stream_generate(
        self,
        payload: Dict[str, Any],
        stream_callback: Optional[Callable[[str], None]],
    ) -> str:
        chunks: List[str] = []
        try:
            with self._client.stream('POST', '/api/generate', json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    text = data.get('response')
                    if text:
                        chunks.append(text)
                        if stream_callback:
                            stream_callback(text)
                    if data.get('done'):
                        break
        except httpx.HTTPStatusError as exc:
            raise self._http_error('generate', exc) from exc
        return ''.join(chunks)

    def complete_stream_iter(
        self,
        prompt: str,
        *,
        num_predict: Optional[int] = None,
        temperature: float = 0.7,
        system: Optional[str] = None,
    ):
        payload = self._build_generate_payload(
            prompt,
            num_predict=num_predict,
            temperature=temperature,
            stream=True,
            system=system,
        )
        self._ensure_resolved_model()
        payload['model'] = self.model
        with self._client.stream('POST', '/api/generate', json=payload) as response:
            response.raise_for_status()
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                text = data.get('response')
                if text:
                    yield text
                if data.get('done'):
                    break

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        format: Optional[Any] = None,
        return_stats: bool = False,
    ) -> str | tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'stream': stream,
            'keep_alive': self._keep_alive,
            'options': {'top_k': 40, 'top_p': 0.9},
        }
        if format is not None:
            # Строка format=json у gpt-oss даёт оборванный не-JSON; схема — нормальный объект.
            payload['format'] = (
                {'type': 'object'} if format == 'json' else format
            )
        if num_predict is not None:
            payload['options']['num_predict'] = num_predict
        if temperature is not None:
            payload['options']['temperature'] = temperature
        if seed is not None:
            payload['options']['seed'] = seed
        if self._device_config:
            extra = dict(self._device_config)
            if 'think' in extra:
                payload['think'] = extra.pop('think')
            payload['options'].update(extra)
        # gpt-oss игнорирует bool think; без уровня low рассуждение съедает num_predict,
        # и в content не остаётся JSON. false + схема даёт прозу вместо объекта.
        if format is not None:
            current = payload.get('think')
            if current in (None, True, False, 'true', 'false'):
                payload['think'] = 'low'

        self._ensure_resolved_model()
        payload['model'] = self.model

        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                if stream:
                    return self._stream_chat(payload, stream_callback, return_stats=return_stats)
                return self._complete_chat(payload, return_stats=return_stats)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    'Ошибка Ollama chat (попытка %s/%s): %s',
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise LLMClientError('Ollama API недоступен') from last_error

    def _complete_chat(self, payload: Dict[str, Any], return_stats: bool = False) -> str | tuple[str, Dict[str, Any]]:
        try:
            response = self._client.post('/api/chat', json=payload)
            response.raise_for_status()
            data = response.json()
            content = _assistant_text(data.get('message') or {})
            if return_stats:
                return content, {
                    'prompt_eval_count': data.get('prompt_eval_count'),
                    'eval_count': data.get('eval_count'),
                }
            return content
        except httpx.HTTPStatusError as exc:
            raise self._http_error('chat', exc) from exc

    def _stream_chat(
        self,
        payload: Dict[str, Any],
        stream_callback: Optional[Callable[[str], None]],
        return_stats: bool = False,
    ) -> str | tuple[str, Dict[str, Any]]:
        chunks: List[str] = []
        stats: Dict[str, Any] = {}
        try:
            with self._client.stream('POST', '/api/chat', json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    message = data.get('message', {})
                    text = _assistant_text(message) if isinstance(message, dict) else ''
                    if text:
                        chunks.append(text)
                        if stream_callback:
                            stream_callback(text)
                    if data.get('done'):
                        stats = {
                            'prompt_eval_count': data.get('prompt_eval_count'),
                            'eval_count': data.get('eval_count'),
                        }
                        break
        except httpx.HTTPStatusError as exc:
            raise self._http_error('chat', exc) from exc
        content = ''.join(chunks)
        if return_stats:
            return content, stats
        return content

    def chat_stream_iter(
        self,
        messages: List[Dict[str, Any]],
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Iterator[str]:
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'stream': True,
            'keep_alive': self._keep_alive,
            'options': {'top_k': 40, 'top_p': 0.9},
        }
        if num_predict is not None:
            payload['options']['num_predict'] = num_predict
        if temperature is not None:
            payload['options']['temperature'] = temperature
        if seed is not None:
            payload['options']['seed'] = seed
        if self._device_config:
            payload['options'].update(self._device_config)

        self._ensure_resolved_model()
        payload['model'] = self.model

        try:
            with self._client.stream('POST', '/api/chat', json=payload) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if not raw_line:
                        continue
                    try:
                        data = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    message = data.get('message', {})
                    text = message.get('content') if isinstance(message, dict) else None
                    if text:
                        yield text
                    if data.get('done'):
                        break
        except httpx.HTTPStatusError as exc:
            raise self._http_error('chat', exc) from exc

    @staticmethod
    def _parse_embed_response(data: Dict[str, Any]) -> List[List[float]]:
        embeddings = data.get('embeddings')
        if embeddings is None and 'embedding' in data:
            single = data['embedding']
            if isinstance(single, list) and single and isinstance(single[0], (int, float)):
                return [single]
            if isinstance(single, list):
                return [single]
        if embeddings is None:
            raise LLMClientError(
                f'Ollama embed: нет embeddings в ответе, ключи: {list(data.keys())}'
            )
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], (int, float)):
            return [embeddings]
        if not isinstance(embeddings, list):
            raise LLMClientError(
                f'Ollama embed: embeddings не список: {type(embeddings).__name__}'
            )
        return embeddings

    def _embed_error_message(self, *, model: str, status: int, body: str) -> str:
        body_l = (body or '').lower()
        if status == 501 or 'does not support embeddings' in body_l or '--embeddings' in body_l:
            return (
                f'Модель {model!r} не поддерживает embeddings. '
                f'Укажите embedding-модель (OLLAMA_EMBEDDINGS_MODEL), например embeddinggemma.'
            )
        detail = (body or '').strip()
        if detail:
            return f'Ollama embed API ошибка {status} (модель {model!r}): {detail[:300]}'
        return f'Ollama embed API ошибка {status} (модель {model!r})'

    def _embed_legacy_prompt(self, texts: List[str], *, model: str) -> List[List[float]]:
        """Legacy /api/embeddings (prompt) — для Ollama без /api/embed."""
        vectors: List[List[float]] = []
        for text in texts:
            response = self._client.post(
                '/api/embeddings',
                json={'model': model, 'prompt': text},
            )
            if response.status_code >= 400:
                raise LLMClientError(
                    self._embed_error_message(
                        model=model,
                        status=response.status_code,
                        body=self._response_body_text(response),
                    )
                )
            parsed = self._parse_embed_response(response.json())
            if not parsed:
                raise LLMClientError('Ollama /api/embeddings вернул пустой embedding')
            vectors.append(parsed[0])
        return vectors

    def embed(self, texts: List[str], *, model: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []

        requested = model or self.model
        embed_model = self._resolve_embed_model(requested)

        payload = {'model': embed_model, 'input': texts}
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post('/api/embed', json=payload)
                body = self._response_body_text(response)
                # 501 «does not support embeddings» — модель не embedding; fallback бесполезен.
                if response.status_code == 501 and (
                    'does not support embeddings' in body.lower()
                    or '--embeddings' in body.lower()
                ):
                    raise LLMClientError(
                        self._embed_error_message(
                            model=embed_model,
                            status=501,
                            body=body,
                        )
                    )
                if response.status_code in (404, 501):
                    logger.warning(
                        'Ollama /api/embed вернул %s, fallback на /api/embeddings',
                        response.status_code,
                    )
                    return self._embed_legacy_prompt(texts, model=embed_model)
                response.raise_for_status()
                return self._parse_embed_response(response.json())
            except LLMClientError:
                raise
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                body = self._response_body_text(exc.response) if exc.response is not None else ''
                if status == 501 and (
                    'does not support embeddings' in body.lower()
                    or '--embeddings' in body.lower()
                ):
                    raise LLMClientError(
                        self._embed_error_message(model=embed_model, status=501, body=body)
                    ) from exc
                if status in (404, 501):
                    logger.warning(
                        'Ollama /api/embed HTTP %s, fallback на /api/embeddings',
                        status,
                    )
                    return self._embed_legacy_prompt(texts, model=embed_model)
                if status is not None and 400 <= status < 500:
                    raise LLMClientError(
                        self._embed_error_message(
                            model=embed_model,
                            status=status,
                            body=body,
                        )
                    ) from exc
                last_error = exc
                logger.warning(
                    'Ошибка Ollama embed (попытка %s/%s): %s',
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                logger.warning(
                    'Ошибка Ollama embed (попытка %s/%s): %s',
                    attempt + 1,
                    self._max_retries + 1,
                    exc,
                )
        raise LLMClientError('Ollama embed API недоступен') from last_error

    @staticmethod
    def _response_body_text(response: httpx.Response) -> str:
        """Текст тела ответа; для stream=True нужен read() до .text."""
        try:
            return response.text
        except httpx.ResponseNotRead:
            try:
                response.read()
                return response.text
            except Exception:
                return ''
        except Exception:
            return ''

    def _http_error(self, endpoint: str, exc: httpx.HTTPStatusError) -> LLMClientError:
        if exc.response.status_code == 404:
            self._invalidate_models_cache()
        body = self._response_body_text(exc.response)
        if exc.response.status_code == 404:
            error_msg = (
                f'Ollama API вернул 404 для /api/{endpoint}.\n'
                f'URL: {self._base_url}/api/{endpoint}\n'
                f"Модель '{self.model}' существует? Выполните: ergoms ollama_framework:ollama --list\n"
                f'Ответ сервера: {body}'
            )
            return LLMClientError(error_msg)
        if body:
            return LLMClientError(f'{exc}; тело: {body}')
        return LLMClientError(str(exc))
