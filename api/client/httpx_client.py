from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from .base import BaseLLMClient, LLMClientError

logger = logging.getLogger(__name__)


def resolve_ollama_model(requested: Optional[str], available: List[str]) -> str:
    """Подбирает установленную модель: точное имя или то же семейство (mistral:7b → mistral:latest)."""
    models = [m for m in available if isinstance(m, str) and m]
    if not models:
        return requested or ''
    if not requested:
        return models[0]
    if requested in models:
        return requested

    base = requested.split(':', 1)[0]
    same_family = [m for m in models if m == base or m.startswith(f'{base}:')]
    if same_family:
        for candidate in same_family:
            if candidate == base or candidate.endswith(':latest'):
                return candidate
        return same_family[0]
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
        self._client = httpx.Client(base_url=self._base_url, timeout=timeout, limits=limits)
        self._keep_alive = keep_alive

    def get_provider_name(self) -> str:
        return 'ollama'

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

    def check_health(self) -> Dict[str, Any]:
        try:
            resp = self._client.get('/api/tags', timeout=5.0)
            resp.raise_for_status()
            data = resp.json()
            models = [m.get('name', '') for m in data.get('models', [])]
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
        health = self.check_health()
        return health.get('models', [])

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
            payload['format'] = format
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
        messages: List[Dict[str, str]],
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
            payload['format'] = format
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
            content = data.get('message', {}).get('content', '')
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
                    text = message.get('content') if isinstance(message, dict) else None
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

    def embed(self, texts: List[str], *, model: Optional[str] = None) -> List[List[float]]:
        embed_model = model or self.model
        payload = {'model': embed_model, 'input': texts}
        last_error: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.post('/api/embed', json=payload)
                response.raise_for_status()
                data = response.json()
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
                return embeddings
            except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
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
