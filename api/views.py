import json
import logging
from queue import Empty, Queue
from threading import Thread

from django.http import StreamingHttpResponse
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.throttling import ScopedRateThrottle, UserRateThrottle
from rest_framework.views import APIView

from .config_security import gateway_error_message, parse_config_param, sanitize_client_config
from .payload_limits import (
    validate_chat_messages,
    validate_embed_texts,
    validate_prompt_payload,
)
from .permissions import CanUseOllamaLlm, CanViewOllamaFramework
from .services.runtime import run_chat, run_embed, run_generate, run_health, run_list_models

logger = logging.getLogger(__name__)


class OllamaLlmThrottle(ScopedRateThrottle):
    """Отдельный лимит на дорогие LLM-эндпоинты (generate/chat/embed)."""

    scope = 'ollama_llm'


def _config_from_request(data: dict, user) -> tuple:
    raw_config = data.get('config')
    parsed = parse_config_param(raw_config) if raw_config is not None else None
    config = sanitize_client_config(parsed, user)
    skip_env = bool(data.get('skip_env_injection', False))
    return config, skip_env


class OllamaStatusView(APIView):
    """GET /api/ollama_framework/status/ — health Ollama и процесса."""

    permission_classes = [permissions.IsAuthenticated, CanViewOllamaFramework]

    def get(self, request):
        parsed_config = parse_config_param(request.query_params.get('config'))
        config = sanitize_client_config(parsed_config, request.user)
        skip_env = request.query_params.get('skip_env_injection', '').lower() in ('1', 'true', 'yes')
        result = run_health(config=config, skip_env_injection=skip_env)
        return Response(result)


class OllamaModelsView(APIView):
    permission_classes = [permissions.IsAuthenticated, CanViewOllamaFramework]

    def get(self, request):
        parsed_config = parse_config_param(request.query_params.get('config'))
        config = sanitize_client_config(parsed_config, request.user)
        skip_env = request.query_params.get('skip_env_injection', '').lower() in ('1', 'true', 'yes')
        models = run_list_models(config=config, skip_env_injection=skip_env)
        return Response({'result': models})


class OllamaGenerateView(APIView):
    permission_classes = [permissions.IsAuthenticated, CanUseOllamaLlm]
    throttle_classes = [UserRateThrottle, OllamaLlmThrottle]
    throttle_scope = 'ollama_llm'

    def post(self, request):
        data = request.data or {}
        err = validate_prompt_payload(data.get('prompt', ''), system=data.get('system'))
        if err:
            return Response({'error': err}, status=status.HTTP_400_BAD_REQUEST)
        if bool(data.get('stream', False)):
            return Response(
                {
                    'error': (
                        'Streaming для generate через REST не поддерживается. '
                        'Используйте ModuleBridge ollama_framework.generate '
                        'со stream_callback или create_client.'
                    ),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        config, skip_env = _config_from_request(data, request.user)
        try:
            text = run_generate(
                data.get('prompt', ''),
                config=config,
                skip_env_injection=skip_env,
                system=data.get('system'),
                num_predict=data.get('num_predict'),
                temperature=float(data.get('temperature', 0.7)),
            )
            return Response({'result': text})
        except Exception:
            logger.exception('ollama_framework generate failed')
            return Response({'error': gateway_error_message()}, status=status.HTTP_502_BAD_GATEWAY)


class OllamaChatView(APIView):
    permission_classes = [permissions.IsAuthenticated, CanUseOllamaLlm]
    throttle_classes = [UserRateThrottle, OllamaLlmThrottle]
    throttle_scope = 'ollama_llm'

    def post(self, request):
        data = request.data or {}
        messages = data.get('messages') or []
        err = validate_chat_messages(messages)
        if err:
            return Response({'error': err}, status=status.HTTP_400_BAD_REQUEST)
        config, skip_env = _config_from_request(data, request.user)
        stream = bool(data.get('stream', False))

        if stream:
            chunk_queue: Queue = Queue()
            sentinel = object()

            def on_chunk(text: str) -> None:
                chunk_queue.put(text)

            def worker() -> None:
                try:
                    run_chat(
                        messages,
                        config=config,
                        skip_env_injection=skip_env,
                        num_predict=data.get('num_predict'),
                        temperature=data.get('temperature'),
                        seed=data.get('seed'),
                        stream=True,
                        stream_callback=on_chunk,
                        format=data.get('format'),
                    )
                    chunk_queue.put(sentinel)
                except Exception:
                    logger.exception('ollama_framework chat stream failed')
                    chunk_queue.put({'__error__': gateway_error_message()})
                    chunk_queue.put(sentinel)

            def event_stream():
                Thread(target=worker, daemon=True).start()
                while True:
                    try:
                        item = chunk_queue.get(timeout=300)
                    except Empty:
                        yield json.dumps({'error': gateway_error_message()}, ensure_ascii=False) + '\n'
                        break
                    if item is sentinel:
                        yield json.dumps({'done': True}, ensure_ascii=False) + '\n'
                        break
                    if isinstance(item, dict) and '__error__' in item:
                        yield json.dumps({'error': item['__error__']}, ensure_ascii=False) + '\n'
                        break
                    yield json.dumps({'chunk': item}, ensure_ascii=False) + '\n'

            return StreamingHttpResponse(event_stream(), content_type='application/x-ndjson')

        try:
            text = run_chat(
                messages,
                config=config,
                skip_env_injection=skip_env,
                num_predict=data.get('num_predict'),
                temperature=data.get('temperature'),
                seed=data.get('seed'),
                format=data.get('format'),
            )
            return Response({'result': text})
        except Exception:
            logger.exception('ollama_framework chat failed')
            return Response({'error': gateway_error_message()}, status=status.HTTP_502_BAD_GATEWAY)


class OllamaEmbedView(APIView):
    permission_classes = [permissions.IsAuthenticated, CanUseOllamaLlm]
    throttle_classes = [UserRateThrottle, OllamaLlmThrottle]
    throttle_scope = 'ollama_llm'

    def post(self, request):
        data = request.data or {}
        texts = data.get('texts') or []
        err = validate_embed_texts(texts)
        if err:
            return Response({'error': err}, status=status.HTTP_400_BAD_REQUEST)
        config, skip_env = _config_from_request(data, request.user)
        try:
            vectors = run_embed(
                texts,
                config=config,
                skip_env_injection=skip_env,
                model=data.get('model'),
            )
            return Response({'result': vectors})
        except Exception:
            logger.exception('ollama_framework embed failed')
            return Response({'error': gateway_error_message()}, status=status.HTTP_502_BAD_GATEWAY)
