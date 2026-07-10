import json
import logging

from django.http import StreamingHttpResponse
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from .services.runtime import run_chat, run_embed, run_generate, run_health, run_list_models

logger = logging.getLogger(__name__)


def _config_from_request(data: dict) -> tuple:
    config = data.get('config')
    skip_env = bool(data.get('skip_env_injection', False))
    return config, skip_env


class OllamaStatusView(APIView):
    """GET /api/ollama_framework/status/ — health Ollama и процесса."""

    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        config = request.query_params.get('config')
        parsed_config = json.loads(config) if config else None
        skip_env = request.query_params.get('skip_env_injection', '').lower() in ('1', 'true', 'yes')
        result = run_health(config=parsed_config, skip_env_injection=skip_env)
        return Response(result)


class OllamaModelsView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        config = request.query_params.get('config')
        parsed_config = json.loads(config) if config else None
        skip_env = request.query_params.get('skip_env_injection', '').lower() in ('1', 'true', 'yes')
        models = run_list_models(config=parsed_config, skip_env_injection=skip_env)
        return Response({'result': models})


class OllamaGenerateView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data or {}
        config, skip_env = _config_from_request(data)
        try:
            text = run_generate(
                data.get('prompt', ''),
                config=config,
                skip_env_injection=skip_env,
                system=data.get('system'),
                num_predict=data.get('num_predict'),
                temperature=float(data.get('temperature', 0.7)),
                stream=bool(data.get('stream', False)),
            )
            return Response({'result': text})
        except Exception as exc:
            logger.exception('ollama_framework generate failed')
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)


class OllamaChatView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data or {}
        config, skip_env = _config_from_request(data)
        messages = data.get('messages') or []
        stream = bool(data.get('stream', False))

        if stream:
            def event_stream():
                chunks = []

                def on_chunk(text: str) -> None:
                    chunks.append(text)

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
                    yield json.dumps({'done': True, 'result': ''.join(chunks)}, ensure_ascii=False)
                except Exception as exc:
                    yield json.dumps({'error': str(exc)}, ensure_ascii=False)

            return StreamingHttpResponse(event_stream(), content_type='application/json')

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
        except Exception as exc:
            logger.exception('ollama_framework chat failed')
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)


class OllamaEmbedView(APIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        data = request.data or {}
        config, skip_env = _config_from_request(data)
        texts = data.get('texts') or []
        try:
            vectors = run_embed(
                texts,
                config=config,
                skip_env_injection=skip_env,
                model=data.get('model'),
            )
            return Response({'result': vectors})
        except Exception as exc:
            logger.exception('ollama_framework embed failed')
            return Response({'error': str(exc)}, status=status.HTTP_502_BAD_GATEWAY)
