"""Structured output preserves the thinking mode of non-gpt-oss models."""
import json
import unittest
import httpx
from modules.ollama_framework.api.client.httpx_client import HttpxOllamaClient


class StructuredChatThinkingTests(unittest.TestCase):
    def check_payload(self, model, think, expected, *, stream=False, local_only=False):
        payloads = []
        def handler(request):
            if request.url.path == '/api/tags':
                return httpx.Response(200, json={'models': [{'name': model}]})
            if request.url.path == '/api/show':
                return httpx.Response(200, json={'model_info': {'general.architecture': 'test'}})
            self.assertEqual(request.url.path, '/api/chat')
            payloads.append(json.loads(request.content))
            if stream:
                return httpx.Response(200, content=b'{"message":{"content":"{}"},"done":true}\n')
            return httpx.Response(200, json={'message': {'content': '{}'}})
        base_url = 'http://127.0.0.1:11434' if local_only else 'http://ollama.test'
        client = HttpxOllamaClient(model=model, base_url=base_url, request_timeout=1,
            stream_timeout=1, concurrency_limit=1, max_retries=0, keep_alive='1m',
            device_config={} if think is None else {'think': think}, local_only=local_only)
        client._client.close()
        client._client = httpx.Client(base_url=base_url, transport=httpx.MockTransport(handler))
        self.addCleanup(client._client.close)
        result = client.chat([{'role': 'user', 'content': 'Return JSON'}],
            format={'type': 'object'}, stream=stream)
        self.assertEqual(result, '{}')
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]['format'], {'type': 'object'})
        if expected is None:
            self.assertNotIn('think', payloads[0])
        else:
            self.assertEqual(payloads[0]['think'], expected)

    def test_qwen_explicit_false_is_preserved(self):
        self.check_payload('qwen3.5:4b', False, False)

    def test_qwen_unspecified_mode_is_not_invented(self):
        self.check_payload('qwen3.5:4b', None, None)

    def test_qwen_streaming_preserves_false(self):
        self.check_payload('qwen3.5:4b', False, False, stream=True)

    def test_gpt_oss_keeps_low_workaround(self):
        self.check_payload('gpt-oss:20b', False, 'low')

    def test_namespaced_gpt_oss_keeps_low_workaround(self):
        self.check_payload('library/gpt-oss:20b', False, 'low')

    def test_explicit_gpt_oss_effort_is_preserved(self):
        self.check_payload('gpt-oss:20b', 'high', 'high')

    def test_strict_local_gpt_oss_preserves_false(self):
        self.check_payload('gpt-oss:20b', False, False, local_only=True)
