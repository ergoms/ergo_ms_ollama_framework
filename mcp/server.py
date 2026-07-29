"""
MCP сервер модуля ollama_framework.
Инструменты для запросов к Ollama, списка моделей, статуса и загрузки моделей.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

import httpx

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from modules.ollama_framework.deployment.paths import (  # noqa: E402
    get_default_model,
    get_ollama_base_url,
    get_ollama_executable,
)
from modules.ollama_framework.deployment.process import (  # noqa: E402
    find_ollama,
    is_ollama_server_available,
    start_ollama_background as deployment_start_ollama_background,
)

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent


OLLAMA_BASE_URL = None
OLLAMA_DEFAULT_MODEL = None

server = Server('ergo-module-ollama-framework')

PULL_HINT = 'ergoms ollama_framework:ollama --pull'


def find_ollama_process() -> Optional[Any]:
    return find_ollama()


def start_ollama_background() -> bool:
    if get_ollama_executable(PROJECT_DIR) is None:
        print(
            '[ERROR] Ollama не найден в virtual_env/packages/ollama. '
            'Установите: ergoms ollama_framework:install-ollama',
            file=sys.stderr,
        )
        return False
    return deployment_start_ollama_background()


async def ensure_ollama_running(base_url: str) -> bool:
    if is_ollama_server_available():
        return True

    print('[INFO] Ollama не запущен. Запускаю portable…', file=sys.stderr)
    if not await asyncio.to_thread(start_ollama_background):
        print(
            '[ERROR] Не удалось запустить Ollama. '
            'Проверьте: ergoms ollama_framework:install-ollama '
            'и ergoms ollama_framework:start-ollama',
            file=sys.stderr,
        )
        return False

    print('[INFO] Ожидание запуска Ollama…', file=sys.stderr)
    for i in range(30):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                response = await client.get(f'{base_url}/api/tags')
                if response.status_code == 200:
                    print('[OK] Ollama готов к работе', file=sys.stderr)
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
        if (i + 1) % 5 == 0:
            print(f'[INFO] ... ещё {30 - i - 1} секунд', file=sys.stderr)
    print('[ERROR] Ollama не стал доступен за отведённое время', file=sys.stderr)
    return False


async def initialize_config():
    global OLLAMA_BASE_URL, OLLAMA_DEFAULT_MODEL
    OLLAMA_BASE_URL = get_ollama_base_url()
    OLLAMA_DEFAULT_MODEL = get_default_model()
    is_running = await ensure_ollama_running(OLLAMA_BASE_URL)
    if is_running:
        print('[OK] MCP ollama_framework инициализирован', file=sys.stderr)
        print(f'  - Ollama URL: {OLLAMA_BASE_URL}', file=sys.stderr)
        print(f'  - Модель по умолчанию: {OLLAMA_DEFAULT_MODEL}', file=sys.stderr)
    else:
        print('[WARNING] MCP ollama_framework инициализирован, Ollama недоступен', file=sys.stderr)
        print(f'  - Ollama URL: {OLLAMA_BASE_URL}', file=sys.stderr)
        print(
            '  - Запуск: ergoms ollama_framework:start-ollama '
            '(или автоматически при первом запросе)',
            file=sys.stderr,
        )


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name='ollama_generate',
            description='Отправляет запрос к Ollama модели и получает ответ. Автоматически запускает Ollama если он не запущен.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'prompt': {'type': 'string', 'description': 'Текст запроса к модели'},
                    'model': {'type': 'string', 'description': 'Название модели Ollama', 'default': 'mistral'},
                    'temperature': {'type': 'number', 'description': 'Температура (0.0–1.0)', 'default': 0.7},
                    'max_tokens': {'type': 'integer', 'description': 'Макс. токенов в ответе', 'default': 2048},
                },
                'required': ['prompt'],
            },
        ),
        Tool(
            name='ollama_list_models',
            description='Получает список установленных моделей Ollama',
            inputSchema={'type': 'object', 'properties': {}},
        ),
        Tool(
            name='ollama_status',
            description='Проверяет статус Ollama сервера и возвращает информацию о доступных моделях',
            inputSchema={'type': 'object', 'properties': {}},
        ),
        Tool(
            name='ollama_pull_model',
            description='Скачивает модель Ollama. Может занять много времени.',
            inputSchema={
                'type': 'object',
                'properties': {
                    'model': {'type': 'string', 'description': 'Название модели (mistral, llama2, …)'},
                },
                'required': ['model'],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    if name in ('ollama_generate', 'ollama_list_models', 'ollama_status', 'ollama_pull_model'):
        base_url = OLLAMA_BASE_URL or 'http://localhost:11434'
        await ensure_ollama_running(base_url)

    if name == 'ollama_generate':
        return await handle_generate(arguments)
    if name == 'ollama_list_models':
        return await handle_list_models()
    if name == 'ollama_status':
        return await handle_status()
    if name == 'ollama_pull_model':
        return await handle_pull_model(arguments)
    raise ValueError(f'Неизвестный инструмент: {name}')


async def handle_generate(arguments: dict) -> list[TextContent]:
    prompt = arguments.get('prompt', '')
    model = arguments.get('model', OLLAMA_DEFAULT_MODEL)
    temperature = arguments.get('temperature', 0.7)
    max_tokens = arguments.get('max_tokens', 2048)

    if not prompt:
        return [TextContent(
            type='text',
            text=json.dumps({'error': "Параметр 'prompt' обязателен"}, ensure_ascii=False, indent=2),
        )]

    try:
        url = f'{OLLAMA_BASE_URL}/api/generate'
        payload = {
            'model': model,
            'prompt': prompt,
            'stream': False,
            'options': {'temperature': temperature, 'num_predict': max_tokens},
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = {
                'model': model,
                'response': data.get('response', ''),
                'done': data.get('done', False),
                'total_duration': data.get('total_duration', 0),
                'eval_count': data.get('eval_count', 0),
                'eval_duration': data.get('eval_duration', 0),
            }
            if result['eval_duration'] > 0 and result['eval_count'] > 0:
                result['tokens_per_second'] = round(
                    result['eval_count'] / (result['eval_duration'] / 1e9), 2,
                )
            return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False, indent=2))]
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            error_msg = {
                'error': f"Модель '{model}' не найдена",
                'suggestion': f'Установите модель: {PULL_HINT} {model}',
            }
        else:
            error_msg = {'error': f'Ошибка HTTP: {e.response.status_code}', 'details': e.response.text}
        return [TextContent(type='text', text=json.dumps(error_msg, ensure_ascii=False, indent=2))]
    except httpx.TimeoutException:
        return [TextContent(type='text', text=json.dumps({'error': 'Таймаут запроса к Ollama'}, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type='text', text=json.dumps({'error': str(e)}, ensure_ascii=False, indent=2))]


async def handle_list_models() -> list[TextContent]:
    try:
        url = f'{OLLAMA_BASE_URL}/api/tags'
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            models = data.get('models', [])
            result = {
                'available': True,
                'models': [{'name': m.get('name', ''), 'modified_at': m.get('modified_at', '')} for m in models],
                'count': len(models),
            }
            return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type='text', text=json.dumps({'error': str(e), 'available': False}, ensure_ascii=False, indent=2))]


async def handle_status() -> list[TextContent]:
    try:
        url = f'{OLLAMA_BASE_URL}/api/tags'
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
            models = [m.get('name', '') for m in data.get('models', [])]
            result = {
                'available': True,
                'base_url': OLLAMA_BASE_URL,
                'models_count': len(models),
                'models': models,
                'process_running': find_ollama_process() is not None,
            }
            return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False, indent=2))]
    except httpx.ConnectError:
        result = {
            'available': False,
            'error': 'Не удалось подключиться к Ollama',
            'base_url': OLLAMA_BASE_URL,
            'process_running': find_ollama_process() is not None,
        }
        return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type='text', text=json.dumps({'error': str(e), 'available': False}, ensure_ascii=False, indent=2))]


async def handle_pull_model(arguments: dict) -> list[TextContent]:
    model = arguments.get('model', '')
    if not model:
        return [TextContent(type='text', text=json.dumps({'error': "Параметр 'model' обязателен"}, ensure_ascii=False, indent=2))]

    try:
        url = f'{OLLAMA_BASE_URL}/api/pull'
        payload = {'name': model, 'stream': False}
        async with httpx.AsyncClient(timeout=600.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = {
                'model': model,
                'status': 'downloaded' if data.get('status') == 'success' else data.get('status', 'unknown'),
                'message': data.get('message', ''),
            }
            return [TextContent(type='text', text=json.dumps(result, ensure_ascii=False, indent=2))]
    except httpx.TimeoutException:
        return [TextContent(type='text', text=json.dumps({
            'error': 'Таймаут при скачивании модели',
            'note': 'Скачивание больших моделей может занять много времени',
        }, ensure_ascii=False, indent=2))]
    except Exception as e:
        return [TextContent(type='text', text=json.dumps({'error': str(e)}, ensure_ascii=False, indent=2))]


async def main():
    await initialize_config()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == '__main__':
    asyncio.run(main())
