# ollama_framework

Единая точка runtime-вызовов Ollama и ops-управления сервером в ERGO MS.

## Документация

- Правило Cursor: [`.cursor/rules/ollama-framework.mdc`](../../.cursor/rules/ollama-framework.mdc)
- Справочник команд: [`ergoms.help.yaml`](ergoms.help.yaml)

## Быстрый старт

```bash
ergoms ollama_framework:install-ollama
ergoms ollama_framework:start-ollama
ergoms ollama_framework:ollama-status
```

## Переменные окружения (корневой `.env`)

| Ключ | Назначение |
|------|------------|
| `OLLAMA_BASE_URL` | URL сервера Ollama (по умолчанию `http://127.0.0.1:11434`) |
| `OLLAMA_DEFAULT_MODEL` | Модель по умолчанию для chat/generate |
| `OLLAMA_EMBEDDINGS_MODEL` | Модель для embeddings |
| `OLLAMA_FRAMEWORK_TRANSPORT` | `local` (ModuleBridge in-process) или `http` (REST API модуля) |
| `OLLAMA_FRAMEWORK_API_BASE` | Базовый URL API при transport=http |
| `OLLAMA_REQUEST_TIMEOUT` | Таймаут запроса, сек |
| `OLLAMA_STREAM_TIMEOUT` | Таймаут streaming, сек |

## ModuleBridge

Операции в [`api/integrations.py`](api/integrations.py): `ollama_framework.health`, `chat`, `generate`, `embed`, `list_models`.

## REST API

- `GET /api/ollama_framework/status/`
- `GET /api/ollama_framework/models/`
- `POST /api/ollama_framework/generate/`, `chat/`, `embed/`

Вызовы из других модулей — через `ollama_invoke` или `create_client` из `api/client/factory.py`, не напрямую к `:11434`.
