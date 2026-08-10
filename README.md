# ollama_framework

Единая точка runtime-вызовов Ollama и ops-управления сервером в ERGO MS.

## Документация

- Правила Cursor: [`.cursor/rules/ollama-framework.mdc`](.cursor/rules/ollama-framework.mdc), [`AGENTS.md`](AGENTS.md)
- Справочник команд: [`ergoms.help.yaml`](ergoms.help.yaml)

## Быстрый старт

Ops-команды (install/start/stop/status/pull) — скрипты в `deployment/`, **без Django**.

```bash
ergoms ollama_framework:install-ollama
ergoms ollama_framework:pull-setup-models
ergoms ollama_framework:start-ollama
ergoms ollama_framework:ollama-status
ergoms ollama_framework:logs-ollama
ergoms ollama_framework:ollama --pull mistral:latest
```

`pull-setup-models` (и setup-full) при необходимости поднимают Ollama в фоне; логи — `logs/ollama-serve.log` (`ergoms ollama_framework:logs-ollama`).

## Переменные окружения

Задаются в корневом `.env` и/или в `modules/ollama_framework/.env` (пример — [`.env.example`](.env.example)). Для ops-скриптов `deployment/paths.read_env` читает: переменные процесса → модульный `.env` → корневой `.env`.

| Ключ | Назначение |
|------|------------|
| `OLLAMA_BASE_URL` | URL сервера Ollama (по умолчанию `http://127.0.0.1:11434`) |
| `OLLAMA_DEFAULT_MODEL` | Модель по умолчанию для chat/generate |
| `OLLAMA_EMBEDDINGS_MODEL` | Модель для embeddings (pull в setup-full через `ollama_models.yaml`) |
| `OLLAMA_FRAMEWORK_TRANSPORT` | `local` (ModuleBridge in-process) или `http` (REST API модуля) |
| `OLLAMA_FRAMEWORK_API_BASE` | Базовый URL API при transport=http |
| `OLLAMA_REQUEST_TIMEOUT` | Таймаут запроса, сек |
| `OLLAMA_STREAM_TIMEOUT` | Таймаут streaming, сек |
| `OLLAMA_COMPUTE_DEVICE` | `auto` (по умолчанию — выбор Ollama), либо принудительно `gpu` / `cpu` |

## ModuleBridge

Операции в [`api/integrations.py`](api/integrations.py): `ollama_framework.health`, `list_models`, `generate`, `chat`, `chat_stream`, `embed`.

## REST API и права ADP

| Эндпоинт | Право |
|----------|--------|
| `GET /api/ollama_framework/status/`, `models/` | `ollama_framework_view` |
| `POST …/generate/`, `chat/`, `embed/` | `ollama_framework_llm` |

Вызовы из других модулей — только через `bridge.call('ollama_framework.<op>', …)`, не через прямой import и не напрямую к `:11434`.

Embeddings идут через `/api/embed`; если сервер отвечает `404`/`501` (старый Ollama или кратковременный отказ), клиент автоматически вызывает legacy `/api/embeddings`.
