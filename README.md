# ollama_framework

Инструмент и модели объявлены в [`runtime_requirements.yaml`](runtime_requirements.yaml).

Для данных, которые нельзя передавать другому серверу, внутренний мост предоставляет `ollama_framework.chat_local(messages, model=..., timeout=30, num_predict=128, format=...)`. Операция использует только `http://127.0.0.1:11434`, не берёт системный HTTP-прокси, не меняет модель по семейству и не повторяет запрос на другом адресе. Перед каждым вызовом `/api/show` должен подтвердить локальные метаданные без `remote_host`/`remote_model`; облачные имена отклоняются. Это вызов локального провайдера моста, не новый REST-метод: потребитель также должен запретить HTTP-транспорт моста. Короткий ответ со схемой сохраняет явное `think=false`; прежний `chat` работает как раньше. Режим не является сетевой песочницей для недоверенного локального демона Ollama.
Ollama устанавливается общим реестром ядра; прежние CLI-алиасы и `ollama_models.yaml` поддерживаются.

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
ergoms ollama_framework:ollama --pull ministral-3:8b
```

`pull-setup-models` (и setup-full) скачивает только обязательные модели; `--include-optional` явно включает остальные. Готовые локальные модели проверяются без запуска сервера. Для отсутствующих весов Ollama при необходимости поднимается в фоне. Serve (фон, `start-ollama` и служба `ergo-ollama`) пишет в один файл `logs/ollama-serve.log` (`ERGO_LOG_FILE_OLLAMA`, ротация `ergoms rotate-logs`). По окончании `setup-full` процесс останавливается через `host_lifecycle.yaml` (`stop-ollama`).

Общий индекс подключает модуль через [`model_providers.yaml`](model_providers.yaml): адаптер использует прежние loader/OllamaOps и не переносит инференс в ядро. `ergoms runtime-model-resolve --provider ollama --model embeddinggemma --json` возвращает проверенный локальный manifest и точное runtime-имя. Проверяются manifest, наличие/размеры blobs; полная SHA256-проверка — `deployment.model_store.inspect_model(..., verify_hashes=True)`. Установки одной модели сериализуются; готовые веса повторно не скачиваются. SDK обращается к настроенному `OLLAMA_BASE_URL` с действующей политикой `ERGO_HTTP_TRUST_ENV`.

Windows-установщик службы передаёт корень проекта в NSSM. Перед переустановкой он проверяет `AppDirectory`, подготавливает wrapper, ждёт остановки и удаления прежней службы не более 30 секунд на каждый шаг. Служба другого корня не изменяется; ошибка NSSM или таймаут прерывают установку до запуска. Wrapper сохраняет пути с пробелами и кириллицей в UTF-8 без BOM. Регрессии в `tests/test_windows_service_installer.py` используют подменённые NSSM и команды служб, без обращения к действующим службам Windows.

## Переменные окружения

Задаются в корневом `.env` и/или в `modules/ollama_framework/.env` (пример — [`.env.example`](.env.example)). Для ops-скриптов `deployment/paths.read_env` читает: переменные процесса → модульный `.env` → корневой `.env`.

| Ключ | Назначение |
|------|------------|
| `OLLAMA_BASE_URL` | URL сервера Ollama (по умолчанию `http://127.0.0.1:11434`) |
| `OLLAMA_HOST` | Адрес прослушивания `ollama serve`. По умолчанию `127.0.0.1:11434`. Доступ не с localhost — только явно, например `0.0.0.0:11434` |
| `OLLAMA_DEFAULT_MODEL` | Модель по умолчанию для chat/generate |
| `OLLAMA_EMBEDDINGS_MODEL` | Имя библиотеки Ollama (`ollama pull` / `/api/embed`) или снимок Hugging Face `org/name`. Снимок (`deepvk/USER-bge-m3`) считается sentence-transformers, без подмены на `embeddinggemma`. Веса — `huggingface_models.yaml` и `ergoms pull-huggingface-models` |
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

Имя библиотеки Ollama идёт через `/api/embed`; если сервер отвечает `404`/`501` (старый Ollama или кратковременный отказ), клиент вызывает legacy `/api/embeddings`. Снимок `org/name` в `OLLAMA_EMBEDDINGS_MODEL` в Ollama не отправляется: векторы считает sentence-transformers.


JSON chat: замена bool `think` на `low` нужна только семейству `gpt-oss`; остальные модели сохраняют переданный режим. Строго локальный режим сохраняет явное `think=false` для любой модели.
