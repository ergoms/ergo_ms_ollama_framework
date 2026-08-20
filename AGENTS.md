# ollama_framework — инструкции агенту

Единая точка runtime-вызовов Ollama и ops-управления сервером в ERGO MS.

## Куда смотреть

- Правила Cursor модуля — `.cursor/rules/ollama-framework.mdc`
- README и `ergoms.help.yaml`
- Мост — `api/integrations.py`
- Ops — `deployment/`, команды `ergoms ollama_framework:…`
- Env ops — `deployment/paths.py` (`read_env`: process → module `.env` → корневой `.env`)

## Запрещено

- Прямой httpx / ollama SDK к `:11434` из других модулей
- `from modules.ollama_framework…` в других модулях (только ModuleBridge)
- Правки classifier ядра ради учёта процесса — только `process_roles.yaml` / `host_lifecycle.yaml` этого модуля (`reserve_host_budget`, `reserve_vram_mb` для бюджета Celery)

## Обязательно

- Вызовы LLM из других модулей — `bridge.call('ollama_framework.<op>', …)`
- Внутри модуля — `create_client` / runtime / transport
- Транспорт потребителя — `OLLAMA_FRAMEWORK_TRANSPORT=local|http`
- REST: status/models — право `ollama_framework_view`; generate/chat/embed — `ollama_framework_llm`
- Модели setup-full — hook `ollama_models.yaml`; loader в `deployment/ollama_models_loader.py`, pull — `ergoms ollama_framework:pull-setup-models` (фоновый и foreground serve пишут в `logs/ollama-serve.log` через `log_env` / `ERGO_LOG_FILE_OLLAMA`, не `stderr=PIPE`); в конце `setup-full` ядро останавливает serve через `host_lifecycle.yaml` → `stop_commands`
- Foreground `start-ollama` на Windows — Job Object `KILL_ON_JOB_CLOSE` + `CREATE_BREAKAWAY_FROM_JOB` (иначе в терминале Cursor serve переживает закрытие вкладки); фоновый `start_ollama_background` по-прежнему переживает выход родителя
- Embed: сначала `/api/embed`, при 404/501 — fallback на legacy `/api/embeddings` (`prompt`); alias имени модели (`embeddinggemma` → `:latest`) логировать один раз на клиент, не на каждый вызов
- Serve слушает `127.0.0.1` (`OLLAMA_HOST` в `build_ollama_env`); LAN — только явным `OLLAMA_HOST=0.0.0.0:11434`
- Документация модуля — синхронизировать с кодом (корневой `module-docs.mdc`)

## Команды

```bash
ergoms ollama_framework:install-ollama
ergoms ollama_framework:pull-setup-models
ergoms ollama_framework:start-ollama
ergoms ollama_framework:ollama-status
ergoms help module ollama_framework
```
