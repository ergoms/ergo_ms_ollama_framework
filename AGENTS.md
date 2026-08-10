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
- Правки classifier ядра ради учёта процесса — только `process_roles.yaml` / `host_lifecycle.yaml` этого модуля

## Обязательно

- Вызовы LLM из других модулей — `bridge.call('ollama_framework.<op>', …)`
- Внутри модуля — `create_client` / runtime / transport
- Транспорт потребителя — `OLLAMA_FRAMEWORK_TRANSPORT=local|http`
- REST: status/models — право `ollama_framework_view`; generate/chat/embed — `ollama_framework_llm`
- Модели setup-full — hook `ollama_models.yaml`; loader в `deployment/ollama_models_loader.py`, pull — `ergoms ollama_framework:pull-setup-models` (фоновый serve через `start_ollama_background` → `logs/ollama-serve.log`, не `stderr=PIPE`)
- Foreground `start-ollama` на Windows — Job Object `KILL_ON_JOB_CLOSE` + `CREATE_BREAKAWAY_FROM_JOB` (иначе в терминале Cursor serve переживает закрытие вкладки); фоновый `start_ollama_background` по-прежнему переживает выход родителя
- Embed: сначала `/api/embed`, при 404/501 — fallback на legacy `/api/embeddings` (`prompt`)
- Документация модуля — синхронизировать с кодом (корневой `module-docs.mdc`)
