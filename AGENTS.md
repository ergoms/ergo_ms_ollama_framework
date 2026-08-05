# ollama_framework — инструкции агенту

Единая точка runtime-вызовов Ollama и ops-управления сервером в ERGO MS.

## Куда смотреть

- Правила Cursor модуля — `.cursor/rules/ollama-framework.mdc`
- README и `ergoms.help.yaml`
- Мост — `api/integrations.py`
- Ops — `deployment/`, команды `ergoms ollama_framework:…`

## Запрещено

- Прямой httpx / ollama SDK к `:11434` из других модулей
- `from modules.ollama_framework…` в других модулях (только ModuleBridge)
- Правки classifier ядра ради учёта процесса — только `process_roles.yaml` / `host_lifecycle.yaml` этого модуля

## Обязательно

- Вызовы LLM из других модулей — `bridge.call('ollama_framework.<op>', …)`
- Внутри модуля — `create_client` / runtime / transport
- Транспорт потребителя — `OLLAMA_FRAMEWORK_TRANSPORT=local|http`
- Модели setup-full — hook `ollama_models.yaml` (discovery + `ergoms ollama_framework:pull-setup-models`)
