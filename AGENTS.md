# ollama_framework — инструкции агенту

Единая точка runtime-вызовов Ollama и ops-управления сервером в ERGO MS.

## Куда смотреть

- Правила Cursor модуля — `.cursor/rules/ollama-framework.mdc`
- README и `ergoms.help.yaml`
- Мост — `api/integrations.py`
- Ops — `deployment/`, команды `ergoms ollama_framework:…`

## Запрещено

- Прямой httpx / ollama SDK к `:11434` из других модулей
- Правки classifier ядра ради учёта процесса — только `process_roles.yaml` / `host_lifecycle.yaml` этого модуля

## Обязательно

- Вызовы LLM — `ollama_invoke` / `create_client` из этого модуля
- Транспорт — `OLLAMA_FRAMEWORK_TRANSPORT=local|http`
