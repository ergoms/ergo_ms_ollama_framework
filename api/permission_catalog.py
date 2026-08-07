from .permissions import MODULE_NAME, OLLAMA_FRAMEWORK_LLM, OLLAMA_FRAMEWORK_VIEW

PERMISSION_CATALOG = {
    'module_name': MODULE_NAME,
    'module_label': 'Фреймворк Ollama',
    'permissions': {
        OLLAMA_FRAMEWORK_VIEW: 'Просмотр статуса и списка моделей Ollama',
        OLLAMA_FRAMEWORK_LLM: 'Вызовы LLM Ollama (generate, chat, embed)',
    },
}
