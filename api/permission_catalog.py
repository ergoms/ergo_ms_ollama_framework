from .permissions import MODULE_NAME, OLLAMA_FRAMEWORK_VIEW

PERMISSION_CATALOG = {
    'module_name': MODULE_NAME,
    'permissions': {
        OLLAMA_FRAMEWORK_VIEW: 'Доступ к API Ollama (status, models, generate, chat, embed)',
    },
}
