from src.core.cms.adp.base_permissions import BaseModulePermission

MODULE_NAME = 'ollama_framework'

OLLAMA_FRAMEWORK_VIEW = 'ollama_framework_view'
OLLAMA_FRAMEWORK_LLM = 'ollama_framework_llm'


class _BaseOllamaFrameworkPermission(BaseModulePermission):
    module_name = MODULE_NAME


class CanViewOllamaFramework(_BaseOllamaFrameworkPermission):
    """Status / list models — лёгкие read-only эндпоинты."""

    required_permission = OLLAMA_FRAMEWORK_VIEW


class CanUseOllamaLlm(_BaseOllamaFrameworkPermission):
    """Generate / chat / embed — дорогие LLM-вызовы."""

    required_permission = OLLAMA_FRAMEWORK_LLM
