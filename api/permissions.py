from src.core.cms.adp.base_permissions import BaseModulePermission

MODULE_NAME = 'ollama_framework'

OLLAMA_FRAMEWORK_VIEW = 'ollama_framework_view'


class _BaseOllamaFrameworkPermission(BaseModulePermission):
    module_name = MODULE_NAME


class CanViewOllamaFramework(_BaseOllamaFrameworkPermission):
    required_permission = OLLAMA_FRAMEWORK_VIEW
