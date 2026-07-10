from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class LLMClientError(RuntimeError):
    """Общее исключение для ошибок работы с Ollama."""


class BaseLLMClient:
    """Базовый интерфейс LLM-клиента ollama_framework."""

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(
        self,
        prompt: str,
        *,
        num_predict: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        system: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        stream: bool = False,
        stream_callback: Optional[Callable[[str], None]] = None,
        format: Optional[Any] = None,
        return_stats: bool = False,
    ) -> str:
        raise NotImplementedError

    def embed(self, texts: List[str], *, model: Optional[str] = None) -> List[List[float]]:
        raise NotImplementedError

    def list_models(self) -> List[str]:
        raise NotImplementedError

    def check_health(self) -> Dict[str, Any]:
        return {'available': False, 'error': 'Метод не реализован'}

    def get_provider_name(self) -> str:
        return 'unknown'
