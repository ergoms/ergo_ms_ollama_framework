from django.urls import path

from .views import (
    OllamaChatView,
    OllamaEmbedView,
    OllamaGenerateView,
    OllamaModelsView,
    OllamaStatusView,
)

urlpatterns = [
    path('status/', OllamaStatusView.as_view(), name='ollama-framework-status'),
    path('models/', OllamaModelsView.as_view(), name='ollama-framework-models'),
    path('generate/', OllamaGenerateView.as_view(), name='ollama-framework-generate'),
    path('chat/', OllamaChatView.as_view(), name='ollama-framework-chat'),
    path('embed/', OllamaEmbedView.as_view(), name='ollama-framework-embed'),
]
