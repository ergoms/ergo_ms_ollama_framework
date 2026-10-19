# ollama_framework — как должен работать модуль

Нормы из чата. Не копия platform-правил ядра.

## Модели

### Какие линейки качать
- Так: из Qwen держи только самую новую уже стоящую линейку, сейчас `qwen3.8`. Рядом можно Gemma 4, Ministral 3 и `gpt-oss`.
- Так: если `OLLAMA_EMBEDDINGS_MODEL` — снимок `deepvk/USER-bge-m3`, эмбеддинги считает эта модель через sentence-transformers.
- Не так: подменять этот снимок на `embeddinggemma` в `/api/embed` или в `pull-setup-models`.
- Не так: самому качать предыдущие линейки, пока об этом не попросили отдельно. Сюда входят Qwen3 без суффикса 3.8, Qwen 3.5, Qwen 3.6, Gemma 3, Llama 3.1, Phi-4, DeepSeek R1 и Mistral Small. Другой размер старой линейки не повод её качать: `qwen3.6:35b` при стоящей `qwen3.8:27b` не нужна.
