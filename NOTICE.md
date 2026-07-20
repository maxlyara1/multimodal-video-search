# Уведомление о происхождении и лицензиях (Notice)

Система мультимодального поиска по видео (`multimodal-video-search`) является независимой реализацией, вдохновленной концепциями и архитектурными принципами научной статьи:

> **Video-RAG: Visually-aligned Retrieval-Augmented Long Video Comprehension**  
> *Authors: Work done by researchers/authors of arXiv:2411.13093*  
> *Original Codebase: https://github.com/Leon1207/Video-RAG-master*

### Отличие реализации:
1. Данный репозиторий представляет собой независимую разработку с нуля, ориентированную на универсальный модульный пайплайн и поддержку русскоязычных видеоматериалов.
2. В отличие от оригинальной архитектуры, использующей сложные внешние VLM-сервисы для генерации описаний плотных кадров, в данном проекте реализовано локальное извлечение признаков:
   - **ASR**: Локальный Whisper (модель turbo) с параллельной нарезкой длинных аудиозаписей.
   - **OCR**: Локальный EasyOCR с поддержкой кириллицы.
   - **Visual**: Локальный BLIP-base с последующим разбором Scene Graph с помощью NLP-библиотеки spaCy.
3. Интегрирована легковесная векторная модель `Qwen3-Embedding-0.6B` для одновременного кодирования текстовых, визуальных и речевых признаков в едином векторном пространстве.
4. Добавлен локальный веб-интерфейс MVP (FastAPI + Vanilla JS/CSS) с интерактивным плеером, SSE-стримингом логов реального времени и поддержкой drag-and-drop загрузки.
5. Интегрированы алгоритмы рангового слияния результатов (Reciprocal Rank Fusion - RRF и Max-per-modality score normalization) для устранения перекоса шкал оценок сходства.

### Лицензии используемых компонентов:
- **FastAPI / Uvicorn**: Uvicorn (MIT License), FastAPI (MIT License).
- **Qdrant Client**: Apache License 2.0.
- **EasyOCR**: Apache License 2.0.
- **OpenAI Whisper**: MIT License.
- **Transformers (Hugging Face)**: Apache License 2.0.
- **spaCy**: MIT License.
