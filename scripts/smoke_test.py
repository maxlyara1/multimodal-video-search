from __future__ import annotations

import compileall
import sys
from unittest.mock import MagicMock


def run_smoke_test() -> None:
    print("=== Запуск облегченного Smoke-теста ===")

    # 1. Проверяем синтаксис всех файлов в проекте через compileall
    print("Проверка синтаксиса файлов проекта...")
    compiled = compileall.compile_dir(".", maxlevels=10, quiet=True)
    if not compiled:
        print("Ошибка: Обнаружены синтаксические ошибки в кодовой базе!")
        sys.exit(1)
    print("Все Python-файлы успешно скомпилированы.")

    # 2. Мокаем тяжелые библиотеки перед импортом
    print("Подготовка mock-заглушек для тяжелых библиотек...")
    for mod in ["transformers", "torch", "easyocr", "whisper", "spacy"]:
        sys.modules[mod] = MagicMock()

    try:
        # 3. Импортируем app и проверяем routes
        print("Импорт app.py...")
        from app import app
        assert app is not None
        print("Проверка FastAPI эндпоинтов...")
        routes = [r.path for r in app.routes]
        print(f"Доступные роуты: {routes}")
        assert "/api/ask" in routes
        assert "/api/upload" in routes

        # 4. Проверяем чтение конфига
        print("Проверка чтения конфигурации...")
        import yaml
        with open("configs/config.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        assert cfg is not None
        assert "search" in cfg

        # 5. Тестируем логику RRF-слияния
        print("Тестирование алгоритма RRF слияния...")
        from src.models import SearchHit
        from src.pipeline import VideoRAGPipeline

        # Инициализируем пайплайн с фейковым конфигом для изоляции
        mock_pipeline = MagicMock(spec=VideoRAGPipeline)
        mock_pipeline.cfg = cfg
        mock_pipeline._deduplicate_candidates = lambda x: x

        # Создаем тестовые SearchHit
        hits = [
            SearchHit(video_file="video1.mp4", modality="asr", start=10.0, end=20.0, score=0.9, text="тест"),
            SearchHit(video_file="video1.mp4", modality="ocr", start=12.0, end=22.0, score=0.8, text="тест"),
        ]

        # Вызываем _merge_hits
        # Так как метод статический/внутренний, вызовем его у класса
        candidates = VideoRAGPipeline._merge_hits(mock_pipeline, hits)
        print(f"RRF слияние вернуло {len(candidates)} кандидатов.")
        assert len(candidates) > 0
        assert candidates[0].video_file == "video1.mp4"

        print("=== Smoke Test PASSED успешно! ===")
        sys.exit(0)

    except Exception as e:
        print(f"Ошибка при прохождении Smoke-теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
