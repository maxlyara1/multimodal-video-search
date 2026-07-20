from __future__ import annotations

import sys
from pathlib import Path

def run_smoke_test() -> None:
    print("=== Запуск Smoke-теста (проверка импортов и синтаксиса) ===")
    
    try:
        # Проверяем наличие конфига
        config_path = Path("configs/config.yaml")
        if not config_path.exists():
            print(f"Ошибка: Конфиг {config_path} не найден!")
            sys.exit(1)
            
        print("Импорт модулей приложения...")
        from app import app
        from src.pipeline import VideoRAGPipeline
        from src.config import load_config
        from src.retrieval.embedder import Embedder
        from src.retrieval.qdrant_store import QdrantStore
        
        print("Проверка экземпляра FastAPI...")
        assert app is not None
        assert app.title == "Video RAG Search Interface"
        
        print("=== Smoke Test PASSED успешно! ===")
        sys.exit(0)
        
    except Exception as e:
        print(f"Ошибка при прохождении Smoke-теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
