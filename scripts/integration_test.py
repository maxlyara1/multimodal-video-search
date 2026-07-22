import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.pipeline import VideoRAGPipeline


def run_integration_test() -> None:
    print("=== Запуск Интеграционного теста Video-RAG Pipeline ===")

    config_path = ROOT_DIR / "configs/config.yaml"
    if not config_path.exists():
        print(f"Ошибка: Конфиг {config_path} не найден!")
        sys.exit(1)

    print("Инициализация пайплайна (требует загрузки локальных моделей)...")
    pipeline = VideoRAGPipeline(str(config_path))

    try:
        print("Проверка списка подготовленных видео...")
        videos = pipeline.list_prepared_videos()
        print(f"Найдено подготовленных видео: {len(videos)}")
        if len(videos) == 0:
            print("Integration test SKIPPED: source videos and built Qdrant index are not included in open repository.")
            print("Pre-extracted features (ASR/OCR/Visual) remain available in data/artifacts/.")
            sys.exit(0)

        print("Выполнение тестового поиска по индексу Qdrant...")
        query = "роблокс"
        decomposition, candidates = pipeline.search(query)

        print(f"Поиск завершен. Найдено кандидатов: {len(candidates)}")
        for idx, c in enumerate(candidates, 1):
            print(f"[{idx}] Видео: {Path(c.video_file).name} | Score: {c.score:.4f} | Интервал: {c.start:.1f}s - {c.end:.1f}s")

        if not candidates:
            print("Внимание: Поиск вернул 0 результатов.")

        print("=== Integration Test PASSED успешно! ===")

    except Exception as e:
        print(f"Ошибка при прохождении интеграционного теста: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        pipeline.close()

if __name__ == "__main__":
    run_integration_test()
