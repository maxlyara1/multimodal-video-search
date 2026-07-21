from __future__ import annotations

import os
import json
from pathlib import Path
from src.pipeline import VideoRAGPipeline

def main():
    # Force local embedding backend
    os.environ["EMBEDDING_BACKEND"] = "local"

    config_path = "configs/config.yaml"
    print(f"Инициализация пайплайна с конфигом: {config_path}")
    pipeline = VideoRAGPipeline(config_path)

    # 1. Загрузка манифеста
    manifest_path = Path("benchmark/dataset_manifest.json")
    if not manifest_path.exists():
        print(f"Ошибка: Манифест {manifest_path} не найден.")
        raise SystemExit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest_video_ids = {v["video_id"] for v in manifest["videos"]}
    print(f"В манифесте описано видео: {len(manifest_video_ids)} шт.")

    # 2. Проверка наличия JSON-артефактов
    artifacts_dir = Path("data/artifacts")
    if not artifacts_dir.exists():
        print(f"Ошибка: Директория артефактов {artifacts_dir} не найдена.")
        raise SystemExit(1)

    artifact_paths = list(artifacts_dir.glob("*.json"))
    artifact_video_ids = {p.stem for p in artifact_paths}
    print(f"В папке data/artifacts найдено JSON-артефактов: {len(artifact_video_ids)} шт.")

    # 3. Проверка строгой целостности
    if manifest_video_ids != artifact_video_ids:
        print("Ошибка: Нарушена целостность набора данных!")
        print(f"В манифесте, но нет в артефактах: {manifest_video_ids - artifact_video_ids}")
        print(f"В артефактах, но нет в манифесте: {artifact_video_ids - manifest_video_ids}")
        raise ValueError("set(video_id из manifest) != set(video_id разрешённых artifacts)")

    print("Целостность набора данных успешно подтверждена.")

    # 4. Сбор записей по модальностям
    modalities = pipeline.enabled_modalities()
    all_records = {m: [] for m in modalities}

    for idx, video_id in enumerate(sorted(manifest_video_ids), 1):
        artifact_path = artifacts_dir / f"{video_id}.json"
        artifact = pipeline._load_artifact(artifact_path)
        for modality, records in artifact.items():
            all_records.setdefault(modality, []).extend(records)

    # Приводим к формату для индексирования (например, распаковка Visual-записей по типам)
    all_records = {
        modality: pipeline._records_for_index(modality, records)
        for modality, records in all_records.items()
    }

    total_records = sum(len(r) for r in all_records.values())
    print(f"Подготовлено записей для построения индекса Qdrant: {total_records} шт.")

    # 5. Эмбеддинг и запись в Qdrant
    print("=== Эмбеддинг и запись в Qdrant ===")
    store = pipeline._get_store()
    embedder = pipeline._get_embedder()

    for modality, records in all_records.items():
        store.recreate_collection(modality)
        if not records:
            print(f"  [{modality.upper()}] 0 записей — пропуск")
            continue
        print(f"  [{modality.upper()}] эмбеддинг {len(records)} записей...")
        embeddings = embedder.embed(
            [record.text for record in records],
            batch_size=pipeline.cfg["indexing"].get("batch_size", 8),
        )
        store.upsert_records(modality, records, embeddings)
        print(f"  [{modality.upper()}] успешно записано в Qdrant.")

    print("\nПостроение векторного индекса из артефактов успешно завершено!")
    pipeline.close()

if __name__ == "__main__":
    main()
