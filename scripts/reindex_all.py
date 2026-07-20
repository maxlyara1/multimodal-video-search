from __future__ import annotations

import os
from pathlib import Path
from src.pipeline import VideoRAGPipeline

def main():
    # Force local embedding backend
    os.environ["EMBEDDING_BACKEND"] = "local"
    
    config_path = "configs/config.yaml"
    print(f"Инициализация пайплайна с конфигом: {config_path}")
    pipeline = VideoRAGPipeline(config_path)
    
    videos_dir = Path("data/videos")
    if not videos_dir.exists():
        print(f"Ошибка: Директория {videos_dir} не найдена. Скопируйте видео.")
        return
        
    video_paths = sorted(
        [p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}]
    )
    
    if not video_paths:
        print("В папке data/videos не найдено видеофайлов.")
        return
        
    print(f"Найдено {len(video_paths)} видеофайлов для индексации.")
    
    # Реиндексируем с префиксом None (будет использован collection_prefix из config.yaml)
    stats = pipeline.index_uploaded_videos(video_paths, prefix=None, recreate=True)
    
    print("\nИндексация успешно завершена!")
    for modality, count in stats.items():
        print(f"  - {modality.upper()}: {count} записей добавлено.")
        
    pipeline.close()

if __name__ == "__main__":
    main()
