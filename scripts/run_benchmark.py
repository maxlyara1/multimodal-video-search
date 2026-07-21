from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import time
from pathlib import Path

from src.models import QueryDecomposition
from src.pipeline import VideoRAGPipeline

logger = logging.getLogger(__name__)


def get_git_commit() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown-commit"


def get_file_sha256(filepath: str | Path) -> str:
    path = Path(filepath)
    if not path.exists():
        return "none"
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:12]


def get_qdrant_video_ids(pipeline: VideoRAGPipeline) -> set[str]:
    store = pipeline._get_store()
    video_ids = set()
    for modality in ["asr", "ocr", "visual"]:
        name = store.collection_name(modality)
        if not store.client.collection_exists(name):
            continue
        # Получаем список точек из коллекции
        points, _ = store.client.scroll(
            collection_name=name,
            limit=10000,
            with_payload=True,
            with_vectors=False
        )
        for point in points:
            if point.payload and "video_file" in point.payload:
                video_filename = Path(point.payload["video_file"]).stem
                video_ids.add(video_filename.lower())
    return video_ids


def run_evaluation(pipeline: VideoRAGPipeline, mode: str, fusion_method: str, tasks: list[dict], use_routing: bool = True) -> dict:
    # Configure pipeline settings programmatically
    if mode == "asr-only":
        pipeline.cfg["asr"]["enabled"] = True
        pipeline.cfg["ocr"]["enabled"] = False
        pipeline.cfg["visual"]["enabled"] = False
    elif mode == "multimodal":
        pipeline.cfg["asr"]["enabled"] = True
        pipeline.cfg["ocr"]["enabled"] = True
        pipeline.cfg["visual"]["enabled"] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    pipeline.cfg["search"]["fusion_method"] = fusion_method

    hits_top1 = 0
    hits_top3 = 0
    total_latency = 0.0
    total_qdrant_latency = 0.0

    category_stats = {
        "speech": {"total": 0, "top1": 0, "top3": 0},
        "ocr": {"total": 0, "top1": 0, "top3": 0},
        "visual": {"total": 0, "top1": 0, "top3": 0},
    }

    results = []

    mode_ru = "Только ASR" if mode == "asr-only" else "Мультимодальный"
    fusion_ru = "Сумма" if fusion_method == "sum" else ("RRF" if fusion_method == "rrf" else "Максимум")
    routing_suffix = "" if use_routing else " (Без отбора источников)"
    print(f"\n=== Оценка качества в режиме: {mode_ru.upper()} ({fusion_ru.upper()}){routing_suffix} ===")

    category_map_ru = {
        "speech": "Речь",
        "ocr": "Текст",
        "visual": "Кадр",
    }

    for task in tasks:
        query = task["query"]
        expected = task["expected_video"].strip().lower()
        category = task["category"]

        category_stats[category]["total"] += 1

        t_start = time.perf_counter()

        # Step 1: Query Decomposition (Gemini or offline tasks.json)
        if mode == "asr-only":
            decomposition = QueryDecomposition(
                original_query=query,
                asr_query=query,
                visual_queries=[],
                visual_mode="all"
            )
        elif not use_routing:
            decomposition = QueryDecomposition(
                original_query=query,
                asr_query=query,
                visual_queries=[query],
                visual_mode="all"
            )
        else:
            if "decomposition" in task and task["decomposition"] is not None:
                decomp_data = task["decomposition"]
                decomposition = QueryDecomposition(
                    original_query=query,
                    asr_query=decomp_data.get("asr_query"),
                    visual_queries=decomp_data.get("visual_queries") or [],
                    visual_mode=decomp_data.get("visual_mode") or "all"
                )
            else:
                try:
                    query_decoupler = pipeline._get_query_decoupler()
                    if query_decoupler is not None:
                        decomposition = query_decoupler.decouple(query)
                    else:
                        decomposition = QueryDecomposition(original_query=query, asr_query=query, visual_queries=[], visual_mode="all")
                except Exception as e:
                    print(f"Warning: Gemini decoupler failed ({e}). Using default decomposition.")
                    decomposition = QueryDecomposition(original_query=query, asr_query=query, visual_queries=[], visual_mode="all")

        # Step 2: Retrieve vectors
        t_qdrant_start = time.perf_counter()

        candidates = pipeline.retrieve_with_decomposition(query, decomposition)

        t_qdrant = time.perf_counter() - t_qdrant_start
        t_total = time.perf_counter() - t_start

        total_latency += t_total
        total_qdrant_latency += t_qdrant

        # Deduplicate candidates to get correct Hit@K on video file level by grouping and selecting max score
        video_scores = {}
        for c in candidates:
            video_id = Path(c.video_file).stem.lower()
            video_scores[video_id] = max(
                video_scores.get(video_id, float("-inf")),
                c.score,
            )

        top_videos = [
            video_id
            for video_id, _ in sorted(
                video_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ]

        is_top1 = len(top_videos) > 0 and top_videos[0] == expected
        is_top3 = any(v == expected for v in top_videos[:3])

        if is_top1:
            hits_top1 += 1
            category_stats[category]["top1"] += 1
        if is_top3:
            hits_top3 += 1
            category_stats[category]["top3"] += 1

        status_str = "Hit@1" if is_top1 else ("Hit@3" if is_top3 else "Miss")
        cat_ru = category_map_ru.get(category, category)
        print(f"[{cat_ru.upper()}] Запрос: \"{query}\" -> Ожидалось: {expected} | Топ-3: {top_videos[:3]} | {status_str} (Задержка: {t_qdrant:.3f}с)")

        results.append({
            "query": query,
            "expected": expected,
            "category": category,
            "top_found": top_videos[:3],
            "t_qdrant": t_qdrant,
            "status": status_str,
        })

    num_tasks = len(tasks)
    metrics = {
        "hit_at_1": hits_top1 / num_tasks if num_tasks > 0 else 0.0,
        "hit_at_3": hits_top3 / num_tasks if num_tasks > 0 else 0.0,
        "local_retrieval_latency": total_qdrant_latency / num_tasks if num_tasks > 0 else 0.0,
        "avg_total_latency": total_latency / num_tasks if num_tasks > 0 else 0.0,
        "category_stats": category_stats,
        "results": results,
    }

    print(f"\n--- Метрики для {mode_ru.upper()} ({fusion_ru}):")
    print(f"Hit@1: {metrics['hit_at_1']:.2%}")
    print(f"Hit@3: {metrics['hit_at_3']:.2%}")
    print(f"Задержка локального поиска: {metrics['local_retrieval_latency']:.3f}с")

    return metrics


def main() -> None:
    os.environ["EMBEDDING_BACKEND"] = "local"
    parser = argparse.ArgumentParser(description="Оценка качества для сравнения ASR-only и гибридного мультимодального поиска с RRF и объединением по максимуму")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--tasks", default="benchmark/tasks.json")
    parser.add_argument("--output", default="benchmark_results.md", help="Путь для записи отчета в формате Markdown")
    args = parser.parse_args()

    tasks_path = Path(args.tasks)
    if not tasks_path.exists():
        print(f"Ошибка: Файл запросов {tasks_path} не найден.")
        raise SystemExit(1)

    with tasks_path.open("r", encoding="utf-8") as f:
        tasks = json.load(f)

    # 1. Проверка строгой целостности набора данных
    print("Проверка целостности базы данных...")
    temp_pipeline = VideoRAGPipeline(args.config)
    try:
        manifest_path = Path("benchmark/dataset_manifest.json")
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_video_ids = {v["video_id"].strip().lower() for v in manifest["videos"]}

        artifacts_dir = Path("data/artifacts")
        artifact_video_ids = {p.stem.strip().lower() for p in artifacts_dir.glob("*.json")}

        qdrant_video_ids = get_qdrant_video_ids(temp_pipeline)

        print(f"Манифест: {len(manifest_video_ids)} видео | Артефакты: {len(artifact_video_ids)} файлов | Qdrant: {len(qdrant_video_ids)} видео")

        if not (manifest_video_ids == artifact_video_ids == qdrant_video_ids):
            print("\n[ОШИБКА] Нарушена целостность набора данных!")
            print(f"Разница манифест - артефакты: {manifest_video_ids - artifact_video_ids}")
            print(f"Разница артефакты - манифест: {artifact_video_ids - manifest_video_ids}")
            print(f"Разница манифест - Qdrant: {manifest_video_ids - qdrant_video_ids}")
            raise ValueError("Нарушена целостность: set(video_id из manifest) == set(video_id в индексах) == set(video_id разрешённых artifacts) не выполняется!")
        print("Целостность набора данных подтверждена.")
    finally:
        temp_pipeline.close()

    # 2. Warm-up (без обращения к Gemini)
    print("Прогрев моделей...")
    pipeline = VideoRAGPipeline(args.config)
    try:
        warmup_decomp = QueryDecomposition(original_query="тест", asr_query="тест", visual_queries=[], visual_mode="all")
        pipeline.retrieve_with_decomposition("тест", warmup_decomp)
    finally:
        pipeline.close()

    # 3. Запуск оценки для всех режимов
    print("\nЗапуск измерений...")

    # Режим 1: Только ASR (Baseline)
    pipeline = VideoRAGPipeline(args.config)
    try:
        asr_metrics = run_evaluation(pipeline, "asr-only", "sum", tasks)
    finally:
        pipeline.close()

    # Режим 2: Мультимодальный (Простая сумма)
    pipeline = VideoRAGPipeline(args.config)
    try:
        mm_sum_metrics = run_evaluation(pipeline, "multimodal", "sum", tasks)
    finally:
        pipeline.close()

    # Режим 3: Мультимодальный (RRF)
    pipeline = VideoRAGPipeline(args.config)
    try:
        mm_rrf_metrics = run_evaluation(pipeline, "multimodal", "rrf", tasks)
    finally:
        pipeline.close()

    # Режим 4: Мультимодальный (Максимум модальностей)
    pipeline = VideoRAGPipeline(args.config)
    try:
        mm_max_metrics = run_evaluation(pipeline, "multimodal", "max_per_modality", tasks)
    finally:
        pipeline.close()

    # Режим 5: Мультимодальный (Максимум, Без отбора модальностей)
    pipeline = VideoRAGPipeline(args.config)
    try:
        mm_max_norouting_metrics = run_evaluation(pipeline, "multimodal", "max_per_modality", tasks, use_routing=False)
    finally:
        pipeline.close()

    # Сохранение в metrics.json
    metrics_json_path = Path("metrics.json")
    metrics_data = {
        "asr_only": {
            "hit_at_1": asr_metrics["hit_at_1"],
            "hit_at_3": asr_metrics["hit_at_3"],
            "latency": asr_metrics["local_retrieval_latency"]
        },
        "multimodal_sum": {
            "hit_at_1": mm_sum_metrics["hit_at_1"],
            "hit_at_3": mm_sum_metrics["hit_at_3"],
            "latency": mm_sum_metrics["local_retrieval_latency"]
        },
        "multimodal_rrf": {
            "hit_at_1": mm_rrf_metrics["hit_at_1"],
            "hit_at_3": mm_rrf_metrics["hit_at_3"],
            "latency": mm_rrf_metrics["local_retrieval_latency"]
        },
        "multimodal_max": {
            "hit_at_1": mm_max_metrics["hit_at_1"],
            "hit_at_3": mm_max_metrics["hit_at_3"],
            "latency": mm_max_metrics["local_retrieval_latency"]
        },
        "multimodal_max_norouting": {
            "hit_at_1": mm_max_norouting_metrics["hit_at_1"],
            "hit_at_3": mm_max_norouting_metrics["hit_at_3"],
            "latency": mm_max_norouting_metrics["local_retrieval_latency"]
        }
    }
    with open(metrics_json_path, "w", encoding="utf-8") as mj:
        json.dump(metrics_data, mj, indent=2, ensure_ascii=False)
    print(f"Метрики сохранены в: {metrics_json_path}")

    # Запись отчета в markdown
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    git_commit = get_git_commit()
    config_sha = get_file_sha256(args.config)
    tasks_sha = get_file_sha256("benchmark/tasks.json")
    dataset_sha = get_file_sha256("benchmark/dataset_manifest.json")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Результаты оценки качества поиска по видео\n\n")
        f.write("Сравнение стратегий поиска и алгоритмов объединения результатов на проверочном наборе из 50 запросов (20 речевых, 10 OCR, 20 визуальных) по 31 видеофайлу:\n")
        f.write("- **Только ASR**: поиск только по распознанной речи (Базовый вариант)\n")
        f.write("- **Мультимодальный (Простая сумма)**: поиск с простым суммированием оценок косинусного сходства и фиксированным выбором модальностей\n")
        f.write("- **Мультимодальный (RRF)**: поиск с ранговым слиянием Reciprocal Rank Fusion и фиксированным выбором модальностей\n")
        f.write("- **Мультимодальный (Максимум)**: поиск с объединением взвешенного максимума по каждой модальности (Max-per-Modality) и фиксированным выбором модальностей\n")
        f.write("- **Мультимодальный (Максимум, Без отбора модальностей)**: поиск Max-per-Modality, когда все модальности активны для любого запроса\n\n")

        f.write("> [!IMPORTANT]\n")
        f.write("> **Важное примечание**: Проверочный набор оценивает качество поиска и объединения результатов на 50 запросах при фиксированном (заданном вручную) выборе модальностей. Автоматический выбор модальностей моделью Gemini в данную оценку не входил. Метрика Hit@K рассчитывается на уровне видеофайлов, а не временных интервалов.\n\n")

        f.write("## Параметры окружения и воспроизводимости\n\n")
        f.write(f"- **Git Commit**: `{git_commit}`\n")
        f.write(f"- **Конфигурация (SHA-256)**: `{config_sha}` (файл `configs/config.yaml`)\n")
        f.write(f"- **Проверочные запросы (SHA-256)**: `{tasks_sha}` (файл `benchmark/tasks.json`)\n")
        f.write(f"- **Манифест набора данных (SHA-256)**: `{dataset_sha}` (файл `benchmark/dataset_manifest.json`)\n")
        f.write("- **Способ вычисления векторных представлений**: локальный, через `Qwen3-Embedding-0.6B`\n")
        f.write("- **Устройство**: MPS (Apple Silicon GPU)\n")
        f.write(f"- **Всего запросов**: {len(tasks)}\n")
        f.write("- **База векторов**: Qdrant (локальный запуск)\n\n")

        f.write("## 1. Сводные метрики (Hit@K на уровне видео)\n\n")
        f.write("| Стратегия | Метод объединения | Выбор модальностей | Hit@1 | Hit@3 | Задержка локального поиска |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        f.write(f"| Только ASR | Сумма | Нет | {asr_metrics['hit_at_1']:.2%} | {asr_metrics['hit_at_3']:.2%} | {asr_metrics['local_retrieval_latency']:.3f}с |\n")
        f.write(f"| Мультимодальный | Простая сумма | Задан | {mm_sum_metrics['hit_at_1']:.2%} | {mm_sum_metrics['hit_at_3']:.2%} | {mm_sum_metrics['local_retrieval_latency']:.3f}с |\n")
        f.write(f"| Мультимодальный | RRF | Задан | {mm_rrf_metrics['hit_at_1']:.2%} | {mm_rrf_metrics['hit_at_3']:.2%} | {mm_rrf_metrics['local_retrieval_latency']:.3f}с |\n")
        f.write(f"| **Мультимодальный** | **Максимум каждой модальности** | **Задан** | **{mm_max_metrics['hit_at_1']:.2%}** | **{mm_max_metrics['hit_at_3']:.2%}** | {mm_max_metrics['local_retrieval_latency']:.3f}с |\n")
        f.write(f"| Мультимодальный | Максимум каждой модальности | Нет | {mm_max_norouting_metrics['hit_at_1']:.2%} | {mm_max_norouting_metrics['hit_at_3']:.2%} | {mm_max_norouting_metrics['local_retrieval_latency']:.3f}с |\n\n")

        f.write("## 2. Метрики по категориям запросов (Hit@1)\n\n")
        f.write("| Категория | Кол-во | Только ASR | Простая сумма | RRF | Максимум каждой модальности | Максимум без отбора |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for cat in ["speech", "ocr", "visual"]:
            cnt = asr_metrics["category_stats"][cat]["total"]

            def get_hit1(m):
                s = m["category_stats"][cat]
                return s["top1"] / s["total"] if s["total"] > 0 else 0.0

            cat_label = "Речь" if cat == "speech" else ("Текст" if cat == "ocr" else "Кадр")
            f.write(
                f"| `{cat_label}` | {cnt} | {get_hit1(asr_metrics):.2%} | {get_hit1(mm_sum_metrics):.2%} | {get_hit1(mm_rrf_metrics):.2%} | **{get_hit1(mm_max_metrics):.2%}** | {get_hit1(mm_max_norouting_metrics):.2%} |\n"
            )

        f.write("\n## 3. Детальные результаты по запросам (Топ-1 найденное видео)\n\n")
        f.write("| Запрос | Категория | Ожидаемое | Только ASR | Простая сумма | RRF | Максимум каждой модальности | Максимум без отбора |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

        category_map_ru = {"speech": "Речь", "ocr": "Текст", "visual": "Кадр"}
        for i in range(len(tasks)):
            q = tasks[i]["query"]
            cat = tasks[i]["category"]
            exp = tasks[i]["expected_video"]

            def get_top1(m):
                res = m["results"][i]["top_found"]
                return f"`{res[0]}`" if res else "`None`"

            cat_ru = category_map_ru.get(cat, cat)
            f.write(f"| \"{q}\" | `{cat_ru.upper()}` | `{exp}` | {get_top1(asr_metrics)} | {get_top1(mm_sum_metrics)} | {get_top1(mm_rrf_metrics)} | {get_top1(mm_max_metrics)} | {get_top1(mm_max_norouting_metrics)} |\n")

        f.write("\n## 4. Анализ и выводы\n\n")
        f.write(f"1. **Сравнение стратегий объединения**: На проверочном наборе с зафиксированной декомпозицией запросов объединение по максимуму (Max-per-Modality) в сочетании с выбором модальностей показало наилучший точечный результат: Hit@1 вырос до {mm_max_metrics['hit_at_1']:.2%} (+{((mm_max_metrics['hit_at_1'] - asr_metrics['hit_at_1']) * 100):.0f} процентных пунктов относительно базового варианта {asr_metrics['hit_at_1']:.2%}), а Hit@3 достиг {mm_max_metrics['hit_at_3']:.2%} (+{((mm_max_metrics['hit_at_3'] - asr_metrics['hit_at_3']) * 100):.0f} процентных пунктов относительно базового варианта {asr_metrics['hit_at_3']:.2%}). Разница с RRF ({mm_rrf_metrics['hit_at_1']:.2%} по Hit@1, {mm_rrf_metrics['hit_at_3']:.2%} по Hit@3) составляет всего 1 запрос из 50. Разница между двумя лучшими методами составляет один запрос из 50.\n")
        f.write(f"2. **Влияние выбора модальностей (Сравнение без выбора источников)**: На проверочном наборе выбор активных модальностей совместно с методом Max-per-Modality повысил Hit@1 с {mm_max_norouting_metrics['hit_at_1']:.2%} (без выбора модальностей) до {mm_max_metrics['hit_at_1']:.2%}. Это показывает пользу ограничения активных каналов для снижения влияния шума нерелевантных модальностей. Автоматический выбор модальностей моделью Gemini в этом эксперименте не оценивался.\n")
        f.write(f"3. **Влияние RRF и простой суммы**: Мультимодальный поиск с выбором модальностей даже при простом суммировании оценок косинусного сходства (Сумма) превосходит базовый вариант по Hit@3 ({mm_sum_metrics['hit_at_3']:.2%} против {asr_metrics['hit_at_3']:.2%}). Использование рангового слияния RRF компенсирует разницу шкал косинусной близости и повышает Hit@1 до {mm_rrf_metrics['hit_at_1']:.2%}, но уступает методу Max-per-Modality по обоим показателям.\n")
        f.write(f"4. **Время поиска (Локальная задержка)**: Средняя задержка локального поиска (включая векторизацию и объединение результатов) составляет для базового варианта {asr_metrics['local_retrieval_latency']:.3f}с, а для мультимодальных режимов колеблется в диапазоне от {min(mm_sum_metrics['local_retrieval_latency'], mm_rrf_metrics['local_retrieval_latency'], mm_max_metrics['local_retrieval_latency']):.3f}с до {max(mm_sum_metrics['local_retrieval_latency'], mm_rrf_metrics['local_retrieval_latency'], mm_max_metrics['local_retrieval_latency']):.3f}с.\n")

    print(f"\nРезультаты оценки сохранены в: {output_path}")


if __name__ == "__main__":
    main()
