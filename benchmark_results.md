# Результаты тестирования поиска по видео (Benchmark)

Сравнение стратегий поиска и алгоритмов слияния результатов на тестовом наборе из 50 запросов (20 речевых, 10 OCR, 20 визуальных) по 31 видеофайлу:
- **ASR-only**: поиск только по аудиодорожке исходного запроса (Baseline)
- **Multimodal (Sum)**: гибридный поиск с простым суммированием сырых косинусных расстояний и фиксированным Query Routing
- **Multimodal (RRF)**: гибридный поиск с ранговым слиянием Reciprocal Rank Fusion и фиксированным Query Routing
- **Multimodal (Max-per-Modality)**: гибридный поиск с объединением взвешенного максимума по каждой модальности и фиксированным Query Routing
- **Multimodal (Max-per-Modality, No Routing)**: гибридный поиск Max-per-Modality без отключения модальностей (все каналы активны для исходного запроса)

> [!IMPORTANT]
> Бенчмарк оценивает качество этапа извлечения (retrieval) и слияния (fusion) на тестовом наборе из 50 запросов при фиксированной (записанной вручную) декомпозиции запросов. Качество автоматического Query Router на базе Gemini в данный benchmark не входит. Метрика Hit@K рассчитывается исключительно на уровне документов (видеофайлов), а не временных интервалов.

## Параметры окружения и воспроизводимости

- **Git Commit**: `01c46089cd4be46134d36921853180ca163842a0`
- **Config SHA-256**: `9c304f39ba19` (файл `configs/config.yaml`)
- **Всего запросов**: 50
- **База векторов**: Qdrant (локальная база)

## 1. Сводные метрики (уровень видеофайлов)

| Стратегия | Метод слияния | Routing | Hit@1 | Hit@3 | Local retrieval pipeline latency |
| :--- | :--- | :---: | :---: | :---: | :---: |
| ASR-only | Sum | Нет | 68.00% | 82.00% | 0.305s |
| Multimodal | Sum | Fixed | 68.00% | 94.00% | 0.433s |
| Multimodal | RRF | Fixed | 86.00% | 94.00% | 0.480s |
| **Multimodal** | **Max-per-Modality** | **Fixed** | **88.00%** | **96.00%** | 0.450s |
| Multimodal | Max-per-Modality | Нет | 76.00% | 88.00% | 0.769s |

## 2. Метрики по категориям запросов (Hit@1)

| Категория | Кол-во | ASR-only | Multimodal (Sum) | Multimodal (RRF) | Multimodal (Max-per-Modality) | Multimodal (Max, No Routing) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `SPEECH` | 20 | 80.00% | 70.00% | 85.00% | **85.00%** | 90.00% |
| `OCR` | 10 | 70.00% | 60.00% | 80.00% | **90.00%** | 80.00% |
| `VISUAL` | 20 | 55.00% | 70.00% | 90.00% | **90.00%** | 60.00% |

## 3. Детальные результаты по запросам (Топ-1 найденное видео)

| Запрос | Категория | Ожидаемое | ASR-only | Multimodal (Sum) | Multimodal (RRF) | Multimodal (Max) | Multimodal (Max, No Routing) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| "В каком направлении нужно откручивать верхушку ананаса" | `SPEECH` | `ananas` | `ananas` | `ananas` | `ananas` | `ananas` | `ananas` |
| "Какая консистенция должна получиться у теста для блинов" | `SPEECH` | `blini` | `blini` | `sirniki` | `blini` | `blini` | `blini` |
| "Стилист Михаил Политковский в титрах" | `OCR` | `bomber` | `zhara` | `sirniki` | `kosuha` | `rubashka` | `kosuha` |
| "Мужчина танцует в зеленой куртке" | `VISUAL` | `bomber` | `borsch` | `bomber` | `bomber` | `bomber` | `trench` |
| "Разбитое яйцо в белой миске" | `VISUAL` | `blini` | `kulich` | `pancakes` | `blini` | `blini` | `blini` |
| "Почему российский или белорусский сыр не подходит для карбонары" | `SPEECH` | `carbonara` | `carbonara` | `blini` | `carbonara` | `carbonara` | `carbonara` |
| "Как сделать ямочку под узлом при завязывании галстука" | `SPEECH` | `galstuk` | `galstuk` | `galstuk` | `galstuk` | `galstuk` | `shnurki` |
| "Инструкция по завязыванию галстука с надписью Slide the Tip Through the Hole" | `OCR` | `galstuk` | `galstuk` | `galstuk` | `galstuk` | `galstuk` | `galstuk` |
| "Человек режет свеклу ножом на разделочной доске" | `VISUAL` | `borsch` | `taco` | `borsch` | `borsch` | `borsch` | `taco` |
| "Нарезка бекона на ярко-желтой доске" | `VISUAL` | `carbonara` | `carbonara` | `carbonara` | `carbonara` | `carbonara` | `carbonara` |
| "кто придумал куртку косуху в 1928 году" | `SPEECH` | `kosuha` | `bomber` | `bomber` | `kosuha` | `kosuha` | `kosuha` |
| "какое соотношение отжатого хлеба к мясу должно быть в идеальных котлетах" | `SPEECH` | `kotleti` | `kotleti` | `kotleti` | `kotleti` | `kotleti` | `kotleti` |
| "косуха Saint Laurent ЦУМ" | `OCR` | `kosuha` | `borsch` | `borsch` | `upload_9eb1c395_2026-05-23_13-32-54` | `kosuha` | `kosuha` |
| "белые кеды на синем фоне" | `VISUAL` | `kedi` | `kedi` | `kedi` | `kedi` | `kedi` | `kedi` |
| "человек жарит котлеты на сковороде" | `VISUAL` | `kotleti` | `kotleti` | `pancakes` | `sirniki` | `sirniki` | `kotleti` |
| "рецепт пышных и влажных куличей со сливками" | `SPEECH` | `kulich` | `kulich` | `kulich` | `kulich` | `kulich` | `kulich` |
| "выкручивать перегоревшую лампочку против часовой стрелки" | `SPEECH` | `lampochka` | `lampochka` | `lampochka` | `lampochka` | `lampochka` | `lampochka` |
| "КАК ПРАВИЛЬНО ЧИСТИТЬ БОТИНКИ ШАГ УБРАТЬ ПЫЛЬ" | `OCR` | `obuv` | `obuv` | `obuv` | `obuv` | `obuv` | `obuv` |
| "натирание лимонной цедры на терке" | `VISUAL` | `kulich` | `kulich` | `pelmeni` | `kulich` | `kulich` | `pelmeni` |
| "три лампочки лежат на столе" | `VISUAL` | `lampochka` | `lampochka` | `lampochka` | `lampochka` | `lampochka` | `lampochka` |
| "Сколько минут нужно варить пельмени после их всплытия?" | `SPEECH` | `pelmeni` | `pelmeni` | `pelmeni` | `pelmeni` | `pelmeni` | `pelmeni` |
| "Почему для пышности панкейков нужно взбивать белки и желтки отдельно?" | `SPEECH` | `pancakes` | `pancakes` | `pancakes` | `pancakes` | `pancakes` | `pancakes` |
| "Надпись СПОРТИВНАЯ ОДЕЖДА на экране" | `OCR` | `palto` | `szhi` | `szhi` | `palto` | `palto` | `palto` |
| "Собака в поварском колпаке перед тарелкой с едой" | `VISUAL` | `pelmeni` | `pelmeni` | `shproti` | `shproti` | `shproti` | `pelmeni` |
| "Стопка панкейков на тарелке рядом с банкой меда" | `VISUAL` | `pancakes` | `pancakes` | `pancakes` | `pancakes` | `pancakes` | `pancakes` |
| "сколько грамм сухих дрожжей нужно для теста на пиццу" | `SPEECH` | `pizza` | `pizza` | `pizza` | `pizza` | `pizza` | `pizza` |
| "как правильно варить желтки с сахаром для домашнего пломбира чтобы они не свернулись" | `SPEECH` | `plombir` | `pancakes` | `pancakes` | `pancakes` | `pancakes` | `plombir` |
| "надпись обход блокировки на экране" | `OCR` | `roblox` | `roblox` | `roblox` | `roblox` | `roblox` | `roblox` |
| "человек нарезает круглую пиццу ножом на пергаменте" | `VISUAL` | `pizza` | `pizza` | `pizza` | `pizza` | `pizza` | `pizza` |
| "накладывание шарика белого мороженого специальной ложкой в миску" | `VISUAL` | `plombir` | `sirniki` | `plombir` | `plombir` | `plombir` | `sirniki` |
| "Сколько времени нужно держать рубашку в пластиковом пакете перед глажкой?" | `SPEECH` | `rubashka` | `plombir` | `plombir` | `pizza` | `tea` | `pizza` |
| "Как правильно собирать домашние шампиньоны - выкручивать или срезать?" | `SPEECH` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` |
| "Надпись Как правильно завязывать шнурки на экране" | `OCR` | `shnurki` | `shnurki` | `shnurki` | `shnurki` | `shnurki` | `shnurki` |
| "Мужчина в коричневом свитере гладит белую ткань на столе" | `VISUAL` | `rubashka` | `taco` | `rubashka` | `rubashka` | `rubashka` | `rubashka` |
| "Грибы шампиньоны в коробке на столе" | `VISUAL` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` | `shampinioni` |
| "Методом проб и ошибок вывел рецепт идеальных сырников" | `SPEECH` | `sirniki` | `sirniki` | `sirniki` | `sirniki` | `sirniki` | `sirniki` |
| "Как вести себя на первом свидании" | `SPEECH` | `svidanie` | `svidanie` | `svidanie` | `svidanie` | `svidanie` | `svidanie` |
| "Рецепты на YouTube Калнина Наталья творог манная крупа" | `OCR` | `sirniki` | `sirniki` | `sirniki` | `sirniki` | `sirniki` | `sirniki` |
| "Собака в поварском колпаке и тарелка с едой" | `VISUAL` | `shproti` | `taco` | `shproti` | `shproti` | `shproti` | `pelmeni` |
| "Женщина в белой рубашке и черных брюках" | `VISUAL` | `svidanie` | `rubashka` | `svidanie` | `svidanie` | `svidanie` | `palto` |
| "Приветствие Евгения на кулинарном канале Макареич Кичин" | `SPEECH` | `taco` | `shproti` | `blini` | `shproti` | `shproti` | `taco` |
| "Как правильно заваривать чай методом проливов" | `SPEECH` | `tea` | `tea` | `tea` | `tea` | `tea` | `tea` |
| "Рекомендация по температуре воды для заваривания черного чая" | `OCR` | `tea` | `tea` | `tea` | `tea` | `tea` | `tea` |
| "Человек перемешивает нашинкованную капусту в миске" | `VISUAL` | `szhi` | `taco` | `borsch` | `szhi` | `szhi` | `taco` |
| "Нарезка репчатого лука кубиками на деревянной доске" | `VISUAL` | `taco` | `taco` | `borsch` | `taco` | `taco` | `taco` |
| "Кто изобрел плащ тренч согласно истории?" | `SPEECH` | `trench` | `trench` | `trench` | `trench` | `trench` | `trench` |
| "Исследование влияния сообщений Банка России на динамику финансовых рынков" | `SPEECH` | `upload_9eb1c395_2026-05-23_13-32-54` | `upload_9eb1c395_2026-05-23_13-32-54` | `upload_9eb1c395_2026-05-23_13-32-54` | `upload_9eb1c395_2026-05-23_13-32-54` | `upload_9eb1c395_2026-05-23_13-32-54` | `upload_9eb1c395_2026-05-23_13-32-54` |
| "Костюм Calvin Klein 205W39NYC и бренд Burbery в описании" | `OCR` | `trench` | `trench` | `sirniki` | `trench` | `trench` | `kosuha` |
| "Кошка лежит на деревянной скамейке" | `VISUAL` | `zhara` | `taco` | `zhara` | `zhara` | `zhara` | `kulich` |
| "Женщина сидит за столом с бокалом вина" | `VISUAL` | `vino` | `vino` | `vino` | `vino` | `vino` | `vino` |

## 4. Анализ и выводы

1. **Преимущество Max-per-Modality и Query Routing**: На текущем development-наборе с зафиксированной декомпозицией запросов Max-per-Modality в сочетании с Query Routing показал наилучший результат: Hit@1 вырос до 88.00% (прирост на 20% относительно baseline 68.00%), а Hit@3 достиг 96.00% (прирост на 14% относительно baseline 82.00%). Данный результат является предварительным и требует проверки на независимой выборке.
2. **Влияние Query Routing (Абляция)**: Без использования Query Routing (режим Multimodal Max-per-Modality без роутинга) метрика Hit@1 падает до 76.00%, а Hit@3 составляет 88.00%. Это наглядно демонстрирует, что ограничение активных модальностей по типу запроса критически важно для фильтрации шума нерелевантных каналов (например, исключения visual-поиска на чисто текстовые вопросы).
3. **Влияние RRF и Sum**: Переход к мультимодальному поиску с Query Routing даже при простом суммировании косинусных расстояний (Sum) превосходит ASR-only baseline по Hit@3 (94.00% против 82.00%). Использование рангового слияния RRF компенсирует разницу шкал и повышает Hit@1 до 86.00%, но уступает Max-per-Modality из-за вытеснения релевантных кандидатов в нижнюю часть выдачи при фиксированном слиянии.
4. **Задержка поиска (Local Latency)**: Средняя задержка локального поиска (включая векторизацию и слияние) составляет для baseline 0.305s, а для мультимодальных режимов колеблется в диапазоне от 0.433s до 0.480s.
