# VK Practice = Browser Image Enhancer

Система автоматического улучшения изображений непосредственно в браузере пользователя.

ML-модель анализирует изображение и подбирает оптимальные коэффициенты:

- яркости;
- контрастности;
- цветности.

После инференса вспомогательный алгоритм применяет параметры к полноразмерному изображению в Web Worker без блокировки основного потока браузера.

## Демо

Основное приложение:

https://renatamal.github.io/vk_practice/

Performance benchmark:

https://renatamal.github.io/vk_practice/benchmark.html

## Возможности

- обработка изображений локально в браузере;
- отсутствие передачи пользовательских изображений на сервер;
- асинхронная обработка через Web Worker;
- отображение статуса и прогресса;
- возможность отмены задачи;
- очередь задач;
- автоматический выбор параметров коррекции;
- отображение технических метрик;
- скачивание результата;
- обработка изображений до 15 Мп;
- встроенный performance benchmark.

## Поддерживаемые форматы

| Входной формат | Выходной формат |
|---|---|
| JPG / JPEG | JPG |
| PNG | PNG |
| BMP | PNG |
| HEIC / HEIF | JPG |

Для BMP поддерживаются несжатые 24-битные и 32-битные изображения.

## Архитектура

```text
Пользователь выбирает изображение
                ↓
ImageEnhancer API создает задачу
                ↓
Задача помещается в очередь
                ↓
Web Worker декодирует изображение
                ↓
Изображение преобразуется в тензор 3 × 96 × 96
                ↓
Компактная CNN предсказывает 3 коэффициента
                ↓
Canvas применяет коррекцию к оригиналу
                ↓
Результат кодируется в JPG или PNG
                ↓
Пользователь скачивает готовое изображение
```

## JavaScript API

Основной класс:

```typescript
const enhancer = new ImageEnhancer();
```

Постановка задачи:

```typescript
const taskId = await enhancer.submit(file);
```

Получение статуса:

```typescript
const status = await enhancer.getStatus(taskId);
```

Получение результата:

```typescript
const result = await enhancer.getResult(taskId);
```

Отмена задачи:

```typescript
await enhancer.cancel(taskId);
```

Подписка на изменение статуса:

```typescript
enhancer.addEventListener(
  'statuschange',
  (event) => {
    console.log(event.detail);
  },
);
```

Статусы задачи:

```text
queued
decoding
analyzing
enhancing
encoding
completed
cancelled
failed
```

## ML-модель

Модель получает RGB-изображение размером `96 × 96` и возвращает три числа:

```text
brightness
contrast
saturation
```

Архитектура:

```text
Conv2D 3 → 8
ReLU
Conv2D 8 → 16
ReLU
Conv2D 16 → 24
ReLU
Conv2D 24 → 32
ReLU
Global Average Pooling
Linear 32 → 16
ReLU
Linear 16 → 3
Sigmoid
```

Диапазоны выходных параметров:

| Параметр | Минимум | Максимум |
|---|---:|---:|
| Яркость | 0.80 | 1.28 |
| Контрастность | 0.88 | 1.35 |
| Цветность | 0.88 | 1.30 |

Модель обучалась на синтетически искаженных изображениях.

Количество исходных изображений: `126`.

Разбиение:

| Выборка | Изображений |
|---|---:|
| Train | 88 |
| Validation | 19 |
| Test | 19 |

Результат тестирования после 5 эпох:

```text
Test loss: 0.10073
Test parameter MAE: 0.10005
```

## Компактный runtime

Стандартный ONNX Runtime Web занимал около 13 МБ и нарушал ограничение проекта.

Для уменьшения размера был реализован собственный TypeScript runtime, который выполняет только операции, используемые моделью:

- Conv2D;
- ReLU;
- Global Average Pooling;
- Linear;
- Sigmoid.

Размер файлов модели:

```text
model-config.json: около 4 КБ
model-weights.bin: около 52 КБ
```

## Размер решения

Production-сборка после добавления HEIC-декодера:

```text
около 3 МБ
```

Требование:

```text
не более 10 МБ
```

Требование выполнено.

## Performance benchmark

Benchmark выполнялся в браузере на синтетических изображениях размером:

- 1 Мп;
- 5 Мп;
- 10 Мп;
- 15 Мп.

Каждый размер обрабатывался три раза.

Итоговые результаты:

| Метрика | Значение |
|---|---:|
| Количество запусков | 12 |
| Среднее время | 384,1 мс |
| Максимальное время | 1,12 с |
| Максимальный размер | 15 Мп |

Требования:

| Требование | Результат |
|---|---|
| Среднее время до 5 секунд | выполнено |
| Максимальное время до 30 секунд | выполнено |
| Обработка до 15 Мп | выполнено |

Скорость зависит от устройства, браузера, формата и сложности кодирования изображения.

## Технологии

### Клиентская часть

- TypeScript;
- Vite;
- Web Worker;
- OffscreenCanvas;
- Canvas 2D;
- File API;
- ImageBitmap;
- WebAssembly HEIC decoder.

### ML

- Python;
- PyTorch;
- torchvision;
- NumPy;
- ONNX для проверки экспортируемой модели.

### Инфраструктура

- GitHub;
- GitHub Actions;
- GitHub Pages.

## Структура проекта

```text
vk_practice/
├── app/
│   ├── public/
│   │   └── models/
│   ├── src/
│   │   ├── api/
│   │   ├── image/
│   │   ├── model/
│   │   ├── types/
│   │   ├── workers/
│   │   ├── benchmark.ts
│   │   └── main.ts
│   ├── benchmark.html
│   ├── index.html
│   └── vite.config.ts
├── ml/
│   ├── data/
│   ├── artifacts/
│   └── src/
├── benchmark/
├── docs/
└── .github/
    └── workflows/
```

## Локальный запуск

Необходим Node.js 24 или совместимая LTS-версия.

```bash
cd app
npm install
npm run dev
```

Приложение будет доступно по адресу, указанному Vite.

Production-сборка:

```bash
npm run build
```

Локальная проверка production-сборки:

```bash
npm run preview
```

## Обучение модели

Активировать Python-окружение:

```bash
source .venv/Scripts/activate
```

Установить зависимости:

```bash
python -m pip install -r ml/requirements.txt
```

Добавить изображения в:

```text
ml/data/source
```

Запустить обучение:

```bash
python -m ml.src.train --epochs 20 --batch-size 16
```

Экспортировать веса для браузера:

```bash
python -m ml.src.export_browser_weights
```

## Ограничения

- BMP поддерживается только без сжатия с глубиной 24 или 32 бита.
- HEIC и HEIF преобразуются в JPG.
- Скорость зависит от производительности устройства.
- Обработка изображений более 15 Мп блокируется.
- Модель выполняет глобальную коррекцию изображения и не изменяет отдельные области независимо.
- Качество результата зависит от распределения изображений, использованных при обучении.

## Автор

Рената Малеванная

Практика VK, проект команды Почты Mail.