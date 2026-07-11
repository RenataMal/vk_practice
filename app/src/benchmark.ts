import './style.css';
import './benchmark.css';

import { ImageEnhancer } from './api/ImageEnhancer';
import type {
  EnhancementParameters,
  ProcessingMetrics,
  TaskStatus,
} from './types/task';

interface BenchmarkCase {
  name: string;
  width: number;
  height: number;
}

interface BenchmarkResult {
  caseName: string;
  run: number;
  width: number;
  height: number;
  megapixels: number;
  sourceSizeMb: number;
  resultSizeMb: number;
  wallTimeMs: number;
  parameters: EnhancementParameters;
  metrics: ProcessingMetrics;
}

const benchmarkCases: BenchmarkCase[] = [
  {
    name: '1 Мп',
    width: 1000,
    height: 1000,
  },
  {
    name: '5 Мп',
    width: 2500,
    height: 2000,
  },
  {
    name: '10 Мп',
    width: 4000,
    height: 2500,
  },
  {
    name: '15 Мп',
    width: 5000,
    height: 3000,
  },
];

const enhancer = new ImageEnhancer();

let results: BenchmarkResult[] = [];
let currentTaskId: string | null = null;

function getRequiredElement<T extends Element>(
  selector: string,
): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Элемент ${selector} не найден.`);
  }

  return element;
}

function formatTime(milliseconds: number): string {
  if (milliseconds >= 1000) {
    return `${(milliseconds / 1000).toFixed(2)} с`;
  }

  return `${milliseconds.toFixed(1)} мс`;
}

function formatNumber(value: number): string {
  return value.toFixed(2);
}

function calculateAverage(
  values: number[],
): number {
  if (values.length === 0) {
    return 0;
  }

  return (
    values.reduce(
      (total, value) => total + value,
      0,
    ) / values.length
  );
}

function calculateMaximum(
  values: number[],
): number {
  if (values.length === 0) {
    return 0;
  }

  return Math.max(...values);
}

async function canvasToFile(
  canvas: HTMLCanvasElement,
  filename: string,
): Promise<File> {
  const blob = await new Promise<Blob>(
    (resolve, reject) => {
      canvas.toBlob(
        (result) => {
          if (!result) {
            reject(
              new Error(
                'Не удалось создать тестовое изображение.',
              ),
            );

            return;
          }

          resolve(result);
        },
        'image/jpeg',
        0.9,
      );
    },
  );

  return new File(
    [blob],
    filename,
    {
      type: 'image/jpeg',
    },
  );
}

async function createSyntheticImage(
  benchmarkCase: BenchmarkCase,
  run: number,
): Promise<File> {
  const canvas = document.createElement('canvas');

  canvas.width = benchmarkCase.width;
  canvas.height = benchmarkCase.height;

  const context = canvas.getContext('2d');

  if (!context) {
    throw new Error(
      'Не удалось создать тестовое изображение.',
    );
  }

  const gradient = context.createLinearGradient(
    0,
    0,
    benchmarkCase.width,
    benchmarkCase.height,
  );

  gradient.addColorStop(0, '#203a8f');
  gradient.addColorStop(0.35, '#55a9db');
  gradient.addColorStop(0.7, '#f1c56e');
  gradient.addColorStop(1, '#9d405d');

  context.fillStyle = gradient;

  context.fillRect(
    0,
    0,
    benchmarkCase.width,
    benchmarkCase.height,
  );

  const shapeCount = 80;

  for (
    let index = 0;
    index < shapeCount;
    index += 1
  ) {
    const x =
      ((index * 137 + run * 31) %
        benchmarkCase.width);

    const y =
      ((index * 89 + run * 47) %
        benchmarkCase.height);

    const radius =
      Math.max(
        benchmarkCase.width,
        benchmarkCase.height,
      ) /
      (35 + (index % 10));

    context.beginPath();

    context.arc(
      x,
      y,
      radius,
      0,
      Math.PI * 2,
    );

    context.fillStyle =
      `hsla(${(index * 29) % 360}, 70%, 60%, 0.32)`;

    context.fill();
  }

  context.fillStyle =
    'rgba(255, 255, 255, 0.75)';

  context.font =
    `${Math.max(
      32,
      Math.round(
        benchmarkCase.width / 24,
      ),
    )}px Arial`;

  context.fillText(
    `${benchmarkCase.name} · запуск ${run}`,
    Math.round(
      benchmarkCase.width * 0.05,
    ),
    Math.round(
      benchmarkCase.height * 0.12,
    ),
  );

  return canvasToFile(
    canvas,
    `benchmark-${benchmarkCase.name}-${run}.jpg`,
  );
}

const app =
  getRequiredElement<HTMLDivElement>(
    '#benchmark-app',
  );

app.innerHTML = `
  <main class="page benchmark-page">
    <header class="header">
      <p class="eyebrow">VK Practice</p>
      <h1>Performance benchmark</h1>

      <p class="description">
        Проверка времени обработки изображений
        размером до 15 мегапикселей.
      </p>

      <a class="back-link" href="./">
        Вернуться к приложению
      </a>
    </header>

    <section class="panel benchmark-controls">
      <label class="benchmark-field">
        <span>Количество запусков для каждого размера</span>

        <select id="repetitions-select">
          <option value="1">1 запуск</option>
          <option value="2" selected>2 запуска</option>
          <option value="3">3 запуска</option>
        </select>
      </label>

      <div class="actions">
        <button
          id="start-button"
          type="button"
        >
          Запустить benchmark
        </button>

        <button
          id="download-button"
          type="button"
          class="secondary"
          disabled
        >
          Скачать CSV
        </button>
      </div>

      <div class="progress-section">
        <div class="status-row">
          <span id="status-text">
            Benchmark не запущен
          </span>

          <span id="progress-text">
            0%
          </span>
        </div>

        <progress
          id="progress-bar"
          value="0"
          max="100"
        ></progress>

        <p
          id="error-text"
          class="error"
          hidden
        ></p>
      </div>
    </section>

    <section
      id="summary-section"
      class="benchmark-summary"
      hidden
    >
      <article class="summary-card">
        <span>Среднее время</span>
        <strong id="average-time">—</strong>
      </article>

      <article class="summary-card">
        <span>Максимальное время</span>
        <strong id="maximum-time">—</strong>
      </article>

      <article class="summary-card">
        <span>Запусков</span>
        <strong id="runs-count">—</strong>
      </article>

      <article class="summary-card">
        <span>Требование ≤ 30 с</span>
        <strong id="requirement-status">—</strong>
      </article>
    </section>

    <section class="panel benchmark-results">
      <h2>Результаты</h2>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Размер</th>
              <th>Запуск</th>
              <th>Разрешение</th>
              <th>Мп</th>
              <th>Исходный файл</th>
              <th>Результат</th>
              <th>ML</th>
              <th>Коррекция</th>
              <th>Кодирование</th>
              <th>Итого</th>
              <th>Wall time</th>
            </tr>
          </thead>

          <tbody id="results-body">
            <tr>
              <td colspan="11">
                Результатов пока нет
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>
`;

const repetitionsSelect =
  getRequiredElement<HTMLSelectElement>(
    '#repetitions-select',
  );

const startButton =
  getRequiredElement<HTMLButtonElement>(
    '#start-button',
  );

const downloadButton =
  getRequiredElement<HTMLButtonElement>(
    '#download-button',
  );

const statusText =
  getRequiredElement<HTMLSpanElement>(
    '#status-text',
  );

const progressText =
  getRequiredElement<HTMLSpanElement>(
    '#progress-text',
  );

const progressBar =
  getRequiredElement<HTMLProgressElement>(
    '#progress-bar',
  );

const errorText =
  getRequiredElement<HTMLParagraphElement>(
    '#error-text',
  );

const summarySection =
  getRequiredElement<HTMLElement>(
    '#summary-section',
  );

const averageTime =
  getRequiredElement<HTMLElement>(
    '#average-time',
  );

const maximumTime =
  getRequiredElement<HTMLElement>(
    '#maximum-time',
  );

const runsCount =
  getRequiredElement<HTMLElement>(
    '#runs-count',
  );

const requirementStatus =
  getRequiredElement<HTMLElement>(
    '#requirement-status',
  );

const resultsBody =
  getRequiredElement<HTMLTableSectionElement>(
    '#results-body',
  );

function showError(message: string): void {
  errorText.textContent = message;
  errorText.hidden = false;
}

function clearError(): void {
  errorText.textContent = '';
  errorText.hidden = true;
}

function renderResults(): void {
  if (results.length === 0) {
    resultsBody.innerHTML = `
      <tr>
        <td colspan="11">
          Результатов пока нет
        </td>
      </tr>
    `;

    summarySection.hidden = true;
    downloadButton.disabled = true;

    return;
  }

  resultsBody.innerHTML = results
    .map(
      (result) => `
        <tr>
          <td>${result.caseName}</td>
          <td>${result.run}</td>
          <td>
            ${result.width} × ${result.height}
          </td>
          <td>
            ${formatNumber(result.megapixels)}
          </td>
          <td>
            ${formatNumber(result.sourceSizeMb)} МБ
          </td>
          <td>
            ${formatNumber(result.resultSizeMb)} МБ
          </td>
          <td>
            ${formatTime(
              result.metrics.analysisMs,
            )}
          </td>
          <td>
            ${formatTime(
              result.metrics.enhancementMs,
            )}
          </td>
          <td>
            ${formatTime(
              result.metrics.encodingMs,
            )}
          </td>
          <td>
            ${formatTime(
              result.metrics.totalMs,
            )}
          </td>
          <td>
            ${formatTime(
              result.wallTimeMs,
            )}
          </td>
        </tr>
      `,
    )
    .join('');

  const totalTimes = results.map(
    (result) => result.metrics.totalMs,
  );

  const average =
    calculateAverage(totalTimes);

  const maximum =
    calculateMaximum(totalTimes);

  averageTime.textContent =
    formatTime(average);

  maximumTime.textContent =
    formatTime(maximum);

  runsCount.textContent =
    String(results.length);

  requirementStatus.textContent =
    maximum <= 30_000
      ? 'Выполнено'
      : 'Не выполнено';

  requirementStatus.dataset.status =
    maximum <= 30_000
      ? 'success'
      : 'failure';

  summarySection.hidden = false;
  downloadButton.disabled = false;
}

function createCsv(): string {
  const headers = [
    'case_name',
    'run',
    'width',
    'height',
    'megapixels',
    'source_size_mb',
    'result_size_mb',
    'brightness',
    'contrast',
    'saturation',
    'decode_ms',
    'ml_inference_ms',
    'enhancement_ms',
    'encoding_ms',
    'total_ms',
    'wall_time_ms',
  ];

  const rows = results.map(
    (result) => [
      result.caseName,
      result.run,
      result.width,
      result.height,
      result.megapixels,
      result.sourceSizeMb,
      result.resultSizeMb,
      result.parameters.brightness,
      result.parameters.contrast,
      result.parameters.saturation,
      result.metrics.decodeMs,
      result.metrics.analysisMs,
      result.metrics.enhancementMs,
      result.metrics.encodingMs,
      result.metrics.totalMs,
      result.wallTimeMs,
    ],
  );

  return [
    headers.join(','),
    ...rows.map(
      (row) => row.join(','),
    ),
  ].join('\n');
}

downloadButton.addEventListener(
  'click',
  () => {
    if (results.length === 0) {
      return;
    }

    const csv = createCsv();

    const blob = new Blob(
      [`\uFEFF${csv}`],
      {
        type: 'text/csv;charset=utf-8',
      },
    );

    const url = URL.createObjectURL(blob);

    const link = document.createElement('a');

    link.href = url;
    link.download =
      `benchmark-${new Date()
        .toISOString()
        .replace(/[:.]/g, '-')}.csv`;

    link.click();

    URL.revokeObjectURL(url);
  },
);

startButton.addEventListener(
  'click',
  async () => {
    clearError();

    results = [];
    renderResults();

    const repetitions = Number(
      repetitionsSelect.value,
    );

    const totalRuns =
      benchmarkCases.length * repetitions;

    let completedRuns = 0;

    startButton.disabled = true;
    repetitionsSelect.disabled = true;
    downloadButton.disabled = true;

    progressBar.value = 0;
    progressText.textContent = '0%';

    try {
      for (
        const benchmarkCase
        of benchmarkCases
      ) {
        for (
          let run = 1;
          run <= repetitions;
          run += 1
        ) {
          statusText.textContent =
            `Создание ${benchmarkCase.name}, запуск ${run}`;

          const file =
            await createSyntheticImage(
              benchmarkCase,
              run,
            );

          statusText.textContent =
            `Обработка ${benchmarkCase.name}, запуск ${run}`;

          currentTaskId =
            await enhancer.submit(file);

          const wallStartedAt =
            performance.now();

          const result =
            await enhancer.getResult(
              currentTaskId,
            );

          const wallTimeMs =
            performance.now() -
            wallStartedAt;

          results.push({
            caseName: benchmarkCase.name,
            run,
            width: result.metrics.width,
            height: result.metrics.height,
            megapixels:
              result.metrics.megapixels,
            sourceSizeMb:
              file.size / 1024 / 1024,
            resultSizeMb:
              result.blob.size / 1024 / 1024,
            wallTimeMs,
            parameters:
              result.parameters,
            metrics:
              result.metrics,
          });

          completedRuns += 1;

          const progress =
            Math.round(
              completedRuns /
                totalRuns *
                100,
            );

          progressBar.value = progress;
          progressText.textContent =
            `${progress}%`;

          renderResults();
        }
      }

      statusText.textContent =
        'Benchmark завершён';
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : 'Ошибка benchmark.';

      showError(message);

      statusText.textContent =
        'Benchmark завершился с ошибкой';
    } finally {
      currentTaskId = null;
      startButton.disabled = false;
      repetitionsSelect.disabled = false;
    }
  },
);

enhancer.addEventListener(
  'statuschange',
  (event: Event) => {
    const customEvent =
      event as CustomEvent<TaskStatus>;

    const status =
      customEvent.detail;

    if (
      currentTaskId &&
      status.taskId !== currentTaskId
    ) {
      return;
    }

    if (status.error) {
      showError(status.error);
    }
  },
);

window.addEventListener(
  'beforeunload',
  () => {
    enhancer.dispose();
  },
);
