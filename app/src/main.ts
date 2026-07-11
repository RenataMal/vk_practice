import './style.css';
import { ImageEnhancer } from './api/ImageEnhancer';
import type {
  EnhancementParameters,
  ProcessingMetrics,
  TaskStatus,
} from './types/task';

const enhancer = new ImageEnhancer();

let selectedFile: File | null = null;
let currentTaskId: string | null = null;
let sourceObjectUrl: string | null = null;
let resultObjectUrl: string | null = null;

function getRequiredElement<T extends Element>(
  selector: string,
): T {
  const element = document.querySelector<T>(selector);

  if (!element) {
    throw new Error(`Элемент ${selector} не найден.`);
  }

  return element;
}

const app = getRequiredElement<HTMLDivElement>('#app');

app.innerHTML = `
  <main class="page">
    <header class="header">
      <p class="eyebrow">VK Practice</p>
      <h1>Улучшение изображений в браузере</h1>

      <p class="description">
        Автоматический анализ яркости, контрастности и цветности
        без блокировки основного потока браузера.
      </p>
      <a class="back-link" href="./benchmark.html">
        Открыть performance benchmark
      </a>
    </header>

    <section class="panel controls">
      <label class="file-picker">
        <span>Выбрать изображение</span>

        <input
          id="file-input"
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/x-ms-bmp,.jpg,.jpeg,.png,.bmp"
        />
      </label>

      <p id="file-info" class="file-info">
        Файл пока не выбран
      </p>

      <div class="actions">
        <button id="process-button" type="button" disabled>
          Улучшить изображение
        </button>

        <button
          id="cancel-button"
          type="button"
          class="secondary"
          disabled
        >
          Отменить
        </button>
      </div>

      <div class="progress-section">
        <div class="status-row">
          <span id="status-text">Ожидание изображения</span>
          <span id="progress-text">0%</span>
        </div>

        <progress
          id="progress-bar"
          value="0"
          max="100"
        ></progress>

        <p id="error-text" class="error" hidden></p>
      </div>

      <section id="result-details" class="result-details" hidden>
        <h2>Результаты автоматического анализа</h2>

        <div class="parameter-grid">
          <article class="metric-card">
            <span>Яркость</span>
            <strong id="brightness-value">1.00×</strong>
          </article>

          <article class="metric-card">
            <span>Контрастность</span>
            <strong id="contrast-value">1.00×</strong>
          </article>

          <article class="metric-card">
            <span>Цветность</span>
            <strong id="saturation-value">1.00×</strong>
          </article>

          <article class="metric-card">
            <span>Время обработки</span>
            <strong id="total-time-value">0 мс</strong>
          </article>
        </div>

        <dl class="technical-metrics">
          <div>
            <dt>Разрешение</dt>
            <dd id="resolution-value">—</dd>
          </div>

          <div>
            <dt>Мегапиксели</dt>
            <dd id="megapixels-value">—</dd>
          </div>

          <div>
            <dt>Декодирование</dt>
            <dd id="decode-time-value">—</dd>
          </div>

          <div>
            <dt>ML-инференс</dt>
            <dd id="analysis-time-value">—</dd>
          </div>

          <div>
            <dt>Коррекция</dt>
            <dd id="enhancement-time-value">—</dd>
          </div>

          <div>
            <dt>Кодирование</dt>
            <dd id="encoding-time-value">—</dd>
          </div>
        </dl>
      </section>
    </section>

    <section class="comparison">
      <article class="image-card">
        <h2>Исходное изображение</h2>

        <div class="image-container">
          <img
            id="source-image"
            alt="Исходное изображение"
            hidden
          />

          <p id="source-placeholder">
            Изображение не выбрано
          </p>
        </div>
      </article>

      <article class="image-card">
        <h2>Результат</h2>

        <div class="image-container">
          <img
            id="result-image"
            alt="Обработанное изображение"
            hidden
          />

          <p id="result-placeholder">
            Результат ещё не готов
          </p>
        </div>

        <a
          id="download-link"
          class="download-link"
          hidden
        >
          Скачать результат
        </a>
      </article>
    </section>
  </main>
`;

const fileInput =
  getRequiredElement<HTMLInputElement>('#file-input');

const fileInfo =
  getRequiredElement<HTMLParagraphElement>('#file-info');

const processButton =
  getRequiredElement<HTMLButtonElement>('#process-button');

const cancelButton =
  getRequiredElement<HTMLButtonElement>('#cancel-button');

const statusText =
  getRequiredElement<HTMLSpanElement>('#status-text');

const progressText =
  getRequiredElement<HTMLSpanElement>('#progress-text');

const progressBar =
  getRequiredElement<HTMLProgressElement>('#progress-bar');

const errorText =
  getRequiredElement<HTMLParagraphElement>('#error-text');

const sourceImage =
  getRequiredElement<HTMLImageElement>('#source-image');

const sourcePlaceholder =
  getRequiredElement<HTMLParagraphElement>(
    '#source-placeholder',
  );

const resultImage =
  getRequiredElement<HTMLImageElement>('#result-image');

const resultPlaceholder =
  getRequiredElement<HTMLParagraphElement>(
    '#result-placeholder',
  );

const downloadLink =
  getRequiredElement<HTMLAnchorElement>('#download-link');

const resultDetails =
  getRequiredElement<HTMLElement>('#result-details');

const brightnessValue =
  getRequiredElement<HTMLElement>('#brightness-value');

const contrastValue =
  getRequiredElement<HTMLElement>('#contrast-value');

const saturationValue =
  getRequiredElement<HTMLElement>('#saturation-value');

const totalTimeValue =
  getRequiredElement<HTMLElement>('#total-time-value');

const resolutionValue =
  getRequiredElement<HTMLElement>('#resolution-value');

const megapixelsValue =
  getRequiredElement<HTMLElement>('#megapixels-value');

const decodeTimeValue =
  getRequiredElement<HTMLElement>('#decode-time-value');

const analysisTimeValue =
  getRequiredElement<HTMLElement>('#analysis-time-value');

const enhancementTimeValue =
  getRequiredElement<HTMLElement>(
    '#enhancement-time-value',
  );

const encodingTimeValue =
  getRequiredElement<HTMLElement>('#encoding-time-value');

const statusNames: Record<TaskStatus['status'], string> = {
  queued: 'Задача поставлена в очередь',
  decoding: 'Декодирование изображения',
  analyzing: 'Анализ изображения',
  enhancing: 'Применение коррекции',
  encoding: 'Кодирование результата',
  completed: 'Обработка завершена',
  cancelled: 'Задача отменена',
  failed: 'Ошибка обработки',
};

function formatMultiplier(value: number): string {
  return `${value.toFixed(2)}×`;
}

function formatTime(milliseconds: number): string {
  if (milliseconds >= 1000) {
    return `${(milliseconds / 1000).toFixed(2)} с`;
  }

  return `${milliseconds.toFixed(1)} мс`;
}

function clearError(): void {
  errorText.hidden = true;
  errorText.textContent = '';
}

function showError(message: string): void {
  errorText.textContent = message;
  errorText.hidden = false;
}

function resetProgress(): void {
  progressBar.value = 0;
  progressText.textContent = '0%';
}

function resetResult(): void {
  if (resultObjectUrl) {
    URL.revokeObjectURL(resultObjectUrl);
    resultObjectUrl = null;
  }

  resultImage.removeAttribute('src');
  resultImage.hidden = true;
  resultPlaceholder.hidden = false;

  downloadLink.hidden = true;
  downloadLink.removeAttribute('href');
  downloadLink.removeAttribute('download');

  resultDetails.hidden = true;
}

function resetSource(): void {
  if (sourceObjectUrl) {
    URL.revokeObjectURL(sourceObjectUrl);
    sourceObjectUrl = null;
  }

  sourceImage.removeAttribute('src');
  sourceImage.hidden = true;
  sourcePlaceholder.hidden = false;
}

function setProcessingState(isProcessing: boolean): void {
  processButton.disabled = isProcessing || !selectedFile;
  cancelButton.disabled = !isProcessing;
  fileInput.disabled = isProcessing;
}

function renderParameters(
  parameters: EnhancementParameters,
): void {
  brightnessValue.textContent =
    formatMultiplier(parameters.brightness);

  contrastValue.textContent =
    formatMultiplier(parameters.contrast);

  saturationValue.textContent =
    formatMultiplier(parameters.saturation);
}

function renderMetrics(metrics: ProcessingMetrics): void {
  totalTimeValue.textContent = formatTime(metrics.totalMs);

  resolutionValue.textContent =
    `${metrics.width} × ${metrics.height}`;

  megapixelsValue.textContent =
    `${metrics.megapixels.toFixed(2)} Мп`;

  decodeTimeValue.textContent =
    formatTime(metrics.decodeMs);

  analysisTimeValue.textContent =
    formatTime(metrics.analysisMs);

  enhancementTimeValue.textContent =
    formatTime(metrics.enhancementMs);

  encodingTimeValue.textContent =
    formatTime(metrics.encodingMs);
}

fileInput.addEventListener('change', () => {
  clearError();
  resetResult();
  resetSource();
  resetProgress();

  const file = fileInput.files?.[0] ?? null;

  selectedFile = file;
  currentTaskId = null;

  setProcessingState(false);

  if (!file) {
    fileInfo.textContent = 'Файл пока не выбран';
    statusText.textContent = 'Ожидание изображения';
    return;
  }

  const supportedMimeTypes = [
    'image/jpeg',
    'image/png',
    'image/bmp',
    'image/x-ms-bmp',
  ];

  const hasSupportedExtension =
    /\.(jpe?g|png|bmp)$/i.test(file.name);

  if (
    !supportedMimeTypes.includes(file.type) &&
    !hasSupportedExtension
  ) {
    selectedFile = null;
    fileInput.value = '';

    fileInfo.textContent = 'Файл пока не выбран';
    statusText.textContent = 'Неподдерживаемый формат';

    showError(
      'Поддерживаются изображения JPG, PNG и BMP.',
    );

    setProcessingState(false);
    return;
  }

  const sizeMb = file.size / 1024 / 1024;

  fileInfo.textContent =
    `${file.name} · ${sizeMb.toFixed(2)} МБ · ${file.type}`;

  statusText.textContent =
    'Изображение готово к обработке';

  sourceObjectUrl = URL.createObjectURL(file);

  sourceImage.src = sourceObjectUrl;
  sourceImage.hidden = false;
  sourcePlaceholder.hidden = true;

  setProcessingState(false);
});

processButton.addEventListener('click', async () => {
  if (!selectedFile) {
    return;
  }

  clearError();
  resetResult();
  resetProgress();

  statusText.textContent = 'Создание задачи';
  setProcessingState(true);

  try {
    currentTaskId = await enhancer.submit(selectedFile);

    const result = await enhancer.getResult(currentTaskId);

    resultObjectUrl = URL.createObjectURL(result.blob);

    resultImage.src = resultObjectUrl;
    resultImage.hidden = false;
    resultPlaceholder.hidden = true;

    renderParameters(result.parameters);
    renderMetrics(result.metrics);
    resultDetails.hidden = false;

    const extension =
      result.blob.type === 'image/png' ? 'png' : 'jpg';

    const originalName =
      selectedFile.name.replace(/\.[^.]+$/, '');

    downloadLink.href = resultObjectUrl;
    downloadLink.download =
      `${originalName}-enhanced.${extension}`;

    downloadLink.hidden = false;
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : 'Неизвестная ошибка обработки.';

    if (message !== 'Задача была отменена.') {
      showError(message);
    }
  } finally {
    setProcessingState(false);
  }
});

cancelButton.addEventListener('click', async () => {
  if (!currentTaskId) {
    return;
  }

  try {
    const cancelled =
      await enhancer.cancel(currentTaskId);

    if (cancelled) {
      statusText.textContent = 'Задача отменена';
    }
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : 'Не удалось отменить задачу.';

    showError(message);
  } finally {
    currentTaskId = null;
    setProcessingState(false);
  }
});

enhancer.addEventListener(
  'statuschange',
  (event: Event) => {
    const customEvent =
      event as CustomEvent<TaskStatus>;

    const status = customEvent.detail;

    if (
      currentTaskId &&
      status.taskId !== currentTaskId
    ) {
      return;
    }

    statusText.textContent =
      statusNames[status.status];

    progressText.textContent =
      `${status.progress}%`;

    progressBar.value = status.progress;

    if (status.error) {
      showError(status.error);
    }

    if (
      status.status === 'completed' ||
      status.status === 'cancelled' ||
      status.status === 'failed'
    ) {
      setProcessingState(false);
    }
  },
);

window.addEventListener('beforeunload', () => {
  enhancer.dispose();

  if (sourceObjectUrl) {
    URL.revokeObjectURL(sourceObjectUrl);
  }

  if (resultObjectUrl) {
    URL.revokeObjectURL(resultObjectUrl);
  }
});
