import './style.css';
import { ImageEnhancer } from './api/ImageEnhancer';
import type { TaskStatus } from './types/task';

const enhancer = new ImageEnhancer();

let selectedFile: File | null = null;
let currentTaskId: string | null = null;
let sourceObjectUrl: string | null = null;
let resultObjectUrl: string | null = null;

function getRequiredElement<T extends Element>(selector: string): T {
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
        Первый MVP: асинхронная обработка JPG и PNG в Web Worker.
      </p>
    </header>

    <section class="panel controls">
      <label class="file-picker">
        <span>Выбрать изображение</span>

        <input
          id="file-input"
          type="file"
          accept="image/jpeg,image/png"
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
  getRequiredElement<HTMLParagraphElement>('#source-placeholder');

const resultImage =
  getRequiredElement<HTMLImageElement>('#result-image');

const resultPlaceholder =
  getRequiredElement<HTMLParagraphElement>('#result-placeholder');

const downloadLink =
  getRequiredElement<HTMLAnchorElement>('#download-link');

const statusNames: Record<TaskStatus['status'], string> = {
  queued: 'Задача поставлена в очередь',
  decoding: 'Декодирование изображения',
  enhancing: 'Применение коррекции',
  encoding: 'Кодирование результата',
  completed: 'Обработка завершена',
  cancelled: 'Задача отменена',
  failed: 'Ошибка обработки',
};

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

  const allowedTypes = ['image/jpeg', 'image/png'];

  if (!allowedTypes.includes(file.type)) {
    selectedFile = null;
    fileInput.value = '';

    fileInfo.textContent = 'Файл пока не выбран';
    statusText.textContent = 'Неподдерживаемый формат';

    showError(
      'В текущей версии поддерживаются только изображения JPG и PNG.',
    );

    setProcessingState(false);
    return;
  }

  const sizeMb = file.size / 1024 / 1024;

  fileInfo.textContent =
    `${file.name} · ${sizeMb.toFixed(2)} МБ · ${file.type}`;

  statusText.textContent = 'Изображение готово к обработке';

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

    resultObjectUrl = URL.createObjectURL(result);

    resultImage.src = resultObjectUrl;
    resultImage.hidden = false;
    resultPlaceholder.hidden = true;

    const extension =
      result.type === 'image/png' ? 'png' : 'jpg';

    downloadLink.href = resultObjectUrl;
    downloadLink.download = `enhanced-image.${extension}`;
    downloadLink.hidden = false;
  } catch (error) {
    const message =
      error instanceof Error
        ? error.message
        : 'Неизвестная ошибка обработки.';

    showError(message);
  } finally {
    setProcessingState(false);
  }
});

cancelButton.addEventListener('click', async () => {
  if (!currentTaskId) {
    return;
  }

  try {
    const cancelled = await enhancer.cancel(currentTaskId);

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

enhancer.addEventListener('statuschange', (event: Event) => {
  const customEvent = event as CustomEvent<TaskStatus>;
  const status = customEvent.detail;

  if (currentTaskId && status.taskId !== currentTaskId) {
    return;
  }

  statusText.textContent = statusNames[status.status];
  progressText.textContent = `${status.progress}%`;
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
});

window.addEventListener('beforeunload', () => {
  if (sourceObjectUrl) {
    URL.revokeObjectURL(sourceObjectUrl);
  }

  if (resultObjectUrl) {
    URL.revokeObjectURL(resultObjectUrl);
  }
});