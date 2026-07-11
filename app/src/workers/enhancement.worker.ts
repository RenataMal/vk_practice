/// <reference lib="webworker" />

import type {
  WorkerInputMessage,
  WorkerOutputMessage,
  TaskState,
} from '../types/task';

const workerContext = self as DedicatedWorkerGlobalScope;

const MAX_PIXELS = 15_000_000;

function sendStatus(
  taskId: string,
  status: TaskState,
  progress: number,
): void {
  const message: WorkerOutputMessage = {
    type: 'status',
    taskId,
    status,
    progress,
  };

  workerContext.postMessage(message);
}

function getOutputType(file: File): 'image/jpeg' | 'image/png' {
  return file.type === 'image/png' ? 'image/png' : 'image/jpeg';
}

async function processImage(taskId: string, file: File): Promise<void> {
  let bitmap: ImageBitmap | null = null;

  try {
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      throw new Error(
        'В текущей версии поддерживаются только изображения JPG и PNG.',
      );
    }

    sendStatus(taskId, 'decoding', 10);

    bitmap = await createImageBitmap(file);

    const totalPixels = bitmap.width * bitmap.height;

    if (totalPixels > MAX_PIXELS) {
      throw new Error(
        `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
          `Максимально допустимо ${MAX_PIXELS.toLocaleString()} пикселей.`,
      );
    }

    sendStatus(taskId, 'enhancing', 35);

    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const context = canvas.getContext('2d');

    if (!context) {
      throw new Error('Не удалось создать контекст обработки изображения.');
    }

    /*
     * В первом MVP используем фиксированные коэффициенты.
     * Позже эти три значения будет предсказывать ML-модель.
     */
    const brightness = 1.05;
    const contrast = 1.08;
    const saturation = 1.08;

    context.filter = [
      `brightness(${brightness})`,
      `contrast(${contrast})`,
      `saturate(${saturation})`,
    ].join(' ');

    context.drawImage(bitmap, 0, 0);

    sendStatus(taskId, 'enhancing', 75);

    bitmap.close();
    bitmap = null;

    sendStatus(taskId, 'encoding', 85);

    const outputType = getOutputType(file);

    const resultBlob = await canvas.convertToBlob({
      type: outputType,
      quality: outputType === 'image/jpeg' ? 0.92 : undefined,
    });

    sendStatus(taskId, 'completed', 100);

    const resultMessage: WorkerOutputMessage = {
      type: 'result',
      taskId,
      blob: resultBlob,
    };

    workerContext.postMessage(resultMessage);
  } catch (error) {
    if (bitmap) {
      bitmap.close();
    }

    const message =
      error instanceof Error ? error.message : 'Неизвестная ошибка обработки.';

    const errorMessage: WorkerOutputMessage = {
      type: 'error',
      taskId,
      error: message,
    };

    workerContext.postMessage(errorMessage);
  }
}

workerContext.onmessage = (event: MessageEvent<WorkerInputMessage>): void => {
  const message = event.data;

  if (message.type === 'process') {
    void processImage(message.taskId, message.file);
  }
};

export {};
