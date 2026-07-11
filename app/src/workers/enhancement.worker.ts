/// <reference lib="webworker" />

import type {
  EnhancementParameters,
  EnhancementResult,
  ProcessingMetrics,
  TaskState,
  WorkerInputMessage,
  WorkerOutputMessage,
} from '../types/task';

const workerContext = self as DedicatedWorkerGlobalScope;

const MAX_PIXELS = 15_000_000;
const ANALYSIS_SIZE = 96;

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

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function round(value: number, digits = 3): number {
  const multiplier = 10 ** digits;
  return Math.round(value * multiplier) / multiplier;
}

function getOutputType(file: File): 'image/jpeg' | 'image/png' {
  return file.type === 'image/png' ? 'image/png' : 'image/jpeg';
}

function calculateParameters(
  imageData: ImageData,
): EnhancementParameters {
  const pixels = imageData.data;

  let validPixelCount = 0;
  let lumaSum = 0;
  let lumaSquaredSum = 0;
  let chromaSum = 0;

  for (let index = 0; index < pixels.length; index += 4) {
    const alpha = pixels[index + 3];

    if (alpha < 16) {
      continue;
    }

    const red = pixels[index] / 255;
    const green = pixels[index + 1] / 255;
    const blue = pixels[index + 2] / 255;

    const luma =
      0.2126 * red +
      0.7152 * green +
      0.0722 * blue;

    const maximum = Math.max(red, green, blue);
    const minimum = Math.min(red, green, blue);
    const chroma = maximum - minimum;

    lumaSum += luma;
    lumaSquaredSum += luma * luma;
    chromaSum += chroma;
    validPixelCount += 1;
  }

  if (validPixelCount === 0) {
    return {
      brightness: 1,
      contrast: 1,
      saturation: 1,
    };
  }

  const meanLuma = lumaSum / validPixelCount;

  const variance = Math.max(
    lumaSquaredSum / validPixelCount - meanLuma ** 2,
    0,
  );

  const standardDeviation = Math.sqrt(variance);
  const meanChroma = chromaSum / validPixelCount;

  let brightness = 1;
  let contrast = 1;
  let saturation = 1;

  if (meanLuma < 0.42) {
    brightness = clamp(
      0.5 / Math.max(meanLuma, 0.08),
      1,
      1.28,
    );
  } else if (meanLuma > 0.68) {
    brightness = clamp(0.58 / meanLuma, 0.8, 1);
  }

  if (standardDeviation < 0.16) {
    contrast = clamp(
      0.2 / Math.max(standardDeviation, 0.05),
      1,
      1.35,
    );
  } else if (standardDeviation > 0.31) {
    contrast = clamp(
      0.28 / standardDeviation,
      0.88,
      1,
    );
  }

  if (meanChroma < 0.12) {
    saturation = clamp(
      0.16 / Math.max(meanChroma, 0.04),
      1,
      1.3,
    );
  } else if (meanChroma > 0.34) {
    saturation = clamp(
      0.3 / meanChroma,
      0.88,
      1,
    );
  }

  return {
    brightness: round(brightness),
    contrast: round(contrast),
    saturation: round(saturation),
  };
}

function createAnalysisImageData(
  bitmap: ImageBitmap,
): ImageData {
  const analysisCanvas = new OffscreenCanvas(
    ANALYSIS_SIZE,
    ANALYSIS_SIZE,
  );

  const analysisContext = analysisCanvas.getContext('2d', {
    willReadFrequently: true,
  });

  if (!analysisContext) {
    throw new Error(
      'Не удалось создать контекст для анализа изображения.',
    );
  }

  analysisContext.drawImage(
    bitmap,
    0,
    0,
    ANALYSIS_SIZE,
    ANALYSIS_SIZE,
  );

  return analysisContext.getImageData(
    0,
    0,
    ANALYSIS_SIZE,
    ANALYSIS_SIZE,
  );
}

async function processImage(
  taskId: string,
  file: File,
): Promise<void> {
  const processStartedAt = performance.now();

  let bitmap: ImageBitmap | null = null;

  try {
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      throw new Error(
        'В текущей версии поддерживаются только изображения JPG и PNG.',
      );
    }

    sendStatus(taskId, 'decoding', 10);

    const decodeStartedAt = performance.now();

    bitmap = await createImageBitmap(file);

    const decodeMs = performance.now() - decodeStartedAt;

    const totalPixels = bitmap.width * bitmap.height;

    if (totalPixels > MAX_PIXELS) {
      throw new Error(
        `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
          `Максимально допустимо ${MAX_PIXELS.toLocaleString()} пикселей.`,
      );
    }

    sendStatus(taskId, 'analyzing', 28);

    const analysisStartedAt = performance.now();

    const analysisImageData = createAnalysisImageData(bitmap);
    const parameters = calculateParameters(analysisImageData);

    const analysisMs = performance.now() - analysisStartedAt;

    sendStatus(taskId, 'enhancing', 45);

    const enhancementStartedAt = performance.now();

    const canvas = new OffscreenCanvas(
      bitmap.width,
      bitmap.height,
    );

    const context = canvas.getContext('2d');

    if (!context) {
      throw new Error(
        'Не удалось создать контекст обработки изображения.',
      );
    }

    context.filter = [
      `brightness(${parameters.brightness})`,
      `contrast(${parameters.contrast})`,
      `saturate(${parameters.saturation})`,
    ].join(' ');

    context.drawImage(bitmap, 0, 0);

    const enhancementMs =
      performance.now() - enhancementStartedAt;

    sendStatus(taskId, 'enhancing', 78);

    const width = bitmap.width;
    const height = bitmap.height;

    bitmap.close();
    bitmap = null;

    sendStatus(taskId, 'encoding', 88);

    const encodingStartedAt = performance.now();

    const outputType = getOutputType(file);

    const resultBlob = await canvas.convertToBlob({
      type: outputType,
      quality: outputType === 'image/jpeg' ? 0.92 : undefined,
    });

    const encodingMs =
      performance.now() - encodingStartedAt;

    const totalMs =
      performance.now() - processStartedAt;

    const metrics: ProcessingMetrics = {
      width,
      height,
      megapixels: round(totalPixels / 1_000_000, 2),
      decodeMs: round(decodeMs, 1),
      analysisMs: round(analysisMs, 1),
      enhancementMs: round(enhancementMs, 1),
      encodingMs: round(encodingMs, 1),
      totalMs: round(totalMs, 1),
    };

    const result: EnhancementResult = {
      blob: resultBlob,
      parameters,
      metrics,
    };

    sendStatus(taskId, 'completed', 100);

    const resultMessage: WorkerOutputMessage = {
      type: 'result',
      taskId,
      result,
    };

    workerContext.postMessage(resultMessage);
  } catch (error) {
    if (bitmap) {
      bitmap.close();
    }

    const message =
      error instanceof Error
        ? error.message
        : 'Неизвестная ошибка обработки.';

    const errorMessage: WorkerOutputMessage = {
      type: 'error',
      taskId,
      error: message,
    };

    workerContext.postMessage(errorMessage);
  }
}

workerContext.onmessage = (
  event: MessageEvent<WorkerInputMessage>,
): void => {
  const message = event.data;

  if (message.type === 'process') {
    void processImage(message.taskId, message.file);
  }
};

export {};