import {
  decodeImage,
  isBmpFile,
  isPngFile,
  isSupportedImageFile,
} from '../image/decode';

import {
  applyEnhancement,
} from '../image/applyEnhancement';

import {
  TinyEnhancementRuntime,
} from '../model/TinyEnhancementRuntime';

import type {
  EnhancementParameters,
  EnhancementResult,
  ProcessingMetrics,
  TaskState,
  WorkerInputMessage,
  WorkerOutputMessage,
} from '../types/task';


interface WorkerContext {
  postMessage(
    message: WorkerOutputMessage,
  ): void;

  onmessage:
    | ((
        event: MessageEvent<WorkerInputMessage>,
      ) => void)
    | null;

  location: Location;
}


const workerContext =
  self as unknown as WorkerContext;

const MAX_PIXELS = 15_000_000;

const MODEL_INPUT_SIZE = 96;

const baseUrl = new URL(
  import.meta.env.BASE_URL,
  workerContext.location.origin,
);

let runtimePromise:
  Promise<TinyEnhancementRuntime> | null = null;


function sendStatus(
  taskId: string,
  status: TaskState,
  progress: number,
): void {
  workerContext.postMessage(
    {
      type: 'status',
      taskId,
      status,
      progress,
    },
  );
}


function round(
  value: number,
  digits = 3,
): number {
  const multiplier =
    10 ** digits;

  return (
    Math.round(
      value * multiplier,
    ) / multiplier
  );
}


function clamp(
  value: number,
  minimum: number,
  maximum: number,
): number {
  return Math.min(
    maximum,
    Math.max(
      minimum,
      value,
    ),
  );
}


function stabilizeParameters(
  parameters: EnhancementParameters,
): EnhancementParameters {
  const brightness = clamp(
    parameters.brightness,
    0.82,
    1.55,
  );

  const minimumContrast =
    brightness >= 1.3
      ? 0.9
      : brightness <= 0.9
        ? 0.9
        : 0.85;

  return {
    brightness,
    contrast: clamp(
      parameters.contrast,
      minimumContrast,
      1.35,
    ),
    saturation: clamp(
      parameters.saturation,
      0.85,
      1.35,
    ),
  };
}


function getOutputType(
  file: File,
): 'image/jpeg' | 'image/png' {
  if (
    isPngFile(file) ||
    isBmpFile(file)
  ) {
    return 'image/png';
  }

  return 'image/jpeg';
}


function getRuntime():
  Promise<TinyEnhancementRuntime> {
  if (!runtimePromise) {
    runtimePromise =
      TinyEnhancementRuntime.create(
        baseUrl,
      );
  }

  return runtimePromise;
}


function createModelInput(
  bitmap: ImageBitmap,
): Float32Array {
  const canvas =
    new OffscreenCanvas(
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    );

  const context =
    canvas.getContext(
      '2d',
      {
        willReadFrequently: true,
      },
    );

  if (!context) {
    throw new Error(
      'Не удалось подготовить изображение для модели.',
    );
  }

  const sourceSize =
    Math.min(
      bitmap.width,
      bitmap.height,
    );

  const sourceX =
    (
      bitmap.width -
      sourceSize
    ) / 2;

  const sourceY =
    (
      bitmap.height -
      sourceSize
    ) / 2;

  context.fillStyle =
    '#ffffff';

  context.fillRect(
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  context.drawImage(
    bitmap,
    sourceX,
    sourceY,
    sourceSize,
    sourceSize,
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  const imageData =
    context.getImageData(
      0,
      0,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    );

  const pixelCount =
    MODEL_INPUT_SIZE *
    MODEL_INPUT_SIZE;

  const tensorData =
    new Float32Array(
      pixelCount * 3,
    );

  for (
    let pixelIndex = 0;
    pixelIndex < pixelCount;
    pixelIndex += 1
  ) {
    const dataIndex =
      pixelIndex * 4;

    tensorData[pixelIndex] =
      imageData.data[
        dataIndex
      ] / 255;

    tensorData[
      pixelCount +
      pixelIndex
    ] =
      imageData.data[
        dataIndex + 1
      ] / 255;

    tensorData[
      pixelCount * 2 +
      pixelIndex
    ] =
      imageData.data[
        dataIndex + 2
      ] / 255;
  }

  return tensorData;
}


async function predictParameters(
  bitmap: ImageBitmap,
): Promise<EnhancementParameters> {
  const runtime =
    await getRuntime();

  const input =
    createModelInput(
      bitmap,
    );

  return runtime.predict(
    input,
  );
}


async function processImage(
  taskId: string,
  file: File,
): Promise<void> {
  const processStartedAt =
    performance.now();

  let bitmap:
    ImageBitmap | null = null;

  try {
    if (
      !isSupportedImageFile(file)
    ) {
      throw new Error(
        'Поддерживаются изображения JPG, PNG, BMP, HEIC и HEIF.',
      );
    }

    sendStatus(
      taskId,
      'decoding',
      10,
    );

    const decodeStartedAt =
      performance.now();

    bitmap =
      await decodeImage(
        file,
        MAX_PIXELS,
      );

    const decodeMs =
      performance.now() -
      decodeStartedAt;

    const totalPixels =
      bitmap.width *
      bitmap.height;

    if (
      totalPixels >
      MAX_PIXELS
    ) {
      throw new Error(
        `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
          `Максимально допустимо ${MAX_PIXELS.toLocaleString()} пикселей.`,
      );
    }

    sendStatus(
      taskId,
      'analyzing',
      25,
    );

    const inferenceStartedAt =
      performance.now();

    const predictedParameters =
      await predictParameters(
        bitmap,
      );

    const parameters =
      stabilizeParameters(
        predictedParameters,
      );

    const inferenceMs =
      performance.now() -
      inferenceStartedAt;

    sendStatus(
      taskId,
      'enhancing',
      50,
    );

    const enhancementStartedAt =
      performance.now();

    const canvas =
      new OffscreenCanvas(
        bitmap.width,
        bitmap.height,
      );

    const context =
      canvas.getContext(
        '2d',
        {
          willReadFrequently: true,
        },
      );

    if (!context) {
      throw new Error(
        'Не удалось создать контекст обработки изображения.',
      );
    }

    applyEnhancement(
      context,
      bitmap,
      parameters,
    );

    const enhancementMs =
      performance.now() -
      enhancementStartedAt;

    sendStatus(
      taskId,
      'enhancing',
      78,
    );

    const width =
      bitmap.width;

    const height =
      bitmap.height;

    bitmap.close();
    bitmap = null;

    sendStatus(
      taskId,
      'encoding',
      88,
    );

    const encodingStartedAt =
      performance.now();

    const outputType =
      getOutputType(file);

    const resultBlob =
      await canvas.convertToBlob(
        {
          type: outputType,
          quality:
            outputType ===
            'image/jpeg'
              ? 0.92
              : undefined,
        },
      );

    const encodingMs =
      performance.now() -
      encodingStartedAt;

    const totalMs =
      performance.now() -
      processStartedAt;

    const metrics:
      ProcessingMetrics = {
        width,
        height,
        megapixels: round(
          totalPixels /
            1_000_000,
          2,
        ),
        decodeMs: round(
          decodeMs,
          1,
        ),
        analysisMs: round(
          inferenceMs,
          1,
        ),
        enhancementMs: round(
          enhancementMs,
          1,
        ),
        encodingMs: round(
          encodingMs,
          1,
        ),
        totalMs: round(
          totalMs,
          1,
        ),
      };

    const result:
      EnhancementResult = {
        blob: resultBlob,
        parameters,
        metrics,
      };

    workerContext.postMessage(
      {
        type: 'result',
        taskId,
        result,
      },
    );
  } catch (error) {
    if (bitmap) {
      bitmap.close();
    }

    const message =
      error instanceof Error
        ? error.message
        : 'Неизвестная ошибка обработки.';

    workerContext.postMessage(
      {
        type: 'error',
        taskId,
        error: message,
      },
    );
  }
}


workerContext.onmessage = (
  event:
    MessageEvent<WorkerInputMessage>,
): void => {
  if (
    event.data.type ===
    'process'
  ) {
    void processImage(
      event.data.taskId,
      event.data.file,
    );
  }
};