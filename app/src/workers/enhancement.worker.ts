import * as ort from 'onnxruntime-web/wasm';

import type {
  EnhancementParameters,
  EnhancementResult,
  ProcessingMetrics,
  TaskState,
  WorkerInputMessage,
  WorkerOutputMessage,
} from '../types/task';

interface WorkerContext {
  postMessage(message: WorkerOutputMessage): void;
  onmessage:
    | ((event: MessageEvent<WorkerInputMessage>) => void)
    | null;
  location: Location;
}

const workerContext = self as unknown as WorkerContext;

const MAX_PIXELS = 15_000_000;
const MODEL_INPUT_SIZE = 96;

const baseUrl = new URL(
  import.meta.env.BASE_URL,
  workerContext.location.origin,
);

const modelUrl = new URL(
  'models/enhancer.onnx',
  baseUrl,
).href;

ort.env.wasm.numThreads = 1;

ort.env.wasm.wasmPaths = {
  mjs: new URL(
    'wasm/ort-wasm-simd-threaded.mjs',
    baseUrl,
  ).href,
  wasm: new URL(
    'wasm/ort-wasm-simd-threaded.wasm',
    baseUrl,
  ).href,
};

let sessionPromise: Promise<ort.InferenceSession> | null =
  null;

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

function clamp(
  value: number,
  minimum: number,
  maximum: number,
): number {
  return Math.min(
    Math.max(value, minimum),
    maximum,
  );
}

function round(
  value: number,
  digits = 3,
): number {
  const multiplier = 10 ** digits;

  return Math.round(value * multiplier) / multiplier;
}

function getOutputType(
  file: File,
): 'image/jpeg' | 'image/png' {
  return file.type === 'image/png'
    ? 'image/png'
    : 'image/jpeg';
}

function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(
      modelUrl,
      {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      },
    );
  }

  return sessionPromise;
}

function createModelInput(
  bitmap: ImageBitmap,
): Float32Array {
  const canvas = new OffscreenCanvas(
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  const context = canvas.getContext('2d', {
    willReadFrequently: true,
  });

  if (!context) {
    throw new Error(
      'Не удалось подготовить изображение для модели.',
    );
  }

  context.fillStyle = '#ffffff';

  context.fillRect(
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  context.drawImage(
    bitmap,
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  const imageData = context.getImageData(
    0,
    0,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  const pixelCount =
    MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;

  const tensorData = new Float32Array(
    pixelCount * 3,
  );

  for (
    let pixelIndex = 0;
    pixelIndex < pixelCount;
    pixelIndex += 1
  ) {
    const dataIndex = pixelIndex * 4;

    tensorData[pixelIndex] =
      imageData.data[dataIndex] / 255;

    tensorData[pixelCount + pixelIndex] =
      imageData.data[dataIndex + 1] / 255;

    tensorData[pixelCount * 2 + pixelIndex] =
      imageData.data[dataIndex + 2] / 255;
  }

  return tensorData;
}

async function predictParameters(
  bitmap: ImageBitmap,
): Promise<EnhancementParameters> {
  const session = await getSession();
  const tensorData = createModelInput(bitmap);

  const inputTensor = new ort.Tensor(
    'float32',
    tensorData,
    [
      1,
      3,
      MODEL_INPUT_SIZE,
      MODEL_INPUT_SIZE,
    ],
  );

  const outputs = await session.run({
    image: inputTensor,
  });

  const outputTensor = outputs.parameters;

  if (!outputTensor) {
    throw new Error(
      'Модель не вернула параметры коррекции.',
    );
  }

  const values = outputTensor.data;

  if (values.length < 3) {
    throw new Error(
      'Модель вернула некорректный результат.',
    );
  }

  return {
    brightness: round(
      clamp(Number(values[0]), 0.8, 1.28),
    ),
    contrast: round(
      clamp(Number(values[1]), 0.88, 1.35),
    ),
    saturation: round(
      clamp(Number(values[2]), 0.88, 1.3),
    ),
  };
}

async function processImage(
  taskId: string,
  file: File,
): Promise<void> {
  const processStartedAt = performance.now();

  let bitmap: ImageBitmap | null = null;

  try {
    if (
      !['image/jpeg', 'image/png'].includes(file.type)
    ) {
      throw new Error(
        'В текущей версии поддерживаются только изображения JPG и PNG.',
      );
    }

    sendStatus(taskId, 'decoding', 10);

    const decodeStartedAt = performance.now();

    bitmap = await createImageBitmap(file);

    const decodeMs =
      performance.now() - decodeStartedAt;

    const totalPixels =
      bitmap.width * bitmap.height;

    if (totalPixels > MAX_PIXELS) {
      throw new Error(
        `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
          `Максимально допустимо ${MAX_PIXELS.toLocaleString()} пикселей.`,
      );
    }

    sendStatus(taskId, 'analyzing', 25);

    const inferenceStartedAt = performance.now();

    const parameters =
      await predictParameters(bitmap);

    const inferenceMs =
      performance.now() - inferenceStartedAt;

    sendStatus(taskId, 'enhancing', 50);

    const enhancementStartedAt =
      performance.now();

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

    const encodingStartedAt =
      performance.now();

    const outputType = getOutputType(file);

    const resultBlob =
      await canvas.convertToBlob({
        type: outputType,
        quality:
          outputType === 'image/jpeg'
            ? 0.92
            : undefined,
      });

    const encodingMs =
      performance.now() - encodingStartedAt;

    const totalMs =
      performance.now() - processStartedAt;

    const metrics: ProcessingMetrics = {
      width,
      height,
      megapixels: round(
        totalPixels / 1_000_000,
        2,
      ),
      decodeMs: round(decodeMs, 1),
      analysisMs: round(inferenceMs, 1),
      enhancementMs: round(
        enhancementMs,
        1,
      ),
      encodingMs: round(encodingMs, 1),
      totalMs: round(totalMs, 1),
    };

    const result: EnhancementResult = {
      blob: resultBlob,
      parameters,
      metrics,
    };

    sendStatus(taskId, 'completed', 100);

    workerContext.postMessage({
      type: 'result',
      taskId,
      result,
    });
  } catch (error) {
    if (bitmap) {
      bitmap.close();
    }

    const message =
      error instanceof Error
        ? error.message
        : 'Неизвестная ошибка обработки.';

    workerContext.postMessage({
      type: 'error',
      taskId,
      error: message,
    });
  }
}

workerContext.onmessage = (
  event: MessageEvent<WorkerInputMessage>,
): void => {
  if (event.data.type === 'process') {
    void processImage(
      event.data.taskId,
      event.data.file,
    );
  }
};