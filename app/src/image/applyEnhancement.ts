import {
  enhancePixels,
} from './enhancePixels';

import type {
  EnhancementParameters,
} from '../types/task';


let canvasFilterSupported:
  boolean | null = null;


function detectCanvasFilterSupport(): boolean {
  if (canvasFilterSupported !== null) {
    return canvasFilterSupported;
  }

  try {
    const sourceCanvas =
      new OffscreenCanvas(1, 1);

    const sourceContext =
      sourceCanvas.getContext('2d');

    const targetCanvas =
      new OffscreenCanvas(1, 1);

    const targetContext =
      targetCanvas.getContext(
        '2d',
        {
          willReadFrequently: true,
        },
      );

    if (
      !sourceContext ||
      !targetContext ||
      !('filter' in targetContext)
    ) {
      canvasFilterSupported = false;

      return false;
    }

    sourceContext.fillStyle =
      'rgb(255, 255, 255)';

    sourceContext.fillRect(
      0,
      0,
      1,
      1,
    );

    targetContext.filter =
      'brightness(0)';

    targetContext.drawImage(
      sourceCanvas,
      0,
      0,
    );

    const pixel =
      targetContext.getImageData(
        0,
        0,
        1,
        1,
      ).data;

    canvasFilterSupported =
      pixel[0] <= 1 &&
      pixel[1] <= 1 &&
      pixel[2] <= 1;

    return canvasFilterSupported;
  } catch {
    canvasFilterSupported = false;

    return false;
  }
}


function applyNativeEnhancement(
  context: OffscreenCanvasRenderingContext2D,
  bitmap: ImageBitmap,
  parameters: EnhancementParameters,
): void {
  context.filter = [
    `brightness(${parameters.brightness})`,
    `contrast(${parameters.contrast})`,
    `saturate(${parameters.saturation})`,
  ].join(' ');

  context.drawImage(
    bitmap,
    0,
    0,
  );

  context.filter = 'none';
}


function applyPixelEnhancement(
  context: OffscreenCanvasRenderingContext2D,
  bitmap: ImageBitmap,
  parameters: EnhancementParameters,
): void {
  context.filter = 'none';

  context.drawImage(
    bitmap,
    0,
    0,
  );

  const imageData =
    context.getImageData(
      0,
      0,
      bitmap.width,
      bitmap.height,
    );

  const enhancedPixels =
    enhancePixels(
      imageData.data,
      parameters,
    );

  imageData.data.set(
    enhancedPixels,
  );

  context.putImageData(
    imageData,
    0,
    0,
  );
}


export function applyEnhancement(
  context: OffscreenCanvasRenderingContext2D,
  bitmap: ImageBitmap,
  parameters: EnhancementParameters,
): void {
  if (detectCanvasFilterSupport()) {
    applyNativeEnhancement(
      context,
      bitmap,
      parameters,
    );

    return;
  }

  applyPixelEnhancement(
    context,
    bitmap,
    parameters,
  );
}