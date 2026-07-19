import type {
  EnhancementParameters,
} from '../types/task';


function clampByte(
  value: number,
): number {
  return Math.min(
    255,
    Math.max(
      0,
      Math.round(value),
    ),
  );
}


export function enhancePixels(
  source: Uint8ClampedArray,
  parameters: EnhancementParameters,
): Uint8ClampedArray {
  if (source.length % 4 !== 0) {
    throw new Error(
      'Некорректный массив пикселей.',
    );
  }

  const result =
    new Uint8ClampedArray(
      source.length,
    );

  const brightness =
    parameters.brightness;

  const contrast =
    parameters.contrast;

  const saturation =
    parameters.saturation;

  const redFromRed =
    0.213 + 0.787 * saturation;

  const redFromGreen =
    0.715 - 0.715 * saturation;

  const redFromBlue =
    0.072 - 0.072 * saturation;

  const greenFromRed =
    0.213 - 0.213 * saturation;

  const greenFromGreen =
    0.715 + 0.285 * saturation;

  const greenFromBlue =
    0.072 - 0.072 * saturation;

  const blueFromRed =
    0.213 - 0.213 * saturation;

  const blueFromGreen =
    0.715 - 0.715 * saturation;

  const blueFromBlue =
    0.072 + 0.928 * saturation;

  for (
    let index = 0;
    index < source.length;
    index += 4
  ) {
    const brightRed =
      source[index] * brightness;

    const brightGreen =
      source[index + 1] * brightness;

    const brightBlue =
      source[index + 2] * brightness;

    const contrastedRed =
      (
        brightRed - 128
      ) *
        contrast +
      128;

    const contrastedGreen =
      (
        brightGreen - 128
      ) *
        contrast +
      128;

    const contrastedBlue =
      (
        brightBlue - 128
      ) *
        contrast +
      128;

    const red =
      redFromRed * contrastedRed +
      redFromGreen * contrastedGreen +
      redFromBlue * contrastedBlue;

    const green =
      greenFromRed * contrastedRed +
      greenFromGreen * contrastedGreen +
      greenFromBlue * contrastedBlue;

    const blue =
      blueFromRed * contrastedRed +
      blueFromGreen * contrastedGreen +
      blueFromBlue * contrastedBlue;

    result[index] =
      clampByte(red);

    result[index + 1] =
      clampByte(green);

    result[index + 2] =
      clampByte(blue);

    result[index + 3] =
      source[index + 3];
  }

  return result;
}