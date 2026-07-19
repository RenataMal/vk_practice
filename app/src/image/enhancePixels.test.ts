import {
  describe,
  expect,
  it,
} from 'vitest';

import {
  enhancePixels,
} from './enhancePixels';


describe(
  'enhancePixels',
  () => {
    it(
      'preserves pixels with neutral parameters',
      () => {
        const source =
          new Uint8ClampedArray([
            12,
            34,
            56,
            78,
            90,
            123,
            210,
            255,
          ]);

        const result = enhancePixels(
          source,
          {
            brightness: 1,
            contrast: 1,
            saturation: 1,
          },
        );

        expect(
          Array.from(result),
        ).toEqual(
          Array.from(source),
        );

        expect(result).not.toBe(source);
      },
    );

    it(
      'applies brightness',
      () => {
        const result = enhancePixels(
          new Uint8ClampedArray([
            100,
            50,
            25,
            128,
          ]),
          {
            brightness: 2,
            contrast: 1,
            saturation: 1,
          },
        );

        expect(
          Array.from(result),
        ).toEqual([
          200,
          100,
          50,
          128,
        ]);
      },
    );

    it(
      'applies contrast and clamps values',
      () => {
        const result = enhancePixels(
          new Uint8ClampedArray([
            100,
            150,
            200,
            255,
          ]),
          {
            brightness: 1,
            contrast: 2,
            saturation: 1,
          },
        );

        expect(
          Array.from(result),
        ).toEqual([
          72,
          172,
          255,
          255,
        ]);
      },
    );

    it(
      'converts color to grayscale with zero saturation',
      () => {
        const result = enhancePixels(
          new Uint8ClampedArray([
            255,
            0,
            0,
            64,
          ]),
          {
            brightness: 1,
            contrast: 1,
            saturation: 0,
          },
        );

        expect(
          Array.from(result),
        ).toEqual([
          54,
          54,
          54,
          64,
        ]);
      },
    );

    it(
      'rejects an invalid pixel array',
      () => {
        expect(
          () =>
            enhancePixels(
              new Uint8ClampedArray([
                1,
                2,
                3,
              ]),
              {
                brightness: 1,
                contrast: 1,
                saturation: 1,
              },
            ),
        ).toThrow(
          'Некорректный массив пикселей.',
        );
      },
    );
  },
);