import {
  afterAll,
  beforeAll,
  describe,
  expect,
  it,
  vi,
} from 'vitest';
import {
  readFileSync,
} from 'node:fs';
import {
  readFile,
} from 'node:fs/promises';

import {
  TinyEnhancementRuntime,
} from './TinyEnhancementRuntime';


interface ExpectedParameters {
  brightness: number;
  contrast: number;
  saturation: number;
}

interface ParityCase {
  name: string;
  expected: ExpectedParameters;
}

interface ParityFixture {
  inputSize: number;
  checkpoint: string;
  cases: ParityCase[];
}


const fixture = JSON.parse(
  readFileSync(
    new URL(
      './runtime-parity.fixture.json',
      import.meta.url,
    ),
    'utf-8',
  ),
) as ParityFixture;


function createInput(
  name: string,
  inputSize: number,
): Float32Array {
  const input = new Float32Array(
    3 * inputSize * inputSize,
  );

  for (
    let channel = 0;
    channel < 3;
    channel += 1
  ) {
    for (
      let y = 0;
      y < inputSize;
      y += 1
    ) {
      for (
        let x = 0;
        x < inputSize;
        x += 1
      ) {
        const index =
          (
            channel * inputSize + y
          ) *
            inputSize +
          x;

        if (name === 'zeros') {
          input[index] = 0;
        } else if (name === 'ones') {
          input[index] = 1;
        } else if (name === 'gradient') {
          input[index] =
            (
              (
                channel * 13 +
                y * 3 +
                x * 5
              ) %
              256
            ) /
            255;
        } else if (
          name === 'checkerboard'
        ) {
          input[index] =
            (
              Math.floor(x / 8) +
              Math.floor(y / 8) +
              channel
            ) %
            2;
        } else {
          throw new Error(
            `Неизвестный тестовый вход: ${name}.`,
          );
        }
      }
    }
  }

  return input;
}


describe(
  'TinyEnhancementRuntime parity',
  () => {
    let runtime: TinyEnhancementRuntime;

    beforeAll(async () => {
      const configText = await readFile(
        new URL(
          '../../public/models/model-config.json',
          import.meta.url,
        ),
        'utf-8',
      );

      const weightsBuffer = await readFile(
        new URL(
          '../../public/models/model-weights.bin',
          import.meta.url,
        ),
      );

      const weights = weightsBuffer.buffer.slice(
        weightsBuffer.byteOffset,
        weightsBuffer.byteOffset +
          weightsBuffer.byteLength,
      ) as ArrayBuffer;

      vi.stubGlobal(
        'fetch',
        vi.fn(
          async (
            input: string | URL | Request,
          ): Promise<Response> => {
            const url = String(input);

            if (
              url.endsWith(
                'model-config.json',
              )
            ) {
              return new Response(
                configText,
                {
                  status: 200,
                  headers: {
                    'Content-Type':
                      'application/json',
                  },
                },
              );
            }

            if (
              url.endsWith(
                'model-weights.bin',
              )
            ) {
              return new Response(
                weights,
                {
                  status: 200,
                  headers: {
                    'Content-Type':
                      'application/octet-stream',
                  },
                },
              );
            }

            return new Response(
              null,
              {
                status: 404,
              },
            );
          },
        ),
      );

      runtime =
        await TinyEnhancementRuntime.create(
          new URL(
            'http://localhost/',
          ),
        );
    });

    afterAll(() => {
      vi.unstubAllGlobals();
    });

    it(
      'uses the expected final checkpoint',
      () => {
        expect(
          fixture.checkpoint,
        ).toBe(
          'best_balanced_synthetic_model.pt',
        );
      },
    );

    it.each(
      fixture.cases,
    )(
      'matches PyTorch for $name',
      (testCase: ParityCase) => {
        const actual = runtime.predict(
          createInput(
            testCase.name,
            fixture.inputSize,
          ),
        );

        expect(
          actual.brightness,
        ).toBeCloseTo(
          testCase.expected.brightness,
          2,
        );

        expect(
          actual.contrast,
        ).toBeCloseTo(
          testCase.expected.contrast,
          2,
        );

        expect(
          actual.saturation,
        ).toBeCloseTo(
          testCase.expected.saturation,
          2,
        );
      },
    );
  },
);