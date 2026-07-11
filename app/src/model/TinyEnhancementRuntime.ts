import type {
  EnhancementParameters,
} from '../types/task';

interface TensorDescriptor {
  offset: number;
  length: number;
  shape: number[];
}

interface ModelConfig {
  version: number;
  inputSize: number;
  parameterMin: number[];
  parameterMax: number[];
  tensors: Record<string, TensorDescriptor>;
}

interface FeatureMap {
  data: Float32Array;
  channels: number;
  height: number;
  width: number;
}

export class TinyEnhancementRuntime {
  private readonly config: ModelConfig;

  private readonly weights: Float32Array;

  private constructor(
    config: ModelConfig,
    weights: Float32Array,
  ) {
    this.config = config;
    this.weights = weights;
    this.validateWeights();
  }

  static async create(
    baseUrl: URL,
  ): Promise<TinyEnhancementRuntime> {
    const configUrl = new URL(
      'models/model-config.json',
      baseUrl,
    );

    const weightsUrl = new URL(
      'models/model-weights.bin',
      baseUrl,
    );

    const [
      configResponse,
      weightsResponse,
    ] = await Promise.all([
      fetch(configUrl),
      fetch(weightsUrl),
    ]);

    if (!configResponse.ok) {
      throw new Error(
        `Не удалось загрузить конфигурацию модели: ${configResponse.status}.`,
      );
    }

    if (!weightsResponse.ok) {
      throw new Error(
        `Не удалось загрузить веса модели: ${weightsResponse.status}.`,
      );
    }

    const config =
      await configResponse.json() as ModelConfig;

    const weightsBuffer =
      await weightsResponse.arrayBuffer();

    if (weightsBuffer.byteLength % 4 !== 0) {
      throw new Error(
        'Файл весов модели повреждён.',
      );
    }

    return new TinyEnhancementRuntime(
      config,
      new Float32Array(weightsBuffer),
    );
  }

  predict(
    inputData: Float32Array,
  ): EnhancementParameters {
    const expectedLength =
      3 *
      this.config.inputSize *
      this.config.inputSize;

    if (inputData.length !== expectedLength) {
      throw new Error(
        'Некорректный размер входа модели.',
      );
    }

    let featureMap: FeatureMap = {
      data: inputData,
      channels: 3,
      height: this.config.inputSize,
      width: this.config.inputSize,
    };

    featureMap = this.conv2dRelu(
      featureMap,
      'features.0.weight',
      'features.0.bias',
    );

    featureMap = this.conv2dRelu(
      featureMap,
      'features.2.weight',
      'features.2.bias',
    );

    featureMap = this.conv2dRelu(
      featureMap,
      'features.4.weight',
      'features.4.bias',
    );

    featureMap = this.conv2dRelu(
      featureMap,
      'features.6.weight',
      'features.6.bias',
    );

    const pooled =
      this.globalAveragePool(featureMap);

    const hidden = this.linear(
      pooled,
      'regressor.1.weight',
      'regressor.1.bias',
      true,
    );

    const normalized = this.linear(
      hidden,
      'regressor.3.weight',
      'regressor.3.bias',
      false,
    );

    const output = normalized.map(
      (value) => this.sigmoid(value),
    );

    return {
      brightness: this.round(
        this.scaleParameter(output[0], 0),
      ),
      contrast: this.round(
        this.scaleParameter(output[1], 1),
      ),
      saturation: this.round(
        this.scaleParameter(output[2], 2),
      ),
    };
  }

  private conv2dRelu(
    input: FeatureMap,
    weightName: string,
    biasName: string,
  ): FeatureMap {
    const weightDescriptor =
      this.getDescriptor(weightName);

    const weights =
      this.getTensor(weightName);

    const biases =
      this.getTensor(biasName);

    const [
      outputChannels,
      inputChannels,
      kernelHeight,
      kernelWidth,
    ] = weightDescriptor.shape;

    if (
      outputChannels === undefined ||
      inputChannels === undefined ||
      kernelHeight === undefined ||
      kernelWidth === undefined
    ) {
      throw new Error(
        `Некорректная форма тензора ${weightName}.`,
      );
    }

    if (input.channels !== inputChannels) {
      throw new Error(
        `Некорректное количество каналов слоя ${weightName}.`,
      );
    }

    const stride = 2;
    const padding = 1;

    const outputHeight =
      Math.floor(
        (
          input.height +
          2 * padding -
          kernelHeight
        ) / stride,
      ) + 1;

    const outputWidth =
      Math.floor(
        (
          input.width +
          2 * padding -
          kernelWidth
        ) / stride,
      ) + 1;

    const outputData = new Float32Array(
      outputChannels *
      outputHeight *
      outputWidth,
    );

    for (
      let outputChannel = 0;
      outputChannel < outputChannels;
      outputChannel += 1
    ) {
      for (
        let outputY = 0;
        outputY < outputHeight;
        outputY += 1
      ) {
        for (
          let outputX = 0;
          outputX < outputWidth;
          outputX += 1
        ) {
          let sum =
            biases[outputChannel] ?? 0;

          for (
            let inputChannel = 0;
            inputChannel < inputChannels;
            inputChannel += 1
          ) {
            for (
              let kernelY = 0;
              kernelY < kernelHeight;
              kernelY += 1
            ) {
              const inputY =
                outputY * stride +
                kernelY -
                padding;

              if (
                inputY < 0 ||
                inputY >= input.height
              ) {
                continue;
              }

              for (
                let kernelX = 0;
                kernelX < kernelWidth;
                kernelX += 1
              ) {
                const inputX =
                  outputX * stride +
                  kernelX -
                  padding;

                if (
                  inputX < 0 ||
                  inputX >= input.width
                ) {
                  continue;
                }

                const inputIndex =
                  (
                    inputChannel *
                    input.height +
                    inputY
                  ) *
                    input.width +
                  inputX;

                const weightIndex =
                  (
                    (
                      outputChannel *
                        inputChannels +
                      inputChannel
                    ) *
                      kernelHeight +
                    kernelY
                  ) *
                    kernelWidth +
                  kernelX;

                sum +=
                  (input.data[inputIndex] ?? 0) *
                  (weights[weightIndex] ?? 0);
              }
            }
          }

          const outputIndex =
            (
              outputChannel *
                outputHeight +
              outputY
            ) *
              outputWidth +
            outputX;

          outputData[outputIndex] =
            Math.max(0, sum);
        }
      }
    }

    return {
      data: outputData,
      channels: outputChannels,
      height: outputHeight,
      width: outputWidth,
    };
  }

  private globalAveragePool(
    input: FeatureMap,
  ): Float32Array {
    const output = new Float32Array(
      input.channels,
    );

    const spatialSize =
      input.height * input.width;

    for (
      let channel = 0;
      channel < input.channels;
      channel += 1
    ) {
      const channelOffset =
        channel * spatialSize;

      let sum = 0;

      for (
        let index = 0;
        index < spatialSize;
        index += 1
      ) {
        sum +=
          input.data[
            channelOffset + index
          ] ?? 0;
      }

      output[channel] =
        sum / spatialSize;
    }

    return output;
  }

  private linear(
    input: Float32Array,
    weightName: string,
    biasName: string,
    applyRelu: boolean,
  ): Float32Array {
    const descriptor =
      this.getDescriptor(weightName);

    const weights =
      this.getTensor(weightName);

    const biases =
      this.getTensor(biasName);

    const [
      outputFeatures,
      inputFeatures,
    ] = descriptor.shape;

    if (
      outputFeatures === undefined ||
      inputFeatures === undefined
    ) {
      throw new Error(
        `Некорректная форма тензора ${weightName}.`,
      );
    }

    if (input.length !== inputFeatures) {
      throw new Error(
        `Некорректный вход слоя ${weightName}.`,
      );
    }

    const output = new Float32Array(
      outputFeatures,
    );

    for (
      let outputIndex = 0;
      outputIndex < outputFeatures;
      outputIndex += 1
    ) {
      let sum =
        biases[outputIndex] ?? 0;

      const weightOffset =
        outputIndex * inputFeatures;

      for (
        let inputIndex = 0;
        inputIndex < inputFeatures;
        inputIndex += 1
      ) {
        sum +=
          (input[inputIndex] ?? 0) *
          (
            weights[
              weightOffset + inputIndex
            ] ?? 0
          );
      }

      output[outputIndex] =
        applyRelu
          ? Math.max(0, sum)
          : sum;
    }

    return output;
  }

  private scaleParameter(
    value: number,
    index: number,
  ): number {
    const minimum =
      this.config.parameterMin[index];

    const maximum =
      this.config.parameterMax[index];

    if (
      minimum === undefined ||
      maximum === undefined
    ) {
      throw new Error(
        'Некорректные границы параметров модели.',
      );
    }

    return minimum +
      value * (maximum - minimum);
  }

  private sigmoid(value: number): number {
    return 1 / (1 + Math.exp(-value));
  }

  private round(
    value: number,
    digits = 3,
  ): number {
    const multiplier = 10 ** digits;

    return (
      Math.round(value * multiplier) /
      multiplier
    );
  }

  private getDescriptor(
    name: string,
  ): TensorDescriptor {
    const descriptor =
      this.config.tensors[name];

    if (!descriptor) {
      throw new Error(
        `Тензор ${name} отсутствует.`,
      );
    }

    return descriptor;
  }

  private getTensor(
    name: string,
  ): Float32Array {
    const descriptor =
      this.getDescriptor(name);

    return this.weights.subarray(
      descriptor.offset,
      descriptor.offset +
        descriptor.length,
    );
  }

  private validateWeights(): void {
    for (
      const [
        name,
        descriptor,
      ] of Object.entries(
        this.config.tensors,
      )
    ) {
      const end =
        descriptor.offset +
        descriptor.length;

      if (
        descriptor.offset < 0 ||
        descriptor.length < 0 ||
        end > this.weights.length
      ) {
        throw new Error(
          `Некорректное описание тензора ${name}.`,
        );
      }
    }
  }
}