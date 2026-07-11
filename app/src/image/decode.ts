const BMP_MIME_TYPES = [
  'image/bmp',
  'image/x-ms-bmp',
];

const HEIC_MIME_TYPES = [
  'image/heic',
  'image/heif',
  'image/heic-sequence',
  'image/heif-sequence',
];

const SUPPORTED_MIME_TYPES = [
  'image/jpeg',
  'image/png',
  ...BMP_MIME_TYPES,
  ...HEIC_MIME_TYPES,
];

export function isBmpFile(file: File): boolean {
  const mimeType = file.type.toLowerCase();

  return (
    BMP_MIME_TYPES.includes(mimeType) ||
    /\.bmp$/i.test(file.name)
  );
}

export function isHeicFile(file: File): boolean {
  const mimeType = file.type.toLowerCase();

  return (
    HEIC_MIME_TYPES.includes(mimeType) ||
    /\.(heic|heif)$/i.test(file.name)
  );
}

export function isPngFile(file: File): boolean {
  return (
    file.type.toLowerCase() === 'image/png' ||
    /\.png$/i.test(file.name)
  );
}

export function isSupportedImageFile(
  file: File,
): boolean {
  return (
    SUPPORTED_MIME_TYPES.includes(
      file.type.toLowerCase(),
    ) ||
    /\.(jpe?g|png|bmp|heic|heif)$/i.test(
      file.name,
    )
  );
}

async function decodeBmp(
  file: File,
  maxPixels: number,
): Promise<ImageBitmap> {
  const buffer = await file.arrayBuffer();
  const view = new DataView(buffer);

  if (view.byteLength < 54) {
    throw new Error(
      'BMP-файл повреждён или слишком мал.',
    );
  }

  if (view.getUint16(0, true) !== 0x4d42) {
    throw new Error(
      'Некорректная сигнатура BMP-файла.',
    );
  }

  const pixelOffset =
    view.getUint32(10, true);

  const dibHeaderSize =
    view.getUint32(14, true);

  if (dibHeaderSize < 40) {
    throw new Error(
      'Данный вариант заголовка BMP не поддерживается.',
    );
  }

  const width =
    view.getInt32(18, true);

  const rawHeight =
    view.getInt32(22, true);

  const planes =
    view.getUint16(26, true);

  const bitsPerPixel =
    view.getUint16(28, true);

  const compression =
    view.getUint32(30, true);

  if (width <= 0 || rawHeight === 0) {
    throw new Error(
      'Некорректные размеры BMP-изображения.',
    );
  }

  if (planes !== 1) {
    throw new Error(
      'Некорректное количество цветовых плоскостей BMP.',
    );
  }

  if (
    bitsPerPixel !== 24 &&
    bitsPerPixel !== 32
  ) {
    throw new Error(
      'Поддерживаются только 24-битные и 32-битные BMP.',
    );
  }

  if (compression !== 0) {
    throw new Error(
      'Сжатые BMP-файлы не поддерживаются.',
    );
  }

  const height = Math.abs(rawHeight);
  const topDown = rawHeight < 0;
  const totalPixels = width * height;

  if (totalPixels > maxPixels) {
    throw new Error(
      `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
        `Максимально допустимо ${maxPixels.toLocaleString()} пикселей.`,
    );
  }

  const rowSize =
    Math.floor(
      (bitsPerPixel * width + 31) / 32,
    ) * 4;

  const requiredBytes =
    pixelOffset + rowSize * height;

  if (requiredBytes > view.byteLength) {
    throw new Error(
      'BMP-файл содержит неполные данные пикселей.',
    );
  }

  const output =
    new Uint8ClampedArray(
      totalPixels * 4,
    );

  const bytesPerPixel =
    bitsPerPixel / 8;

  for (
    let targetY = 0;
    targetY < height;
    targetY += 1
  ) {
    const sourceY = topDown
      ? targetY
      : height - 1 - targetY;

    const sourceRowOffset =
      pixelOffset + sourceY * rowSize;

    for (
      let x = 0;
      x < width;
      x += 1
    ) {
      const sourceIndex =
        sourceRowOffset +
        x * bytesPerPixel;

      const targetIndex =
        (targetY * width + x) * 4;

      output[targetIndex] =
        view.getUint8(sourceIndex + 2);

      output[targetIndex + 1] =
        view.getUint8(sourceIndex + 1);

      output[targetIndex + 2] =
        view.getUint8(sourceIndex);

      output[targetIndex + 3] = 255;
    }
  }

  const canvas =
    new OffscreenCanvas(width, height);

  const context =
    canvas.getContext('2d');

  if (!context) {
    throw new Error(
      'Не удалось создать контекст декодирования BMP.',
    );
  }

  const imageData =
    context.createImageData(
      width,
      height,
    );

  imageData.data.set(output);

  context.putImageData(
    imageData,
    0,
    0,
  );

  return canvas.transferToImageBitmap();
}

async function decodeHeic(
  file: File,
  maxPixels: number,
): Promise<ImageBitmap> {
  const { heicTo } =
    await import('heic-to/next');

  const result = await heicTo({
    blob: file,
    type: 'bitmap',
    options: {
      imageOrientation: 'flipY',
    },
  });

  const bitmap = result as ImageBitmap;

  if (
    !bitmap ||
    typeof bitmap.width !== 'number' ||
    typeof bitmap.height !== 'number'
  ) {
    throw new Error(
      'Не удалось декодировать HEIC-изображение.',
    );
  }

  const totalPixels =
    bitmap.width * bitmap.height;

  if (totalPixels > maxPixels) {
    bitmap.close();

    throw new Error(
      `Изображение содержит ${totalPixels.toLocaleString()} пикселей. ` +
        `Максимально допустимо ${maxPixels.toLocaleString()} пикселей.`,
    );
  }

  return bitmap;
}

export async function decodeImage(
  file: File,
  maxPixels: number,
): Promise<ImageBitmap> {
  if (!isSupportedImageFile(file)) {
    throw new Error(
      'Поддерживаются изображения JPG, PNG, BMP, HEIC и HEIF.',
    );
  }

  if (isBmpFile(file)) {
    return decodeBmp(file, maxPixels);
  }

  if (isHeicFile(file)) {
    return decodeHeic(file, maxPixels);
  }

  return createImageBitmap(file);
}