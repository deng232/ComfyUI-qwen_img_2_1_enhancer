export function decodeFloat32(encoded) {
  const binary = atob(encoded);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}

export function imagePixelAt(x, y, rect, naturalWidth, naturalHeight) {
  if (x < rect.x || y < rect.y || x >= rect.x + rect.width || y >= rect.y + rect.height) return null;
  return {
    x: Math.min(naturalWidth - 1, Math.floor((x - rect.x) * naturalWidth / rect.width)),
    y: Math.min(naturalHeight - 1, Math.floor((y - rect.y) * naturalHeight / rect.height)),
  };
}

function clamped(values, width, height, x, y) {
  const px = Math.max(0, Math.min(width - 1, x));
  const py = Math.max(0, Math.min(height - 1, y));
  return values[py * width + px];
}

function bilinear(values, width, height, x, y) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const dx = x - x0;
  const dy = y - y0;
  const top = clamped(values, width, height, x0, y0) * (1 - dx)
    + clamped(values, width, height, x0 + 1, y0) * dx;
  const bottom = clamped(values, width, height, x0, y0 + 1) * (1 - dx)
    + clamped(values, width, height, x0 + 1, y0 + 1) * dx;
  return top * (1 - dy) + bottom * dy;
}

export function displayValue(raw, capture, mode) {
  if (mode === "percentile_99") return Math.min(1, raw / capture.visual_ceiling);
  if (mode === "log") return Math.log1p(1000 * raw / capture.raw_max) / Math.log(1001);
  return Math.min(1, raw / capture.raw_max);
}

export function sampleAttention(capture, mode, pixelX, pixelY, imageWidth, imageHeight) {
  const width = capture.grid_width;
  const height = capture.grid_height;
  const tokenX = Math.max(0, Math.min(width - 1, Math.floor((pixelX + 0.5) * width / imageWidth)));
  const tokenY = Math.max(0, Math.min(height - 1, Math.floor((pixelY + 0.5) * height / imageHeight)));
  const sourceX = (pixelX + 0.5) * width / imageWidth - 0.5;
  const sourceY = (pixelY + 0.5) * height / imageHeight - 0.5;
  const displayGrid = capture.displayValues ?? Float32Array.from(
    capture.values,
    (value) => displayValue(value, capture, mode),
  );
  return {
    tokenX,
    tokenY,
    tokenValue: capture.values[tokenY * width + tokenX],
    interpolatedRaw: bilinear(capture.values, width, height, sourceX, sourceY),
    displayIntensity: bilinear(displayGrid, width, height, sourceX, sourceY),
  };
}
