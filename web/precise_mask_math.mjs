export function quantizeMaskValue(value) {
  return Math.round(Math.max(0, Math.min(1, Number(value) || 0)) * 255);
}

export function brushRadius(diameter) {
  return Math.max(0.5, Number(diameter) / 2 || 0.5);
}

export function brushCursorGeometry(clientX, clientY, viewportRect, canvasRect, sourceWidth, diameter) {
  return {
    left: clientX - viewportRect.left,
    top: clientY - viewportRect.top,
    diameter: Math.max(1, Number(diameter) * canvasRect.width / sourceWidth),
  };
}

function paintCircle(mask, width, height, centerX, centerY, radius, value) {
  const cx = Math.round(centerX);
  const cy = Math.round(centerY);
  const r = Math.max(0.5, Number(radius));
  const extent = Math.ceil(r);
  const radiusSquared = r * r;
  for (let y = Math.max(0, cy - extent); y <= Math.min(height - 1, cy + extent); y++) {
    for (let x = Math.max(0, cx - extent); x <= Math.min(width - 1, cx + extent); x++) {
      if ((x - cx) ** 2 + (y - cy) ** 2 <= radiusSquared) mask[y * width + x] = value;
    }
  }
}

function paintSegment(mask, width, height, from, to, radius, value) {
  const distance = Math.hypot(to[0] - from[0], to[1] - from[1]);
  const steps = Math.max(1, Math.ceil(distance / Math.max(1, radius * 0.5)));
  for (let step = 0; step <= steps; step++) {
    const amount = step / steps;
    paintCircle(
      mask,
      width,
      height,
      from[0] + (to[0] - from[0]) * amount,
      from[1] + (to[1] - from[1]) * amount,
      radius,
      value,
    );
  }
}

export function applyAction(mask, width, height, action) {
  if (action.type === "fill") {
    mask.fill(action.value);
    return mask;
  }
  if (action.type !== "stroke" || !action.points?.length) return mask;
  paintCircle(mask, width, height, action.points[0][0], action.points[0][1], action.radius, action.value);
  for (let index = 1; index < action.points.length; index++) {
    paintSegment(mask, width, height, action.points[index - 1], action.points[index], action.radius, action.value);
  }
  return mask;
}

export function replayActions(initialMask, width, height, actions) {
  const mask = initialMask.slice();
  for (const action of actions) applyAction(mask, width, height, action);
  return mask;
}

export function maskFromAlpha(alpha) {
  return Uint8Array.from(alpha, (value) => 255 - value);
}

export function maskToRgba(mask) {
  const rgba = new Uint8ClampedArray(mask.length * 4);
  for (let index = 0; index < mask.length; index++) rgba[index * 4 + 3] = 255 - mask[index];
  return rgba;
}

export function maskPointAt(clientX, clientY, rect, width, height) {
  if (clientX < rect.left || clientY < rect.top || clientX >= rect.left + rect.width || clientY >= rect.top + rect.height) return null;
  return {
    x: Math.min(width - 1, Math.floor((clientX - rect.left) * width / rect.width)),
    y: Math.min(height - 1, Math.floor((clientY - rect.top) * height / rect.height)),
  };
}

export function parseImageReference(value) {
  if (value && typeof value === "object" && typeof value.filename === "string") {
    return { filename: value.filename, subfolder: value.subfolder ?? "", type: value.type ?? "input" };
  }
  const text = String(value ?? "");
  if (text.includes("?filename=")) {
    const url = new URL(text, "http://localhost");
    return {
      filename: url.searchParams.get("filename") ?? "",
      subfolder: url.searchParams.get("subfolder") ?? "",
      type: url.searchParams.get("type") ?? "input",
    };
  }
  const match = text.match(/^(.*?)(?: \[(input|output|temp)\])?$/);
  const path = match?.[1] ?? text;
  const slash = path.lastIndexOf("/");
  return {
    filename: path.slice(slash + 1),
    subfolder: slash < 0 ? "" : path.slice(0, slash),
    type: match?.[2] ?? "input",
  };
}

export function maskFilename(source, timestamp) {
  const { filename } = parseImageReference(source);
  const stem = filename.replace(/\.[^.]*$/, "").replace(/[^a-zA-Z0-9._-]+/g, "-") || "image";
  return `${stem}-precise-mask-${timestamp}.png`;
}

export function widgetImageValue(upload) {
  const path = upload.subfolder ? `${upload.subfolder}/${upload.name}` : upload.name;
  return upload.type === "input" ? `${path} [input]` : `${path} [${upload.type}]`;
}
