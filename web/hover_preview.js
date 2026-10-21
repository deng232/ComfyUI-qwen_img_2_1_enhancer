import { decodeFloat32, displayValue, imagePixelAt, sampleAttention } from "./hover_math.mjs";

const { app } = window.comfyAPI.app;
const ATTENTION = "Qwen21ReferenceAttentionPreview";
const MASK = "Qwen21ReferenceMaskPreview";
const imageDataCache = new WeakMap();

const tooltip = document.createElement("div");
Object.assign(tooltip.style, {
  position: "fixed",
  display: "none",
  zIndex: "100000",
  pointerEvents: "none",
  padding: "6px 8px",
  border: "1px solid #777",
  borderRadius: "5px",
  background: "rgba(18, 18, 18, 0.94)",
  color: "#fff",
  font: "12px/1.45 monospace",
  whiteSpace: "pre",
});
document.body.appendChild(tooltip);

function previewWidget(node) {
  return node.widgets?.find((widget) =>
    Number.isFinite(widget.y) && Number.isFinite(widget.computedHeight) && widget.computedHeight > 40
  );
}

function imageRect(node, index) {
  const img = node.imgs?.[index];
  const widget = previewWidget(node);
  if (!img || !widget) return null;
  if (node.imageIndex == null && node.imageRects?.[index]) {
    const [cellX, cellY, cellWidth, cellHeight] = node.imageRects[index];
    const scale = Math.min(cellWidth / img.naturalWidth, cellHeight / img.naturalHeight);
    const width = img.naturalWidth * scale;
    const height = img.naturalHeight * scale;
    return { x: cellX + (cellWidth - width) / 2, y: cellY + (cellHeight - height) / 2, width, height };
  }
  const availableWidth = node.size[0];
  const showImageSize = app.ui?.settings?.getSettingValue?.("Comfy.Node.AllowImageSizeDraw") ?? false;
  const availableHeight = widget.computedHeight - (showImageSize ? 15 : 0);
  const scale = Math.min(availableWidth / img.naturalWidth, availableHeight / img.naturalHeight, 1);
  const width = img.naturalWidth * scale;
  const height = img.naturalHeight * scale;
  return {
    x: (availableWidth - width) / 2,
    y: widget.y + (availableHeight - height) / 2,
    width,
    height,
  };
}

function hoveredPreview(event) {
  const canvas = app.canvas;
  const point = canvas.convertEventToCanvasOffset(event);
  const nodes = canvas.getCurrentGraph?.()?._nodes ?? app.graph?._nodes ?? [];
  for (let i = nodes.length - 1; i >= 0; i--) {
    const node = nodes[i];
    if (node.comfyClass !== ATTENTION && node.comfyClass !== MASK && node.type !== ATTENTION && node.type !== MASK) continue;
    const localX = point[0] - node.pos[0];
    const localY = point[1] - node.pos[1];
    if (localX < 0 || localY < 0 || localX >= node.size[0] || localY >= node.size[1]) continue;
    const index = node.imageIndex ?? node.overIndex ?? 0;
    const img = node.imgs?.[index];
    const rect = imageRect(node, index);
    if (!img || !rect) continue;
    const pixel = imagePixelAt(localX, localY, rect, img.naturalWidth, img.naturalHeight);
    if (pixel) return { node, index, img, pixel };
  }
  return null;
}

function maskPixels(img) {
  let data = imageDataCache.get(img);
  if (data) return data;
  const canvas = document.createElement("canvas");
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(img, 0, 0);
  data = context.getImageData(0, 0, canvas.width, canvas.height);
  imageDataCache.set(img, data);
  return data;
}

function show(event, text) {
  tooltip.textContent = text;
  tooltip.style.left = `${event.clientX + 14}px`;
  tooltip.style.top = `${event.clientY + 14}px`;
  tooltip.style.display = "block";
}

function onMove(event) {
  const hit = hoveredPreview(event);
  if (!hit) {
    tooltip.style.display = "none";
    return;
  }
  if (hit.node.comfyClass === MASK || hit.node.type === MASK) {
    try {
      const pixels = maskPixels(hit.img);
      const value = pixels.data[(hit.pixel.y * pixels.width + hit.pixel.x) * 4];
      show(event, `batch ${hit.index}\npixel (${hit.pixel.x}, ${hit.pixel.y})\nopacity ${(value / 255).toFixed(4)}  ${value}/255  ${(value / 2.55).toFixed(1)}%`);
    } catch {
      tooltip.style.display = "none";
    }
    return;
  }
  const payload = hit.node.qwen21AttentionHover;
  const capture = payload?.captures?.[hit.index];
  if (!capture) {
    tooltip.style.display = "none";
    return;
  }
  const sample = sampleAttention(
    capture,
    payload.visual_scale,
    hit.pixel.x,
    hit.pixel.y,
    payload.reference_width,
    payload.reference_height,
  );
  show(event,
    `capture ${hit.index}  t ${capture.requested_t} → ${capture.actual_t}\n`
    + `pixel (${hit.pixel.x}, ${hit.pixel.y})  token (${sample.tokenX}, ${sample.tokenY})\n`
    + `token attention ${sample.tokenValue.toExponential(6)}\n`
    + `bilinear attention ${sample.interpolatedRaw.toExponential(6)}\n`
    + `display intensity ${sample.displayIntensity.toFixed(6)}`
  );
}

app.registerExtension({
  name: "Qwen21Enhancer.HoverPreview",
  setup() {
    app.canvas.canvas.addEventListener("pointermove", onMove);
    app.canvas.canvas.addEventListener("pointerleave", () => { tooltip.style.display = "none"; });
  },
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== ATTENTION) return;
    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      const payload = Array.isArray(message?.qwen21_attention_hover)
        ? message.qwen21_attention_hover[0]
        : message?.qwen21_attention_hover;
      if (!payload) return;
      for (const capture of payload.captures ?? []) {
        capture.values = decodeFloat32(capture.raw_f32_base64);
        capture.displayValues = Float32Array.from(
          capture.values,
          (value) => displayValue(value, capture, payload.visual_scale),
        );
        delete capture.raw_f32_base64;
      }
      this.qwen21AttentionHover = payload;
    };
  },
});
