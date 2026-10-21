import {
  applyAction,
  brushCursorGeometry,
  brushRadius,
  maskFilename,
  maskFromAlpha,
  maskPointAt,
  maskToRgba,
  parseImageReference,
  quantizeMaskValue,
  replayActions,
  widgetImageValue,
} from "./precise_mask_math.mjs";
import { installPreciseMaskMenu } from "./precise_mask_menu.mjs";

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;
let activeEditor = null;

function make(tag, properties = {}, children = []) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(properties)) {
    if (key === "style") Object.assign(element.style, value);
    else if (key === "className") element.className = value;
    else if (key.startsWith("on")) element.addEventListener(key.slice(2).toLowerCase(), value);
    else element[key] = value;
  }
  element.append(...children);
  return element;
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error(`Unable to load ${url}`));
    image.src = url;
  });
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error("Unable to encode mask PNG.")), "image/png");
  });
}

function sourceReference(node) {
  const widget = node.widgets?.find((item) => item.name === "image");
  const candidates = [widget?.value, node.images?.[0], node.imgs?.[node.imageIndex ?? 0]?.src];
  for (const candidate of candidates) {
    if (!candidate) continue;
    const reference = parseImageReference(candidate);
    if (reference.filename && !reference.filename.startsWith("blake3:")) return reference;
  }
  throw new Error("The selected LoadImage file could not be resolved to a local filename.");
}

function imageUrl(reference, channel) {
  const query = new URLSearchParams({ filename: reference.filename, type: reference.type, channel });
  if (reference.subfolder) query.set("subfolder", reference.subfolder);
  query.set("qwen21_precise_mask", String(Date.now()));
  return api.apiURL(`/view?${query}`);
}

class PreciseMaskEditor {
  constructor(node) {
    this.node = node;
    this.actions = [];
    this.cursor = 0;
    this.tool = "brush";
    this.dirty = false;
    this.spaceDown = false;
    this.panX = 0;
    this.panY = 0;
    this.zoom = 1;
    this.onKeyDown = (event) => this.keyDown(event);
    this.onKeyUp = (event) => { if (event.code === "Space") this.spaceDown = false; };
    this.buildShell();
  }

  buildShell() {
    const buttonStyle = { padding: "5px 9px", cursor: "pointer" };
    this.brushButton = make("button", { textContent: "Brush", style: buttonStyle, onclick: () => this.selectTool("brush") });
    this.eraserButton = make("button", { textContent: "Eraser", style: buttonStyle, onclick: () => this.selectTool("eraser") });
    this.brushColorInput = make("input", {
      type: "color",
      value: "#ff2020",
      title: "Mask overlay and brush cursor color",
      oninput: () => {
        this.selectTool("brush");
        if (this.overlayCanvas) this.renderOverlay();
      },
    });
    this.valueInput = make("input", { type: "number", min: "0", max: "1", step: String(1 / 255), value: "0.5", title: "Absolute mask value" });
    this.sizeInput = make("input", { type: "range", min: "1", max: "500", value: "48", title: "Brush diameter", oninput: () => this.updateBrushCursor() });
    this.undoButton = make("button", { textContent: "Undo", style: buttonStyle, onclick: () => this.undo() });
    this.redoButton = make("button", { textContent: "Redo", style: buttonStyle, onclick: () => this.redo() });
    const presets = [0, 0.25, 0.5, 0.75, 1].map((value) => make("button", {
      textContent: String(value),
      style: buttonStyle,
      onclick: () => { this.valueInput.value = String(value); this.selectTool("brush"); },
    }));
    const toolbar = make("div", { style: { display: "flex", flexWrap: "wrap", gap: "6px", alignItems: "center", padding: "8px", borderBottom: "1px solid #555" } }, [
      this.brushButton,
      this.brushColorInput,
      this.eraserButton,
      make("span", { textContent: "Value" }),
      this.valueInput,
      ...presets,
      make("span", { textContent: "Size" }),
      this.sizeInput,
      make("button", { textContent: "Fill All", style: buttonStyle, onclick: () => this.fill(this.inputValue()) }),
      make("button", { textContent: "Clear", style: buttonStyle, onclick: () => this.fill(0) }),
      this.undoButton,
      this.redoButton,
    ]);
    this.viewport = make("div", { style: { position: "relative", flex: "1", overflow: "hidden", background: "#111" } });
    this.brushCursor = make("div", { style: { position: "absolute", zIndex: "2", display: "none", pointerEvents: "none", border: "1px solid #fff", borderRadius: "50%", boxSizing: "border-box", transform: "translate(-50%, -50%)" } });
    this.viewport.appendChild(this.brushCursor);
    this.status = make("div", { textContent: "Loading image…", style: { flex: "1", font: "12px monospace" } });
    this.saveButton = make("button", { textContent: "Save as New Masked Copy", style: buttonStyle, onclick: () => this.save() });
    const footer = make("div", { style: { display: "flex", gap: "10px", alignItems: "center", padding: "8px", borderTop: "1px solid #555" } }, [
      this.status,
      this.saveButton,
      make("button", { textContent: "Close", style: buttonStyle, onclick: () => this.close() }),
    ]);
    this.dialog = make("div", {
      role: "dialog",
      ariaModal: "true",
      ariaLabel: "Precise Mask Editor",
      style: { width: "min(1400px, 94vw)", height: "92vh", display: "flex", flexDirection: "column", color: "#eee", background: "#282828", border: "1px solid #666", borderRadius: "7px", boxShadow: "0 12px 40px #000" },
    }, [toolbar, this.viewport, footer]);
    this.backdrop = make("div", { style: { position: "fixed", inset: "0", zIndex: "100001", display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,.72)" } }, [this.dialog]);
    document.body.appendChild(this.backdrop);
    window.addEventListener("keydown", this.onKeyDown);
    window.addEventListener("keyup", this.onKeyUp);
    this.updateButtons();
    this.selectTool("brush");
  }

  async open() {
    this.reference = sourceReference(this.node);
    const [rgbImage, alphaImage] = await Promise.all([
      loadImage(imageUrl(this.reference, "rgb")),
      loadImage(imageUrl(this.reference, "a")),
    ]);
    this.width = rgbImage.naturalWidth;
    this.height = rgbImage.naturalHeight;
    if (!this.width || !this.height || alphaImage.naturalWidth !== this.width || alphaImage.naturalHeight !== this.height) {
      throw new Error("Image and alpha dimensions do not match.");
    }
    this.baseCanvas = make("canvas", { width: this.width, height: this.height, style: { position: "absolute", inset: "0" } });
    this.baseCanvas.getContext("2d").drawImage(rgbImage, 0, 0);
    const alphaCanvas = make("canvas", { width: this.width, height: this.height });
    const alphaContext = alphaCanvas.getContext("2d", { willReadFrequently: true });
    alphaContext.drawImage(alphaImage, 0, 0);
    const alphaRgba = alphaContext.getImageData(0, 0, this.width, this.height).data;
    const alpha = new Uint8Array(this.width * this.height);
    for (let index = 0; index < alpha.length; index++) alpha[index] = alphaRgba[index * 4 + 3];
    this.initialMask = maskFromAlpha(alpha);
    this.mask = this.initialMask.slice();
    this.overlayCanvas = make("canvas", { width: this.width, height: this.height, style: { position: "absolute", inset: "0", touchAction: "none", cursor: "none" } });
    this.overlayPixels = this.overlayCanvas.getContext("2d").createImageData(this.width, this.height);
    this.stage = make("div", { style: { position: "absolute", left: "50%", top: "50%", width: `${this.width}px`, height: `${this.height}px`, transformOrigin: "center" } }, [this.baseCanvas, this.overlayCanvas]);
    this.viewport.appendChild(this.stage);
    this.renderOverlay();
    this.bindCanvas();
    requestAnimationFrame(() => {
      this.zoom = Math.min((this.viewport.clientWidth - 24) / this.width, (this.viewport.clientHeight - 24) / this.height, 1);
      this.updateTransform();
    });
    this.status.textContent = `${this.width}×${this.height} · mask values are quantized to 1/255`;
  }

  bindCanvas() {
    this.overlayCanvas.addEventListener("pointerdown", (event) => this.pointerDown(event));
    this.overlayCanvas.addEventListener("pointermove", (event) => this.pointerMove(event));
    this.overlayCanvas.addEventListener("pointerleave", () => { this.brushCursor.style.display = "none"; });
    this.overlayCanvas.addEventListener("pointerup", (event) => this.pointerUp(event));
    this.overlayCanvas.addEventListener("pointercancel", (event) => this.pointerUp(event));
    this.viewport.addEventListener("wheel", (event) => {
      event.preventDefault();
      this.zoom = Math.max(0.02, Math.min(16, this.zoom * (event.deltaY < 0 ? 1.15 : 1 / 1.15)));
      this.updateTransform();
    }, { passive: false });
    this.viewport.addEventListener("pointerdown", (event) => {
      if (event.button !== 1 && !this.spaceDown) return;
      event.preventDefault();
      const startX = event.clientX;
      const startY = event.clientY;
      const panX = this.panX;
      const panY = this.panY;
      const move = (moveEvent) => {
        this.panX = panX + moveEvent.clientX - startX;
        this.panY = panY + moveEvent.clientY - startY;
        this.updateTransform();
      };
      const stop = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", stop);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", stop);
    });
  }

  inputValue() {
    const value = Math.max(0, Math.min(1, Number(this.valueInput.value) || 0));
    this.valueInput.value = String(value);
    return quantizeMaskValue(value);
  }

  selectedValue() {
    return this.tool === "eraser" ? 0 : this.inputValue();
  }

  selectTool(tool) {
    this.tool = tool;
    this.brushButton.style.outline = tool === "brush" ? "2px solid #6af" : "";
    this.eraserButton.style.outline = tool === "eraser" ? "2px solid #6af" : "";
    this.brushCursor.style.borderColor = tool === "brush" ? this.brushColorInput.value : "#ffad42";
    this.updateBrushCursor();
  }

  point(event) {
    return maskPointAt(event.clientX, event.clientY, this.overlayCanvas.getBoundingClientRect(), this.width, this.height);
  }

  pointerDown(event) {
    if (event.button !== 0 || this.spaceDown) return;
    this.lastPointer = { clientX: event.clientX, clientY: event.clientY };
    this.updateBrushCursor();
    const point = this.point(event);
    if (!point) return;
    event.preventDefault();
    this.overlayCanvas.setPointerCapture(event.pointerId);
    this.currentAction = {
      type: "stroke",
      radius: brushRadius(this.sizeInput.value),
      value: this.selectedValue(),
      points: [[point.x, point.y]],
    };
    applyAction(this.mask, this.width, this.height, this.currentAction);
    this.renderStrokeRegion(point, point, this.currentAction.radius);
  }

  pointerMove(event) {
    this.lastPointer = { clientX: event.clientX, clientY: event.clientY };
    this.updateBrushCursor();
    const point = this.point(event);
    if (!point) return;
    const value = this.mask[point.y * this.width + point.x];
    this.status.textContent = `pixel (${point.x}, ${point.y}) · mask ${value}/255 = ${(value / 255).toFixed(4)}`;
    if (!this.currentAction) return;
    const previous = this.currentAction.points[this.currentAction.points.length - 1];
    if (previous[0] === point.x && previous[1] === point.y) return;
    this.currentAction.points.push([point.x, point.y]);
    applyAction(this.mask, this.width, this.height, {
      ...this.currentAction,
      points: [previous, [point.x, point.y]],
    });
    this.renderStrokeRegion({ x: previous[0], y: previous[1] }, point, this.currentAction.radius);
  }

  pointerUp(event) {
    if (!this.currentAction) return;
    if (this.overlayCanvas.hasPointerCapture(event.pointerId)) this.overlayCanvas.releasePointerCapture(event.pointerId);
    this.commitAction(this.currentAction, false);
    this.currentAction = null;
  }

  commitAction(action, apply = true) {
    this.actions.splice(this.cursor);
    this.actions.push(action);
    this.cursor = this.actions.length;
    if (apply) applyAction(this.mask, this.width, this.height, action);
    this.dirty = true;
    this.updateButtons();
  }

  fill(value) {
    this.commitAction({ type: "fill", value });
    this.renderOverlay();
  }

  undo() {
    if (this.cursor === 0) return;
    this.cursor--;
    this.replay();
  }

  redo() {
    if (this.cursor === this.actions.length) return;
    this.cursor++;
    this.replay();
  }

  replay() {
    this.mask = replayActions(this.initialMask, this.width, this.height, this.actions.slice(0, this.cursor));
    this.dirty = true;
    this.renderOverlay();
    this.updateButtons();
  }

  updateButtons() {
    this.undoButton.disabled = this.cursor === 0;
    this.redoButton.disabled = this.cursor === this.actions.length;
  }

  renderStrokeRegion(from, to, radius) {
    const padding = Math.ceil(radius) + 2;
    this.renderOverlay({
      x: Math.max(0, Math.floor(Math.min(from.x, to.x) - padding)),
      y: Math.max(0, Math.floor(Math.min(from.y, to.y) - padding)),
      width: Math.min(this.width, Math.ceil(Math.max(from.x, to.x) + padding)) - Math.max(0, Math.floor(Math.min(from.x, to.x) - padding)),
      height: Math.min(this.height, Math.ceil(Math.max(from.y, to.y) + padding)) - Math.max(0, Math.floor(Math.min(from.y, to.y) - padding)),
    });
  }

  renderOverlay(region = { x: 0, y: 0, width: this.width, height: this.height }) {
    const data = this.overlayPixels.data;
    const color = Number.parseInt(this.brushColorInput.value.slice(1), 16);
    const [red, green, blue] = [color >> 16, (color >> 8) & 255, color & 255];
    for (let y = region.y; y < region.y + region.height; y++) {
      for (let x = region.x; x < region.x + region.width; x++) {
        const pixel = y * this.width + x;
        const rgba = pixel * 4;
        data[rgba] = red;
        data[rgba + 1] = green;
        data[rgba + 2] = blue;
        data[rgba + 3] = Math.round(this.mask[pixel] * 0.65);
      }
    }
    this.overlayCanvas.getContext("2d").putImageData(this.overlayPixels, 0, 0, region.x, region.y, region.width, region.height);
  }

  updateTransform() {
    this.stage.style.transform = `translate(-50%, -50%) translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
    this.updateBrushCursor();
  }

  updateBrushCursor() {
    if (!this.lastPointer || !this.overlayCanvas || !this.width) return;
    const canvasRect = this.overlayCanvas.getBoundingClientRect();
    if (!maskPointAt(this.lastPointer.clientX, this.lastPointer.clientY, canvasRect, this.width, this.height)) {
      this.brushCursor.style.display = "none";
      return;
    }
    const geometry = brushCursorGeometry(
      this.lastPointer.clientX,
      this.lastPointer.clientY,
      this.viewport.getBoundingClientRect(),
      canvasRect,
      this.width,
      this.sizeInput.value,
    );
    Object.assign(this.brushCursor.style, {
      display: "block",
      left: `${geometry.left}px`,
      top: `${geometry.top}px`,
      width: `${geometry.diameter}px`,
      height: `${geometry.diameter}px`,
    });
  }

  keyDown(event) {
    if (event.code === "Space") {
      this.spaceDown = true;
      return;
    }
    if (event.key === "Escape") this.close();
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
      event.preventDefault();
      event.shiftKey ? this.redo() : this.undo();
    }
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
      event.preventDefault();
      this.redo();
    }
  }

  async save() {
    this.saveButton.disabled = true;
    this.status.textContent = "Saving mask copy…";
    try {
      const canvas = make("canvas", { width: this.width, height: this.height });
      const context = canvas.getContext("2d");
      const imageData = context.createImageData(this.width, this.height);
      imageData.data.set(maskToRgba(this.mask));
      context.putImageData(imageData, 0, 0);
      const filename = maskFilename(this.reference, Date.now());
      const form = new FormData();
      form.append("image", await canvasBlob(canvas), filename);
      form.append("type", "input");
      form.append("original_ref", JSON.stringify(this.reference));
      const response = await api.fetchApi("/upload/mask", { method: "POST", body: form });
      if (!response.ok) throw new Error(`Mask upload failed (${response.status}).`);
      const upload = await response.json();
      const widget = this.node.widgets?.find((item) => item.name === "image");
      if (!widget) throw new Error("LoadImage image widget is unavailable.");
      const value = widgetImageValue(upload);
      widget.value = value;
      widget.callback?.(value);
      this.node.setDirtyCanvas?.(true, true);
      this.node.graph?.change?.();
      this.dirty = false;
      this.close(true);
    } catch (error) {
      this.status.textContent = error instanceof Error ? error.message : String(error);
      this.saveButton.disabled = false;
    }
  }

  close(force = false) {
    if (!force && this.dirty && !window.confirm("Discard unsaved mask changes?")) return;
    window.removeEventListener("keydown", this.onKeyDown);
    window.removeEventListener("keyup", this.onKeyUp);
    this.backdrop.remove();
    if (activeEditor === this) activeEditor = null;
  }
}

async function openPreciseMaskEditor(node) {
  if (activeEditor) activeEditor.close(true);
  const editor = new PreciseMaskEditor(node);
  activeEditor = editor;
  try {
    await editor.open();
  } catch (error) {
    editor.status.textContent = error instanceof Error ? error.message : String(error);
    editor.saveButton.disabled = true;
  }
}

app.registerExtension({
  name: "Qwen21Enhancer.PreciseMaskEditor",
  beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === "LoadImage") installPreciseMaskMenu(nodeType, openPreciseMaskEditor);
  },
});
