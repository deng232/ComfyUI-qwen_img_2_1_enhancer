import base64
import json
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from comfy_api.latest import io, ui

from .probe import ProbeConfig, ProbeHandle, ProbeInputError, install_probe, validate_model


ProbeHandleIO = io.Custom("QWEN21_PROBE_HANDLE")
AttentionDataIO = io.Custom("QWEN21_ATTENTION_DATA")


@dataclass(frozen=True)
class AttentionData:
    results: tuple
    requested_ts: tuple
    metadata: dict


def _parse_capture_ts(value):
    parts = [part.strip() for part in str(value).split(",")]
    if not parts or any(not part for part in parts):
        raise ProbeInputError("capture_t must be a comma-separated list such as 0,0.25,0.5,0.75,1.")
    try:
        values = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ProbeInputError("capture_t contains a value that is not a number.") from error
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in values):
        raise ProbeInputError("Every capture_t value must be finite and between 0 and 1.")
    if len(set(values)) != len(values):
        raise ProbeInputError("capture_t contains duplicate values.")
    if len(values) > 32:
        raise ProbeInputError("capture_t supports at most 32 values per sampling run.")
    return values


def _normalize_for_display(heatmap, mode):
    maximum = float(heatmap.max())
    if maximum <= 0.0:
        raise ProbeInputError("Captured reference attention is zero; refusing to render a false heatmap.")
    if mode == "linear":
        return (heatmap / maximum).clamp(0.0, 1.0), maximum
    if mode == "percentile_99":
        ceiling = float(torch.quantile(heatmap.flatten(), 0.99))
        ceiling = ceiling if ceiling > 0.0 else maximum
        return (heatmap / ceiling).clamp(0.0, 1.0), ceiling
    scaled = heatmap / maximum
    return (torch.log1p(1000.0 * scaled) / math.log(1001.0)).clamp(0.0, 1.0), maximum


def _colorize(normalized):
    red = torch.minimum(2.0 * normalized, torch.ones_like(normalized))
    green = 1.0 - (2.0 * normalized - 1.0).abs()
    blue = (1.0 - 2.0 * normalized).clamp(0.0, 1.0)
    return torch.stack((red, green.clamp(0.0, 1.0), blue), dim=-1)


def _render(result, reference_image, overlay_alpha, visual_scale):
    if reference_image.ndim != 4 or reference_image.shape[0] < 1 or reference_image.shape[-1] < 3:
        raise ProbeInputError("reference_image must contain at least one RGB image in ComfyUI IMAGE format.")
    reference = reference_image[:1, :, :, :3].detach().float().cpu().clamp(0.0, 1.0)
    height, width = (int(value) for value in reference.shape[1:3])
    normalized, display_ceiling = _normalize_for_display(result.heatmap, visual_scale)
    resized = F.interpolate(normalized[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    colors = _colorize(resized)[None]
    overlay = reference * (1.0 - overlay_alpha) + colors * overlay_alpha
    return overlay.clamp(0.0, 1.0), display_ceiling


def _metadata(results, config):
    captures = []
    for requested_t, result in zip(config.capture_ts, results):
        warnings = []
        if result.attention_mass < 1e-4:
            warnings.append("Reference attention mass is below 1e-4; relative overlay colors may be visually misleading.")
        if result.sampling_mode == "uniform_1d" and result.query_count < result.total_target_queries:
            warnings.append(
                "Queries were uniformly subsampled from the flattened target grid; RoPE position affinity can turn "
                "that regular sampling lattice into periodic spatial peaks."
            )
        captures.append({
            "requested_t": requested_t,
            "actual_t": result.actual_t,
            "reference_attention_mass": result.attention_mass,
            "raw_min": float(result.heatmap.min()),
            "raw_max": float(result.heatmap.max()),
            "attention_mask_applied": result.attention_mask_applied,
            "probe_elapsed_ms": result.probe_elapsed_ms,
            "probe_observed_peak_allocated_delta_bytes": result.probe_observed_peak_allocated_delta_bytes,
            "warnings": warnings,
        })
    first = results[0]
    return {
        "status": "complete",
        "capture_count": len(results),
        "captures": captures,
        "block_index": first.block_index,
        "head_index": first.head_index,
        "reference_index": first.reference_index,
        "condition_branch": config.condition_branch,
        "batch_index": config.batch_index,
        "query_count": first.query_count,
        "total_target_queries": first.total_target_queries,
        "sampling_mode": first.sampling_mode,
        "query_chunk_size": first.query_chunk_size,
        "sampled_positions_yx": first.sampled_positions if first.sampling_mode == "uniform_1d" else None,
        "reference_hw": first.reference_hw,
        "target_hw": first.target_hw,
        "target_h": first.target_hw[0],
        "target_w": first.target_hw[1],
        "reference_span": first.reference_span,
        "target_span": first.target_span,
        "total_tokens": first.total_tokens,
        "attention_basis": "effective_softmax_after_attention_mask",
        "probe_measurement_scope": "attention_recomputation_only",
        "probe_memory_measurement": "device_allocator_delta_sampled_at_probe_tensor_boundaries",
        "ref_attention_mass_output": "first requested capture",
    }


def _float32_base64(tensor):
    raw = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


class Qwen21ReferenceAttentionProbe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen21ReferenceAttentionProbe",
            display_name="Qwen21 Reference Attention Probe",
            category="advanced/debug/qwen image",
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Boolean.Input("enabled", default=True),
                io.Int.Input("reference_index", default=0, min=0, max=63),
                io.Int.Input("block_index", default=16, min=0, max=4095),
                io.Int.Input("head_index", default=0, min=0, max=4095),
                io.String.Input("capture_t", default="0.5"),
                io.Int.Input("max_queries", default=128, min=1),
                io.Int.Input("batch_index", default=0, min=0, max=4095),
                io.Combo.Input("condition_branch", options=["positive", "negative"], default="positive"),
                io.Combo.Input("sampling_mode", options=["uniform_1d", "full_chunked"], default="uniform_1d"),
                io.Combo.Input("query_chunk_size", options=[16, 32, 64, 128], default=32),
            ],
            outputs=[
                io.Model.Output(display_name="patched_model"),
                ProbeHandleIO.Output(display_name="probe_handle"),
            ],
        )

    @classmethod
    def execute(cls, model, enabled, reference_index, block_index, head_index, capture_t, max_queries, batch_index, condition_branch, sampling_mode, query_chunk_size):
        if not enabled:
            return io.NodeOutput(model, ProbeHandle(disabled=True))
        validate_model(model, block_index, head_index)
        capture_ts = _parse_capture_ts(capture_t)
        config = ProbeConfig(
            reference_index=reference_index,
            block_index=block_index,
            head_index=head_index,
            capture_ts=capture_ts,
            max_queries=max_queries,
            batch_index=batch_index,
            condition_branch=condition_branch,
            sampling_mode=sampling_mode,
            query_chunk_size=query_chunk_size,
        )
        handle = ProbeHandle(config=config)
        return io.NodeOutput(install_probe(model, handle), handle)


class Qwen21ReferenceAttentionProbing(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen21ReferenceAttentionProbing",
            display_name="Qwen21 Reference Attention Probing",
            category="advanced/debug/qwen image",
            is_experimental=True,
            inputs=[
                ProbeHandleIO.Input("probe_handle"),
                io.Latent.Input("samples"),
            ],
            outputs=[
                AttentionDataIO.Output(display_name="attention_data"),
                io.Float.Output(display_name="ref_attention_mass"),
                io.String.Output(display_name="metadata"),
            ],
        )

    @classmethod
    def execute(cls, probe_handle, samples):
        results = probe_handle.result_for(samples)
        metadata = _metadata(results, probe_handle.config)
        data = AttentionData(tuple(results), tuple(probe_handle.config.capture_ts), metadata)
        return io.NodeOutput(data, results[0].attention_mass, json.dumps(metadata, indent=2))


class Qwen21ReferenceAttentionPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen21ReferenceAttentionPreview",
            display_name="Qwen21 Reference Attention Preview",
            category="advanced/debug/qwen image",
            is_experimental=True,
            is_output_node=True,
            inputs=[
                AttentionDataIO.Input("attention_data"),
                io.Image.Input("reference_image"),
                io.Float.Input("overlay_alpha", default=0.55, min=0.0, max=1.0, step=0.01),
                io.Combo.Input("visual_scale", options=["linear", "percentile_99", "log"], default="percentile_99"),
            ],
            outputs=[io.Image.Output(display_name="overlay")],
        )

    @classmethod
    def execute(cls, attention_data, reference_image, overlay_alpha, visual_scale):
        image_height, image_width = (int(value) for value in reference_image.shape[1:3])
        overlays = []
        hover_captures = []
        for requested_t, result in zip(attention_data.requested_ts, attention_data.results):
            overlay, display_ceiling = _render(result, reference_image, overlay_alpha, visual_scale)
            overlays.append(overlay)
            grid_height, grid_width = result.reference_hw
            hover_captures.append({
                "requested_t": requested_t,
                "actual_t": result.actual_t,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "raw_min": float(result.heatmap.min()),
                "raw_max": float(result.heatmap.max()),
                "visual_ceiling": display_ceiling,
                "raw_f32_base64": _float32_base64(result.heatmap),
            })
        overlay_batch = torch.cat(overlays, dim=0)
        preview = ui.PreviewImage(overlay_batch, cls=cls).as_dict()
        preview["qwen21_attention_hover"] = {
            "version": 1,
            "visual_scale": visual_scale,
            "reference_width": image_width,
            "reference_height": image_height,
            "captures": hover_captures,
        }
        return io.NodeOutput(overlay_batch, ui=preview)


class Qwen21ReferenceMaskPreview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Qwen21ReferenceMaskPreview",
            display_name="Qwen21 Reference Mask Preview",
            category="advanced/debug/qwen image",
            is_output_node=True,
            inputs=[io.Mask.Input("mask")],
            outputs=[io.Mask.Output(display_name="mask")],
        )

    @classmethod
    def execute(cls, mask):
        return io.NodeOutput(mask, ui=ui.PreviewMask(mask, cls=cls))
