import hashlib
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

import comfy.model_management
import comfy.patcher_extension
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel


WRAPPER_KEY = "qwen21_attention_probe"


class ProbeInputError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenSpan:
    start: int
    end: int
    height: int
    width: int


@dataclass(frozen=True)
class SequenceLayout:
    references: tuple[TokenSpan, ...]
    target: TokenSpan
    total_tokens: int


@dataclass(frozen=True)
class ForwardContext:
    layout: SequenceLayout
    batch_row: int
    actual_t: float
    run: object


@dataclass(frozen=True)
class ProbeConfig:
    reference_index: int
    block_index: int
    head_index: int
    capture_ts: tuple[float, ...]
    max_queries: int
    batch_index: int
    condition_branch: str
    sampling_mode: str
    query_chunk_size: int


@dataclass(frozen=True)
class ProbeResult:
    heatmap: torch.Tensor
    attention_mass: float
    actual_t: float
    block_index: int
    head_index: int
    reference_index: int
    query_count: int
    total_target_queries: int
    sampling_mode: str
    query_chunk_size: int
    sampled_positions: tuple[tuple[int, int], ...]
    reference_span: tuple[int, int]
    target_span: tuple[int, int]
    reference_hw: tuple[int, int]
    target_hw: tuple[int, int]
    total_tokens: int
    attention_mask_applied: bool
    probe_elapsed_ms: float
    probe_observed_peak_allocated_delta_bytes: int | None


@dataclass
class RunState:
    status: str = "running"
    error: str | None = None
    results: dict[float, ProbeResult] | None = None
    best_distances: dict[float, float] | None = None


def _sample_fingerprint(samples):
    if not isinstance(samples, torch.Tensor):
        raise ProbeInputError("The captured sampler output is not a Tensor.")
    tensor = samples.detach().to(
        device="cpu",
        dtype=comfy.model_management.intermediate_dtype(),
    ).contiguous()
    digest = hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()
    return tuple(tensor.shape), str(tensor.dtype), digest


def build_sequence_layout(x, context, ref_latents, image_slots):
    references = list(ref_latents or [])
    if not references:
        raise ProbeInputError("This Qwen-Image-2.1 conditioning has no VAE reference_latents.")

    context_length = int(context.shape[1])
    slots = (list(image_slots or []) + [context_length] * len(references))[:len(references)]
    if any(slot < 0 or slot > context_length for slot in slots) or slots != sorted(slots):
        raise AssertionError(f"Invalid Qwen-Image-2.1 image slots: {slots}")

    bounds = [0] + slots + [context_length]
    cursor = 0
    spans = []
    target = None
    images = references + [x]
    for index, image in enumerate(images):
        text_start, text_end = bounds[index], bounds[index + 1]
        cursor += text_end - text_start
        height, width = (int(v) for v in image.shape[-2:])
        span = TokenSpan(cursor, cursor + height * width, height, width)
        if index < len(references):
            spans.append(span)
        else:
            target = span
        cursor = span.end

    assert target is not None
    assert all(span.end - span.start == span.height * span.width for span in spans)
    assert all(0 <= span.start < span.end <= cursor for span in spans)
    assert all(left.end <= right.start for left, right in zip(spans, spans[1:]))
    assert target.end == cursor
    return SequenceLayout(tuple(spans), target, cursor)


def _resolve_batch_row(transformer_options, batch_size, batch_index, condition_branch):
    groups = list(transformer_options.get("cond_or_uncond") or [0])
    requested = 0 if condition_branch == "positive" else 1
    matches = [index for index, value in enumerate(groups) if value == requested]
    if len(matches) != 1:
        raise ProbeInputError(
            f"Requested {condition_branch} branch is unavailable or ambiguous; cond_or_uncond={groups}."
        )
    if batch_size % len(groups):
        raise ProbeInputError(f"Attention batch {batch_size} cannot be divided into CFG groups {groups}.")
    logical_batch = batch_size // len(groups)
    if batch_index >= logical_batch:
        raise ProbeInputError(
            f"batch_index {batch_index} is out of range for branch batch size {logical_batch}."
        )
    return matches[0] * logical_batch + batch_index, logical_batch


def _actual_t(timestep, x, batch_row, logical_batch, batch_index):
    rounded = ((timestep * 1000).to(x.dtype) / 1000).to(x.dtype).detach().flatten()
    if rounded.numel() == 1:
        value = rounded[0]
    elif rounded.numel() == x.shape[0]:
        value = rounded[batch_row]
    elif rounded.numel() == logical_batch:
        value = rounded[batch_index]
    else:
        raise ProbeInputError(
            f"Cannot map timestep shape {tuple(timestep.shape)} to attention batch {x.shape[0]}."
        )
    return float(value.float().cpu())


def _uniform_subset(indices, limit):
    if indices.numel() <= limit:
        return indices
    offsets = torch.linspace(0, indices.numel() - 1, limit).round().long()
    return indices[offsets].unique()


def select_target_queries(config, target):
    count = target.height * target.width
    local = torch.arange(count, dtype=torch.long)
    if config.sampling_mode == "full_chunked":
        return local
    if config.sampling_mode == "uniform_1d":
        return _uniform_subset(local, config.max_queries)
    raise ProbeInputError(f"Unknown sampling_mode: {config.sampling_mode}.")


class ProbeHandle:
    def __init__(self, config=None, disabled=False):
        self.config = config
        self.disabled = disabled
        self._lock = threading.Lock()
        self._local = threading.local()
        self._completed = OrderedDict()

    def start_run(self):
        previous = getattr(self._local, "run", None)
        run = RunState(
            results={},
            best_distances={target: math.inf for target in self.config.capture_ts},
        )
        self._local.run = run
        return run, previous

    def record_error(self, run, error):
        if run.error is None:
            run.error = str(error)

    def fail_run(self, run, previous, error):
        run.status = "failed"
        run.error = str(error)
        self._local.run = previous

    def finish_run(self, run, previous, samples):
        fingerprint = _sample_fingerprint(samples)
        if run.error is not None:
            run.status = "error"
        elif len(run.results) != len(self.config.capture_ts):
            run.status = "error"
            run.error = "Sampling completed without all requested attention captures."
        else:
            run.status = "complete"
        with self._lock:
            self._completed[fingerprint] = run
            self._completed.move_to_end(fingerprint)
            while len(self._completed) > 8:
                self._completed.popitem(last=False)
        self._local.run = previous

    def current_run(self):
        return getattr(self._local, "run", None)

    def better_targets(self, run, actual_t):
        if run.status != "running" or run.error is not None:
            return ()
        return tuple(
            target
            for target, best_distance in run.best_distances.items()
            if abs(actual_t - target) < best_distance
        )

    def store(self, run, result, targets):
        if run.status != "running" or run.error is not None:
            return
        for target in targets:
            distance = abs(result.actual_t - target)
            if distance < run.best_distances[target]:
                run.best_distances[target] = distance
                run.results[target] = result

    def result_for(self, samples):
        tensor = samples.get("samples") if isinstance(samples, dict) else None
        if self.disabled:
            raise ProbeInputError("The attention probe is disabled.")
        if tensor is None:
            raise ProbeInputError("Preview samples do not contain a LATENT samples tensor.")
        fingerprint = _sample_fingerprint(tensor)
        with self._lock:
            run = self._completed.get(fingerprint)
            if run is None:
                raise ProbeInputError("Preview samples are not the output of the KSampler run captured by this probe.")
        if run.status != "complete":
            detail = f": {run.error}" if run.error else ""
            raise ProbeInputError(f"Attention probe status is {run.status}{detail}")
        return tuple(run.results[target] for target in self.config.capture_ts)


def _attention_mask_for_queries(mask, q, k, heads, context, target_indices, head_index):
    if mask is None:
        return None
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim != 4:
        raise ProbeInputError(f"Attention mask must have 2, 3, or 4 dimensions, got {mask.ndim}.")
    expected = (q.shape[0], heads, q.shape[1], k.shape[1])
    if any(size not in (1, wanted) for size, wanted in zip(mask.shape, expected)):
        raise ProbeInputError(f"Attention mask shape {tuple(mask.shape)} cannot broadcast to {expected}.")
    batch_row = 0 if mask.shape[0] == 1 else context.batch_row
    mask_head = 0 if mask.shape[1] == 1 else head_index
    selected = mask[batch_row, mask_head]
    if selected.shape[0] != 1:
        selected = selected.index_select(0, target_indices.to(selected.device))
    if selected.shape[1] == 1:
        selected = selected.expand(selected.shape[0], k.shape[1])
    return selected


def _capture_attention(q, k, attention_mask, heads, context, config):
    layout = context.layout
    if config.reference_index >= len(layout.references):
        raise ProbeInputError(
            f"reference_index {config.reference_index} is out of range for {len(layout.references)} references."
        )
    if q.ndim != 3 or k.ndim != 3:
        raise ProbeInputError(f"Expected flattened attention tensors [B,N,C], got Q={tuple(q.shape)}, K={tuple(k.shape)}.")
    if context.batch_row >= q.shape[0]:
        raise ProbeInputError(f"Selected batch row {context.batch_row} is out of range for Q batch {q.shape[0]}.")
    if config.head_index >= heads:
        raise ProbeInputError(f"head_index {config.head_index} is out of range for {heads} heads.")
    if q.shape[-1] % heads or k.shape[-1] % heads:
        raise ProbeInputError(f"Attention channels are not divisible by {heads} heads.")
    if q.shape[1] != layout.target.end - layout.target.start or k.shape[1] != layout.total_tokens:
        raise ProbeInputError(
            f"Qwen21 probe layout mismatch: target queries={q.shape[1]}, expected={layout.target.end - layout.target.start}, "
            f"keys={k.shape[1]}, expected={layout.total_tokens}."
        )

    target_indices = select_target_queries(config, layout.target)
    reference = layout.references[config.reference_index]

    with torch.no_grad():
        if q.is_cuda:
            torch.cuda.synchronize(q.device)
            baseline_memory = torch.cuda.memory_allocated(q.device)
            observed_peak_delta = 0
        else:
            baseline_memory = None
            observed_peak_delta = None
        started = time.perf_counter()
        head_dim = q.shape[-1] // heads
        q_head = q.reshape(q.shape[0], q.shape[1], heads, head_dim)[
            context.batch_row, :, config.head_index
        ]
        k_all = k.reshape(k.shape[0], k.shape[1], heads, head_dim)[
            context.batch_row, :, config.head_index
        ].float()
        heatmap_sum = torch.zeros(reference.end - reference.start, device=q.device, dtype=torch.float32)
        mass_sum = torch.zeros((), device=q.device, dtype=torch.float32)
        if q.is_cuda:
            observed_peak_delta = max(observed_peak_delta, torch.cuda.memory_allocated(q.device) - baseline_memory)
        for start in range(0, target_indices.numel(), config.query_chunk_size):
            chunk_indices = target_indices[start:start + config.query_chunk_size]
            query_chunk = q_head.index_select(0, chunk_indices.to(q.device)).float()
            logits = query_chunk @ k_all.transpose(0, 1)
            logits *= head_dim ** -0.5
            if q.is_cuda:
                observed_peak_delta = max(observed_peak_delta, torch.cuda.memory_allocated(q.device) - baseline_memory)
            chunk_mask = _attention_mask_for_queries(
                attention_mask, q, k, heads, context, chunk_indices, config.head_index
            )
            if chunk_mask is not None:
                if chunk_mask.dtype == torch.bool:
                    logits.masked_fill_(~chunk_mask, -math.inf)
                else:
                    float_mask = chunk_mask.float()
                    if q.is_cuda:
                        observed_peak_delta = max(observed_peak_delta, torch.cuda.memory_allocated(q.device) - baseline_memory)
                    logits += float_mask
                    del float_mask
            probabilities = torch.softmax(logits, dim=-1)
            probabilities.nan_to_num_(nan=0.0)
            if q.is_cuda:
                observed_peak_delta = max(observed_peak_delta, torch.cuda.memory_allocated(q.device) - baseline_memory)
            reference_probabilities = probabilities[:, reference.start:reference.end]
            chunk_sum = reference_probabilities.sum(dim=0)
            heatmap_sum += chunk_sum
            mass_sum += chunk_sum.sum()
            del logits, probabilities, reference_probabilities, chunk_sum

        if q.is_cuda:
            torch.cuda.synchronize(q.device)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

    query_count = int(target_indices.numel())
    heatmap = (heatmap_sum / query_count).reshape(reference.height, reference.width).cpu()
    positions = () if config.sampling_mode == "full_chunked" else tuple(
        (int(index // layout.target.width), int(index % layout.target.width)) for index in target_indices
    )
    return ProbeResult(
        heatmap=heatmap,
        attention_mass=float((mass_sum / query_count).cpu()),
        actual_t=context.actual_t,
        block_index=config.block_index,
        head_index=config.head_index,
        reference_index=config.reference_index,
        query_count=query_count,
        total_target_queries=layout.target.height * layout.target.width,
        sampling_mode=config.sampling_mode,
        query_chunk_size=config.query_chunk_size,
        sampled_positions=positions,
        reference_span=(reference.start, reference.end),
        target_span=(layout.target.start, layout.target.end),
        reference_hw=(reference.height, reference.width),
        target_hw=(layout.target.height, layout.target.width),
        total_tokens=layout.total_tokens,
        attention_mask_applied=attention_mask is not None,
        probe_elapsed_ms=elapsed_ms,
        probe_observed_peak_allocated_delta_bytes=observed_peak_delta,
    )


class EffectiveAttentionProbe:
    def __init__(self, handle, context, config, previous):
        self.handle = handle
        self.context = context
        self.config = config
        self.previous = previous

    def _capture_then_call(self, function, q, k, v, heads, mask=None, **kwargs):
        block_index = kwargs.get("transformer_options", {}).get("block_index")
        target_length = self.context.layout.target.end - self.context.layout.target.start
        is_target_attention = (
            isinstance(q, torch.Tensor)
            and isinstance(k, torch.Tensor)
            and q.ndim == 3
            and k.ndim == 3
            and q.shape[1] == target_length
            and k.shape[1] == self.context.layout.total_tokens
        )
        targets = self.handle.better_targets(self.context.run, self.context.actual_t)
        if block_index == self.config.block_index and is_target_attention and targets:
            try:
                result = _capture_attention(q, k, mask, heads, self.context, self.config)
                self.handle.store(self.context.run, result, targets)
            except ProbeInputError as error:
                self.handle.record_error(self.context.run, error)
        return function(q, k, v, heads, mask=mask, **kwargs)

    def __call__(self, function, q, k, v, heads, mask=None, **kwargs):
        if self.previous is None:
            return self._capture_then_call(function, q, k, v, heads, mask=mask, **kwargs)

        def capture_then_call(q, k, v, heads, mask=None, **inner_kwargs):
            return self._capture_then_call(function, q, k, v, heads, mask=mask, **inner_kwargs)

        return self.previous(capture_then_call, q, k, v, heads, mask=mask, **kwargs)


def install_probe(model, handle):
    config = handle.config

    def outer_sample_wrapper(executor, *args, **kwargs):
        run, previous = handle.start_run()
        try:
            samples = executor(*args, **kwargs)
        except BaseException as error:
            handle.fail_run(run, previous, error)
            raise
        try:
            handle.finish_run(run, previous, samples)
        except BaseException as error:
            handle.fail_run(run, previous, error)
            raise
        return samples

    def diffusion_wrapper(executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options={}, **kwargs):
        options = transformer_options.copy()
        run = handle.current_run()
        if run is None:
            return executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)
        try:
            layout = build_sequence_layout(x, context, ref_latents, image_slots)
            batch_row, logical_batch = _resolve_batch_row(
                options, int(x.shape[0]), config.batch_index, config.condition_branch
            )
            actual_t = _actual_t(timestep, x, batch_row, logical_batch, config.batch_index)
            forward_context = ForwardContext(layout, batch_row, actual_t, run)
            options["optimized_attention_override"] = EffectiveAttentionProbe(
                handle,
                forward_context,
                config,
                options.get("optimized_attention_override"),
            )
        except ProbeInputError as error:
            handle.record_error(run, error)
        return executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)

    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, WRAPPER_KEY, outer_sample_wrapper)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY, diffusion_wrapper)
    return patched


def validate_model(model, block_index, head_index):
    diffusion_model = model.model.diffusion_model
    if not isinstance(diffusion_model, QwenImage21Transformer2DModel):
        raise ProbeInputError(
            "Qwen21 Reference Attention Probe requires a native Qwen-Image-2.1 diffusion model."
        )
    block_count = len(diffusion_model.transformer_blocks)
    if block_index >= block_count:
        raise ProbeInputError(f"block_index {block_index} is out of range for {block_count} blocks.")
    head_count = diffusion_model.transformer_blocks[block_index].attn.heads
    if head_index >= head_count:
        raise ProbeInputError(f"head_index {head_index} is out of range for {head_count} heads.")
