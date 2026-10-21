# ComfyUI Qwen-Image-2.1 Attention Probe

Read-only diagnostics for native Qwen-Image-2.1 reference-image attention. The probe measures how one selected DiT attention head distributes target queries over one VAE reference-token grid.

It does not inject features, replace Q/K/V, modify prompts, train weights, or claim that attention is causal attribution. It observes the DiT VAE-token path, not the separate Qwen3-VL vision encoder.

## Install

Install the parent `ComfyUI-qwen_img_2_1_enhancer` directory under `ComfyUI/custom_nodes/` and restart ComfyUI. No additional Python packages are required.

## Workflow

1. Connect the diffusion model to **Qwen21 Reference Attention Probe**, then connect `patched_model` to KSampler.
2. Connect the Probe's `probe_handle` and the same KSampler latent output to **Qwen21 Reference Attention Probing**. The latent connection is an execution dependency checked against the exact captured result.
3. Connect `attention_data` and the same reference image to **Qwen21 Reference Attention Preview**. `reference_index` is zero-based and follows the encoder's image order.

`sampling_mode=uniform_1d` preserves the original flattened-grid subsampling and uses at most `max_queries`. `sampling_mode=full_chunked` visits every target query and ignores `max_queries`; `query_chunk_size` controls the temporary query-by-all-keys matrix size.

`capture_t` accepts one or more comma-separated normalized timesteps, for example `0,0.25,0.5,0.75,1`. Each requested value uses the closest timestep actually visited by the sampler. The overlay is an IMAGE batch in the same order; per-capture `actual_t` and attention mass values are recorded in metadata. The standalone FLOAT mass output remains the first requested capture.

Three sanitized example graphs are included in the parent package's `workflows/` directory. Select locally installed model files, a reference image, and prompts after opening one.

## Probing outputs

- `attention_data`: the original float32 reference-token grids and capture information consumed by Attention Preview.
- `ref_attention_mass`: the fraction of all-key effective attention assigned to the selected reference, averaged over the processed target queries. Additive or boolean inference masks installed by compatible model patches are included before softmax.
- `metadata`: capture settings, actual timestep, token spans and grids, Query coverage, Probe elapsed time, sampled CUDA allocator delta, and warnings.

**Qwen21 Reference Attention Preview** renders only `overlay`, a blue-to-yellow-to-red relative visualization over the supplied reference image. Hover it to inspect reference pixel coordinates, the nearest token's raw attention, bilinearly sampled raw attention, and display intensity. Hover **Qwen21 Reference Mask Preview** to inspect 8-bit mask opacity; it passes the MASK through unchanged.

For a uniform fractional mask, right-click the Core **LoadImage** node and select **Open Precise Mask Editor**. It writes absolute 8-bit values, so repeated strokes do not accumulate opacity. Saving creates a new alpha-bearing PNG copy and switches that LoadImage to the copy without changing the original input file.

The underlying reference-grid values sum to `ref_attention_mass` within normal floating-point tolerance. Display scaling affects only the overlay, never hover's raw values or numeric mass. `full_chunked` is an all-query measurement; `uniform_1d` is not when `max_queries` is smaller than the target-token count.

Uniform query subsampling is performed on the flattened row-major target grid. On some grid widths its nearly constant stride forms a visible two-dimensional lattice; RoPE position affinity can make that lattice appear as diagonal repeating peaks. Treat this as a sampling artifact unless it survives comparison with other query selections and heads.

For attention-modifying model nodes, connect the model as `MODEL -> Probe -> modifier -> sampler`. This lets the modifier pass its final attention mask through the probe, so the heatmap and mass reflect the effective softmax. The probe observes the optimized-attention call and does not install an `attn1_patch`, so native Qwen Image 2.1 prefix caching remains available.

## Important limitations

- This node targets the checked-out native `QwenImage21Transformer2DModel`. Other Qwen versions and model wrappers that replace that class are rejected.
- The probe recomputes softmax rows in float32 for diagnostics. It does not replace the backend output, but `full_chunked` adds runtime proportional to all target queries. Only one query chunk and the aggregate reference grid are retained.
- CUDA memory metadata is an observed allocator delta sampled at Probe tensor boundaries. It does not reset process-wide CUDA peak counters and is not an isolated whole-inference VRAM benchmark.
- Attention is softmaxed over every key visible to target-image queries: text, all references, and target tokens. The overlay is never normalized only over the chosen reference.
- A very small mass can still produce high-contrast relative colors. Metadata warns below `1e-4`; use the numeric mass and raw hover value when judging whether the model meaningfully read the reference.
- The preview image is for display only. Its crop and aspect ratio must match the image supplied to Qwen21 conditioning.

## Errors

The Probe rejects a non-Qwen21 model and out-of-range block/head settings immediately. Missing reference latents, unavailable CFG branches, invalid batch selection, interrupted sampling, and mismatched sampler outputs are reported by Probing. Internal sequence-layout disagreement aborts sampling rather than producing a plausible-looking but incorrect heatmap.
