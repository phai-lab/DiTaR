# Diffusion latent-normalization stats

`q4_stats.json` holds the per-channel WAN-source / TC-target normalization stats consumed by
`configs/diffusion_q4.yaml` (`target_stats_path` / `source_stats_path`). It is **data-derived**
and intentionally not shipped.

Generate it from a trained Stage-2 encoder (the diffusion teacher):

```bash
python scripts/compute_diffusion_stats.py \
    --config configs/diffusion_q4.yaml \
    --stage2-ckpt $DITAR_STAGE2_CKPT \
    --out configs/stats/q4_stats.json \
    --max-batches 200
```

The script reuses the model's exact frozen-teacher extraction path
(`WanTcLatentDiffusionLAM._extract_online_latents`), so the stats match training. Output schema:

```json
{"target_mean": [..32..], "target_std": [..32..],
 "source_mean": [..48..], "source_std": [..48..]}
```

If the file is missing, the diffusion model raises an actionable error pointing back here. For a
quick development/smoke run you may instead set `target_stats_path: ''` and `source_stats_path: ''`
in the config (identity normalization) — this does **not** reproduce the paper recipe.
