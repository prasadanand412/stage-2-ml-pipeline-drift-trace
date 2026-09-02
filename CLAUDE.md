# CLAUDE.md — oiltrace Project Context

## What this project is
oiltrace (internal codename; official submission may be titled differently) is a three-stage AI pipeline built for Smart India Hackathon 2026, Problem Statement SIH26143 (issued by NTRO): detect oil spills from satellite SAR imagery, trace them backward to estimate origin, and identify likely responsible vessels using AIS ship-tracking data correlation.

## Repository structure
```
oiltrace/
├── frontend/        # React + Vite web client (already scaffolded by team)
├── backend/         # FastAPI orchestration layer
├── ml/
│   ├── detection/    # Stage 1: SAR segmentation (U-Net, PyTorch) — COMPLETE
│   ├── drift_trace/  # Stage 2: OpenDrift-based backward/forward tracing — in progress
│   ├── attribution/  # Stage 3: AIS data pipeline & vessel scoring — not started
│   ├── datasets/     # training/test data (gitignored where large)
│   └── notebooks/    # exploration notebooks
└── docs/             # documentation, submission materials, STAGE2_3_HANDOFF.md
```

## Tech stack conventions
- **ML**: PyTorch (CUDA build — this project requires the nightly build with CUDA 12.8+ due to newer GPU hardware; see `docs/dev_environment_notes.md`). Python 3.12 (not 3.14 — PyTorch compatibility).
- **Backend**: FastAPI, Python. REST endpoints follow `/detect`, `/trace-origin`, `/rank-vessels` naming.
- **Frontend**: React + Vite. Map rendering via Mapbox GL JS or Leaflet.js.
- **Data**: PostgreSQL + PostGIS for storage; GeoJSON as the interchange format between backend and frontend for spatial data.
- **Environmental data sources**: Copernicus Marine Service (ocean currents), OpenWeatherMap (wind — NOAA NOMADS OpenDAP is retired as of Feb 2026, do not use it).

## Model architecture conventions
- **Stage 1 (Detection)**: U-Net, single-channel grayscale SAR input (256x256), binary output mask (oil probability per pixel), trained with BCELoss.
- **Stage 2 (Drift Trace)**: Not a trained model — uses OpenDrift (physics simulator) directly at inference time, run in reverse for origin estimation and forward for spread prediction. Real wind/current data required, not fallback/dummy values, for production use.
- **Stage 3 (Attribution)**: Rule-based scoring (proximity + trajectory alignment + behavioral anomaly flags), not a trained model, at least for the initial version.

## Code style notes
- Keep model definitions (`UNet`, dataset classes, etc.) in `ml/*/models.py` or similar — not left inline in notebooks once stabilized, so the backend can import them directly.
- Use `torch.device("cuda" if torch.cuda.is_available() else "cpu")` pattern consistently; never hardcode device.
- All new environmental/AIS data fetchers should cache raw downloads locally rather than re-fetching on every call — external APIs (Copernicus, AIS sources) are rate-limited and slow.

## Known environment gotchas (do not repeat these mistakes)
- RTX 5060 (Blackwell) requires PyTorch nightly + CUDA 12.8; stable PyTorch (cu124 and earlier) does not support this GPU (sm_120).
- Python 3.14 is not yet supported by PyTorch — use a Python 3.12 environment/kernel.
- NOAA NOMADS OpenDAP format was retired Feb 2026 — use OpenWeatherMap or NOMADS Grib Filter (with cfgrib) instead, not the old OpenDAP URL pattern.
- numpy version must stay compatible with scipy's supported range (`numpy<2.5.0` as of this project's setup) — a plain `pip install numpy` can grab too new a version and break scipy/opendrift imports.

## Current status (as of Sept 2, 2026 — internal target: working end-to-end by Sept 6, hackathon event Sept 10)

**Stage 1 (Detection) — COMPLETE.**
- U-Net trained 100 epochs (10 on local RTX 5060, 90 on a rented RTX 5090) on the Kaggle "Deep-SAR SOS" Sentinel-1 subset (256x256 PNG image/label pairs).
- Result: **Average IoU 0.6630** on 839 test images.
- Known limitation: some false positives on look-alike/no-spill cases (a documented hard problem in SAR oil-detection research) — state this honestly in docs, don't hide it.
- Files: `sar_detection_pipeline.ipynb` (clean: load model → evaluate → visualize → inference), `sar_detection_model_training.ipynb` (full training history, kept for reference), `sar_unet_best.pt` (tracked via Git LFS), `training_results.json`, loss curve images.
- Inference function `predict_oil_spill(image_path, model, device, threshold=0.5)` is built and tested — returns a binary mask, oil pixel count, and area fraction.
- Known gap: no georeferencing step yet. Real satellite imagery carries geospatial metadata (GeoTIFF) mapping pixels to real lat/long, but the training dataset used (PNG-based) stripped this out. Demo scenarios use a manually-specified lat/long rather than one derived from the image. Documented as a known production-integration gap, not a blocker for idea-submission stage.

**Stage 2 (Drift Trace) — in progress.**
- OpenDrift backward/forward tracing prototyped successfully on dummy/fallback wind & current data — confirmed working, including correct stranding behavior against real coastline data.
- Remaining: wire in real wind (OpenWeatherMap) and current (Copernicus Marine) data, handle early-stranding edge cases gracefully, finish the clean `trace_origin()` wrapper function, validate against a real historical case if one is found by Areen.
- Working notebook: `drift_trace_pipeline.ipynb` (separate from Stage 1's notebooks).

**Stage 3 (Vessel Attribution) — not yet started in code; being built independently (e.g., via Claude Code), not blocked on Nishad.**
- Nishad (nominally responsible for this stage) has shown inconsistent engagement — do not wait on him; treat any of his output as a bonus/backup, not a dependency.
- Will have its own separate notebook: `ml/attribution/attribution_pipeline.ipynb`, plus `ml/attribution/attribution.py` with `rank_vessels()` as the main entry point.
- Plan: use synthetic AIS data for initial testing (a planted "correct" suspect vessel in an otherwise randomized dataset) since real AIS data acquisition may not happen in time — this is an accepted, documented simplification, not a hidden shortcut.

See `docs/STAGE2_3_HANDOFF.md` for the full detailed task breakdown per stage, and `TASKS.md` for granular checklist tracking.

## Team
5-person team, roles and ownership documented in `docs/SpillTrace_SIH26143_Documentation.docx` Section 15 and `docs/STAGE2_3_HANDOFF.md` Section 1. ML/architecture decisions are owned by the ML lead (Prasad); do not restructure model architectures without checking `PLAN.md` / `STAGE2_3_HANDOFF.md` for the agreed approach first.
