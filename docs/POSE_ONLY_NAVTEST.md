# Pose-only Drive-JEPA and NAVTEST notes

Based on upstream Drive-JEPA commit `e21f47410b4d26b61f05f9bd23e169c0390cae2a`, using its vendored NAVSIM 2.0.0. These are local planner-training experiments using released V-JEPA weights; they do not reproduce video pretraining or the complete original hardware/training recipe.

## Accepted experiment design

The compatible, opt-in `pdm_supervision=false` mode retains camera and ego inputs, 32 proposals, four shared refinement stages, and score/auxiliary heads. It bypasses PDM training workers, metric-cache and anchor loading, good-anchor pseudo targets, and all score/agent/area/BEV auxiliary losses. Validation uses pose loss and checkpoint selection minimizes `val/loss_epoch`.

Every proposal at every stage is supervised against the same logged future trajectory using mean L1 over batch, proposals, time, and coordinates. With `prev_weight=0.1`, the objective is `0.001 L1 + 0.01 L2 + 0.1 L3 + L4`. It is not the original best-of-proposals loss with only the auxiliary terms set to zero. No velocity or smoothness loss was added.

The scorer still executes and inference still uses its final-score argmax. Its score head receives no score supervision in this mode; its output must not be interpreted as a trained PDM estimate. All proposals are pulled toward the same target, but identical outputs are not guaranteed. This behavior was retained for the completed experiment and checkpoint compatibility.

Only two front-camera images (256 x 512) reach V-JEPA. The latest 11-dimensional ego vector includes pose, velocity, acceleration and command. Prediction is eight relative poses over four seconds. The target builder retains the token as metadata; pose-only loss reads only the trajectory. Dataset infrastructure can still load scene annotations/maps; the claim is absence of annotation supervision in the objective, not a completely annotation-free dataset loader.

The 4060 profile freezes the V-JEPA image backbone, while training its projector and planner. It uses a differentiable PyTorch deformable-attention fallback and localizes released cache paths. The existing anchor-loss refinement-loop correction is also included. Model and feature-builder architecture files otherwise retain the release behavior.

## Completed experiments

Both runs used NAVTRAIN, batch 32, accumulation 1, 20 epochs, mixed precision, gradient clipping 1.0 and Adam with planner learning rate 1e-4. The pose-only run started with released V-JEPA weights and no planner checkpoint. Validation/checkpoint criteria differ by objective; logging and sanity-validation settings also differ.

| Run under `/media/dell/Storage/Drive-JEPA-runs` | Selected checkpoint | Stored NAVTEST all-frame score | Post-filter visible-metric recomputation |
| --- | --- | --- | --- |
| `full-navtrain-4060-bs32` | `drive_jepa_navsim_v2_4060/checkpoints/best_epoch=19.ckpt` | 0.8733336118 | 0.9088948961 |
| `pose-only-navtrain-4060-v2` | `drive_jepa_navsim_v2_4060/checkpoints/best_epoch=14.ckpt` | 0.8362689851 | approximately 0.8756112 |

Both original evaluations completed 12,146 scenarios with zero failures. Pose-only best validation loss was approximately 0.278, and final epoch validation loss was approximately 0.313. The earlier `pose-only-navtrain-4060` attempt was interrupted and superseded by `-v2`.

The original CSVs are `drive_jepa_navtest_epdms_epoch19/2026.08.31.20.06.39/2026.08.31.21.36.28.csv` and `pose-only-navtest-epdms/2026.09.05.16.07.39.csv`. Checkpoints, datasets and run outputs are not included in this repository.

## NAVTEST aggregation issue

The release's human-penalty filter updates visible metric columns but leaves `multiplicative_metrics_prod` and `weighted_metrics` stale. The sequential evaluator subsequently calculates final scores from those hidden aggregates. This can produce a stored zero even when all visible metrics equal one. The issue was reproduced in a fresh Drive-JEPA checkout; the previous investigation traced it to historical NAVSIM code and found a fix in NAVSIM v2.2 (upstream issue #151).

This fork includes the previously prepared correction in `navsim_v2/navsim/evaluate/pdm_score.py`: after human filtering, rebuild the multiplicative product and weighted values from the updated columns, then recompute `pdm_score`. `weighted_metrics_array` holds weights and is preserved. The sequential evaluator subsequently incorporates two-frame comfort as before.

The recomputed values above are arithmetic audits of saved visible metrics, not new completed model-evaluation runs or verified leaderboard results. The saved CSVs lack the state information needed to reconstruct pseudo-closed-loop weighting independently; that row was not corrected in the full-run recomputed CSV. Do not compare a corrected result directly with a historical uncorrected score without stating the evaluator difference. The logger also averages after appending summary rows; use scenario rows or `average_all_frames` for the reported all-frame result.

The corrected full-model and pose-only inference attempts did not produce final CSVs. During the September 14 inspection, four orphaned pose-only worker processes remained after the parent had been stopped; logs had not advanced since September 5. That is an incomplete run, not evidence of a new corrected result.

## Reproduction on Dell

The scripts have Dell-specific defaults; override paths for another machine. Set `DRIVE_JEPA_FREEZE_VJEPA=1` and `DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE=1` for the completed profile. The training launcher accepts:

```bash
WANDB_MODE=offline DRIVE_JEPA_PDM_SUPERVISION=0 \
BATCH_SIZE=32 ACCUMULATE_GRAD_BATCHES=1 MAX_EPOCHS=20 \
NAVSIM_OUTPUT_ROOT=/path/to/new-run \
bash scripts/train_navsim_v2_4060.sh
```

Use a fresh output directory. Override `PYTHON`, `OPENSCENE_DATA_ROOT`, `DRIVE_JEPA_CACHE_ROOT`, and `NUPLAN_MAPS_ROOT` as appropriate. Default `DRIVE_JEPA_PDM_SUPERVISION=1` retains the original objective. For direct Hydra use, pass `+agent.config.pdm_supervision=false`.

For NAVTEST, with those environment variables and `NAVSIM_DEVKIT_ROOT`/`PYTHONPATH` pointing to this checkout's `navsim_v2`:

```bash
"$PYTHON" navsim_v2/navsim/planning/script/run_pdm_score_sequential.py \
  train_test_split=navtest agent=drive_jepa_perception_based_agent \
  agent.config.latent=false +agent.config.pdm_supervision=false \
  'agent.checkpoint_path="/path/to/best_epoch=14.ckpt"' \
  worker=sequential \
  metric_cache_path=/path/to/metric_cache_v2 \
  navsim_log_path=/path/to/navsim_logs/test \
  sensor_blobs_path=/path/to/sensor_blobs/test \
  synthetic_scenes_path=/path/to/synthetic_scenes \
  experiment_name=pose_only_navtest output_dir=/path/to/new-evaluation
```

PDM and map assets remain required for evaluation. The original pose-only evaluation used four process workers; sequential execution is shown for simpler process management. Reuse saved metrics for arithmetic audits rather than repeating GPU inference. The existing `DRIVE_JEPA_SAMPLE_CONTRACT.md` describes the earlier PDM baseline and should be read together with these later notes.
