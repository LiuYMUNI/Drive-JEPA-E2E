# Drive-JEPA NAVSIM v2 Sample Contract and Training Audit

This document describes the implementation that is actually present in this
workspace. It is a contract for one NAVSIM v2 training/evaluation sample, not a
generic description of the Drive-JEPA paper. The completed experiment is
**NAVSIM action training with a pretrained V-JEPA image encoder**, not V-JEPA
self-supervised video pretraining.

The source of truth is the code under `navsim_v2/`, the resolved Hydra
configuration saved with the run, and the final checkpoint:

- Code: `navsim_v2/navsim/agents/drive_jepa_perception_based/`
- Training entry point: `navsim_v2/navsim/planning/script/run_training.py`
- Resolved config: `/media/dell/Storage/Drive-JEPA-runs/full-navtrain-4060-bs32/drive_jepa_navsim_v2_4060/code/hydra/config.yaml`
- Checkpoint: `/media/dell/Storage/Drive-JEPA-runs/full-navtrain-4060-bs32/drive_jepa_navsim_v2_4060/checkpoints/best_epoch=19.ckpt`

## 1. One-sentence contract

Given a sliding NAVSIM log window containing **4 history frames and 10 future
frames**, construct two front RGB images (history indices 2 and 3), the latest
11-value ego state, and current-frame camera calibration metadata; predict 32
candidate local trajectories of 8 `(x, y, heading)` poses over 4 seconds; score
the candidates with PDM-derived supervision and auxiliary collision/area heads;
select the candidate with the largest predicted final score.

## 2. Dataset window and time convention

The NAVSIM v2 `navtrain` and `navtest` scene filters both resolve to:

```yaml
num_history_frames: 4
num_future_frames: 10
frame_interval: 1
has_route: true
```

Frames are sampled at 2 Hz (`0.5 s` spacing). For a starting row `i`, the
scene window contains rows `i .. i+13`:

| Window portion | Rows | Time relative to current frame | Used for |
| --- | ---: | ---: | --- |
| History | `i .. i+3` | `-1.5, -1.0, -0.5, 0.0 s` | Agent input |
| Future | `i+4 .. i+13` | `+0.5 .. +5.0 s` | Ground-truth scene context |

The current frame is history row `i+3` (`num_history_frames - 1`). The target
builder asks for `num_poses=8`, so it uses only future rows `i+4 .. i+11`, a
4-second target. The last two future rows are loaded by the scene contract but
are not part of the supervised trajectory tensor.

`SceneLoader` creates one sample for every valid sliding start in each selected
log (`frame_interval: 1`). A sample is represented by:

```text
(features: Dict[str, Tensor], targets: Dict[str, Tensor])
```

The `Dataset` class can compute these dictionaries from raw scenes or load
gzip-pickled dictionaries named `pad_feature.gz` and `pad_target.gz`. The final
run used `cache_path=null`, so it streamed raw scene data rather than loading a
feature cache.

### Official repository workflow versus this 4060 run

The released Drive-JEPA scripts use a two-step, offline materialization
workflow:

1. `scripts/training/run_drive_jepa_perception_based_cache.sh` runs
   `run_dataset_caching.py`. That constructs each scene, calls the feature and
   target builders once, and writes `pad_feature.gz` and `pad_target.gz` under
   the cache directory.
2. `scripts/training/train_drive_jepa_perception_based.sh` passes that cache
   with `use_cache_without_dataset=True`. `CacheOnlyDataset` then reads the
   compressed dictionaries during training and does not rebuild raw scenes or
   camera features each epoch.

The generic NAVSIM `Dataset` also supports online materialization when
`cache_path=null`, and it can populate a cache in its constructor when a cache
path is supplied. However, that is not the released Drive-JEPA shell-script
workflow; the dedicated cache script is intended to perform the one-time
precomputation.

The completed RTX 4060 experiment deliberately used the online mode:
`cache_path=null` and `use_cache_without_dataset=false`. Consequently, raw
scene loading, camera decoding, resizing, and feature/target construction were
repeated on demand by the DataLoader. This was chosen to avoid another large
feature-cache export and to fit the available storage/compute setup.

This distinction applies only to feature/target materialization. PDM proposal
simulation, metric scoring, and anchor-index supervision are computed inside
`compute_loss()` for each training batch in both modes; those losses are not
made offline by the feature cache. The metric-cache files and 8192-anchor
table themselves are precomputed assets.

## 3. Raw NAVSIM frame shape

Each raw log is a pickle containing a list of frame dictionaries. A typical
frame has the following fields:

```python
{
    "token": str,
    "timestamp": int,
    "scene_token": str,
    "log_name": str,
    "map_location": str,
    "ego2global_translation": float64[3],
    "ego2global_rotation": float64[4],       # quaternion
    "ego_dynamic_state": float64[4],          # velocity x/y, acceleration x/y
    "driving_command": int[4],
    "cams": {
        "CAM_B0": camera_record,
        "CAM_F0": camera_record,
        "CAM_L0": camera_record,
        "CAM_L1": camera_record,
        "CAM_L2": camera_record,
        "CAM_R0": camera_record,
        "CAM_R1": camera_record,
        "CAM_R2": camera_record,
    },
    "lidar_path": str,
    "anns": {
        "gt_boxes": float64[N, 7],
        "gt_names": str[N],
        "gt_velocity_3d": float64[N, 3],
        "instance_tokens": str[N],
        "track_tokens": str[N],
    },
    "roadblock_ids": list[str],
    "traffic_lights": list,
    ...
}
```

Each camera record contains a relative image path and calibration:

```python
{
    "data_path": str,
    "sensor2lidar_rotation": float32[3, 3],
    "sensor2lidar_translation": float32[3],
    "cam_intrinsic": float32[3, 3],
    "distortion": float32[5],
}
```

The image files in this dataset are normally `1920 x 1080` JPEGs. `Scene` also
materializes annotations, map API objects, and camera/lidar dataclasses, but
those privileged values are not all passed to the neural network.

## 4. Sensor loading contract

`DriveJEPAAgent.get_sensor_config()` requests:

```python
SensorConfig(
    cam_f0=[2, 3],
    cam_l0=[3],
    cam_l1=[], cam_l2=[],
    cam_r0=[3],
    cam_r1=[], cam_r2=[],
    cam_b0=[3],
    lidar_pc=[],
)
```

Therefore the four `AgentInput` history entries contain:

| History index | Loaded camera records | Lidar |
| ---: | --- | --- |
| 0 | none | none |
| 1 | none | none |
| 2 | `CAM_F0` | none |
| 3 | `CAM_B0`, `CAM_F0`, `CAM_L0`, `CAM_R0` | none |

Absent sensors are represented by empty `Camera`/`Lidar` dataclasses. The raw
dataset must contain all requested files for the requested history indices; a
missing image is a hard `Image.open` failure when that sample is materialized.

### Important implementation fact: only two front frames reach V-JEPA

`DriveJEPAFeatureBuilder._get_camera_feature()` uses only:

```python
cameras[-1].cam_f0.image   # history index 3, current front image
cameras[-2].cam_f0.image   # history index 2, previous front image
```

Each image is cropped vertically by `28:-28` pixels and resized to `(width,
height) = (512, 256)`. `ToTensor()` produces two tensors with shape
`float32[3, 256, 512]` and values initially in `[0, 1]`.

The four-view tensor made by `_get_bev_feature()` is **not the V-JEPA image
input**. It produces calibration/shape metadata. The actual trained visual
contract is two front views, not eight cameras or a four-view temporal BEV
video.

## 5. Feature dictionary contract

`DriveJEPAFeatureBuilder.compute_features()` returns these keys for one sample
before the DataLoader adds a batch dimension:

| Key | Shape and dtype | Construction and consumer |
| --- | --- | --- |
| `camera_feature_1` | `float32[3, 256, 512]` | Current front image; ImageNet-normalized in model forward |
| `camera_feature_2` | `float32[3, 256, 512]` | Previous front image; ImageNet-normalized in model forward |
| `camera_feature` | `float32[4, 3, 448, 768]` | Current `B0/F0/L0/R0` views after NAVSIM normalization, 0.4 scale, and 32-pixel padding; currently dead as a neural input |
| `img_shape` | `float32[4, 3]` | Four padded `(height, width, channels)` records, normally `[448, 768, 3]` each |
| `lidar2img` | `float32[4, 4, 4]` | Current-view lidar-to-image matrices in order `B0, F0, L0, R0` |
| `ego_status` | `float32[4, 11]` | Four history ego records; model uses only row `-1` |

`_get_bev_feature()` applies mean/std normalization
`([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])`, converts BGR to RGB,
scales by exactly `0.4`, and pads to a multiple of 32. The resulting
`camera_feature` is preserved in the dictionary but `ImgEncoder.forward()` is
called with the two explicit front tensors instead.

### Ego status layout

`AgentInput.from_scene_dict_list()` first converts all four global ego poses to
coordinates relative to the last history pose. For each frame the builder
concatenates:

```text
ego_status[t] = [pose_x, pose_y, pose_heading,
                 velocity_x, velocity_y,
                 acceleration_x, acceleration_y,
                 command_0, command_1, command_2, command_3]
```

The resulting `11` values are not standardized by this code. The command is
stored as a four-value integer/one-hot-like vector in the raw data, but there is
no explicit one-hot assertion. The model applies `Linear(11, 256)` to **only
the latest row**, so the earlier three ego records do not directly enter the
forward pass.

## 6. Target dictionary contract

`DriveJEPATargetBuilder.compute_targets()` returns:

```python
{
    "trajectory": float64[8, 3],
    "token": str,
}
```

`trajectory` is the human/operator future trajectory in local rear-axle
coordinates, with columns `(x, y, heading)`. Although the type annotation in
`Trajectory` mentions float32, the current implementation constructs this
tensor with `torch.tensor(numpy_float64_array)`, so the actual target dtype is
float64 unless a caller casts it. It is sampled at `0.5 s` for a 4-second
horizon. The origin is the current history frame (row `i+3`). The
target builder contains dormant implementations for agent boxes and a BEV
semantic map, but those keys are commented out and are not present in the
actual target dictionary.

## 7. Forward pass and tensor shapes

The resolved final configuration is:

```yaml
proposal_num: 32
num_poses: 8
ref_num: 4
tf_d_model: 256
tf_d_ffn: 1024
tf_num_head: 8
num_bev_layers: 1
num_points_in_pillar: 4
trajectory_sampling:
  time_horizon: 4
  interval_length: 0.5
agent_pred: true
area_pred: true
bev_map: false
bev_agent: false
```

The forward path is:

1. Select the front calibration matrix: `lidar2img[:, 1:2]`. Metadata order is
   `B0, F0, L0, R0`, so only `F0` is retained.
2. Normalize `camera_feature_1/2` with ImageNet mean/std and concatenate them
   as `float[B, 2, 3, 256, 512]` (or mixed precision on GPU).
3. Run the pretrained ViT-L V-JEPA encoder configured for resolution
   `(256, 512)` and two video frames. Its output is `float[B, 512, 1024]`.
4. Project `1024 -> 256` and reshape to image features
   `float[B, 256, 16, 32]`.
5. Encode the latest ego row with `Linear(11, 256)` and add it to a learned
   embedding of shape `[32 * 8, 256]`, producing refinement tokens
   `[B, 256, 256]` (32 proposals times 8 poses).
6. Apply four iterative refinement calls. The same `Traj_refiner` instance is
   inserted four times, so its weights are shared across all four stages. Each
   stage emits proposals `[B, 32, 8, 3]`. Pose references passed to BEVFormer
   are detached; they guide deformable attention but do not receive gradients
   through reference-point construction.
7. The scorer pools each proposal's eight pose tokens and emits
   `pred_logit: float[B, 32, 6]`. During training it also emits auxiliary heads.
8. `sigmoid(pred_logit[:, :, -1])` is the learned final-score estimate. The
   model selects `argmax` over 32 candidates and returns one
   `trajectory: float[B, 8, 3]`.

### Auxiliary output shapes during training

| Output | Shape | Meaning in code |
| --- | --- | --- |
| `pred_agents_states` | `[B, 32, 8, 40, 2, 9]` | Packed box-corner/validity predictions; last dimension is 8 box-corner coordinates plus 1 validity logit |
| `pred_area_logit` | `[B, 256, 10]` | Ten logits per proposal-pose token, reshaped for 40 rollout steps and 2 area flags |
| `pred_logit` | `[B, 32, 6]` | Six score logits per trajectory proposal |

The `40` dimension comes from PDM's 0.1-second evaluation sampling. The
`pred_agents_states` layout is unusual: the loss reshapes the packed
`[8, 40, 2, 9]` tensor to the target layout rather than comparing a plainly
named per-time-step structure. Any replacement implementation must preserve
this reshape contract or change both head and loss together.

## 8. PDM/anchor supervision path

The trajectory and auxiliary targets are generated online from the metric
cache, not read from the human target dictionary alone.

For every batch sample, `compute_score(..., test=False)`:

1. Loads a metric cache identified by the target token.
2. Converts each 8-pose proposal to 40 poses at `0.1 s` and simulates it with
   `PDMSimulator`.
3. Scores the simulated proposals with `PDMScorer` against the cached
   observation, centerline, route lanes, drivable map, and past human path.
4. Returns proposal scores, collision/TTC key-agent geometry, ego-area flags,
   and an `anchors_scores_index` array.

The PDM scorer has four multiplicative metrics:

```text
no-at-fault collision
drivable-area compliance
traffic-light compliance
driving-direction compliance
```

and four weighted metrics:

```text
progress (weight 5)
time-to-collision (weight 5)
lane keeping (weight 2)
history comfort (weight 2)
```

The training-time final score is the PDM score **without** two-frame extended
comfort. The sequential NAVSIM v2 evaluation adds that metric afterward using
consecutive simulated frames. The anchor table contains 8192 trajectories of
shape `[40, 3]`; `data/8192.npy[:, 4::5]` selects their 8-pose versions of shape
`[8192, 8, 3]`. Each sample's score-index file normally provides up to 256
candidate anchor IDs.

## 9. Exact loss function used by the final run

Let `B` be batch size, `P=32` proposals, `T=8` poses, and `R=4` refinement
stages. The implementation is in
`navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py`.

### 9.1 Trajectory loss with human and anchor targets

For each refinement stage `r`, the human min-over-proposals term is:

```text
L_gt(r) = mean_b min_p mean_t || proposal[r,b,p,t] - target[b,t] ||_1
```

For each batch item, the code samples at most four anchor IDs from its
PDM-derived score-index array. It computes the mean over sampled anchors of the
minimum proposal distance, then adds it with a factor of `0.5`:

```text
L_anchor(r) = 0.5 * mean_b mean_k min_p mean_t
               || proposal[r,b,p,t] - anchor[b,k,t] ||_1
```

The implementation combines these as `min_loss = L_gt + L_anchor` (with the
anchor accumulator divided by batch length), computes a logged diversity
quantity, and recursively aggregates stages:

```text
L_stage(r) = 0.1 * L_stage(r-1) + min_loss(r)
             + inter_weight * diversity(r)
L_trajectory = L_stage(3)
```

`diversity(r)` is the negative minimum pairwise proposal L1 distance. In the
resolved configuration `inter_weight=0`, so it is logged but contributes no
gradient. The negative `inter_loss` values visible in the training log are
therefore diagnostics, not an active optimization term.

### 9.2 Score and auxiliary losses

The active score/auxiliary terms are:

```text
L_final_score = BCEWithLogits(pred_logit[:, :, -1], target_scores[:, :, -1])
L_pred_valid  = BCEWithLogits(predicted_validity, gt_valid)
L_pred_box    = masked L1(predicted_box_state, gt_box_state)
L_area        = BCEWithLogits(pred_area_logits, gt_ego_areas)
```

The final weighted objective is exactly:

```text
L = 1.0 * L_trajectory
  + 1.0 * L_final_score
  + 1.0 * L_pred_valid
  + 0.1 * L_pred_box
  + 2.0 * L_area
```

The code also computes `sub_score_loss`, but its configured weight is zero.
`pred_area_logits` are reshaped to `[B, 32, 40, 2]`; the two area labels are
the non-drivable-area and oncoming-traffic flags after the initial time step.
The multi-lane flag is not included in that target slice.

The BEV agent loss and semantic-map cross-entropy are inactive in the final
run:

- `bev_agent=false` means `pred["agent_states"] is None`, so
  the separate BEV-agent `agent_class_loss` and `agent_box_loss` are set to
  zero. The similarly named `pred_agents_states` head is still active and is
  trained by `score_loss` (`L_pred_valid` and `L_pred_box` above).
- `bev_map=false` and the target builder's BEV map output is commented out, so
  `bev_semantic_loss=0`.
- `sub_score_weight=0` disables the six-logit sub-score BCE.
- `inter_weight=0` disables the diversity term.

### 9.3 Score-head indexing caveat

`PDMScorer` returns nine columns in this order:

```text
[collision, drivable area, traffic light, driving direction,
 progress, TTC, lane keeping, history comfort, final PDM score]
```

`pred_logit` has only six columns, and `sub_score_loss` slices the last six
target columns. Thus the score head does not directly supervise the first
three PDM columns (collision, drivable area, traffic light). In the final
configuration the sub-score loss is disabled anyway, and only the final PDM
column trains the score-selection head. Collision/TTC geometry and area flags
are trained through the auxiliary heads described above; traffic-light
compliance has no direct loss term.

## 10. Optimizer and 4060 adaptation

The resolved final run used:

```yaml
batch_size: 32
num_workers: 2
prefetch_factor: 2
max_epochs: 20
accumulate_grad_batches: 1
precision: 16-mixed
accelerator: gpu
devices: 1
gradient_clip_val: 1.0
seed: 0
```

The dataset sizes recorded by the run were 85,109 training samples and 18,179
validation samples. `drop_last=true` gives 2,659 training batches per epoch
and 568 validation batches per epoch. The effective/global batch was therefore
32, with no gradient accumulation.

`DriveJEPAAgent.get_optimizers()` returns Adam with no scheduler and no weight
decay:

```text
head parameters:     lr = 1e-4
trainable backbone:  lr = 1e-5  (only if unfrozen)
```

For the 8 GB RTX 4060 profile, `DRIVE_JEPA_FREEZE_VJEPA=1` froze the pretrained
ViT-L backbone and excluded it from the optimizer. The trainable part was the
projector, ego projection, proposal/refinement modules, scorer, and active
auxiliary heads. `DRIVE_JEPA_FORCE_PYTORCH_DEFORMABLE=1` selected the
differentiable PyTorch deformable-attention fallback because the bundled MMCV
CUDA extension targets a different GPU architecture. Mixed precision and
gradient clipping were enabled to fit the model reliably.

## 11. Validation, checkpoint selection, and official-style evaluation

`AgentLightningModule.validation_step()` does not return a loss for this agent.
It forwards the selected trajectory (`predictions["trajectory"][:, None]`),
scores it with the local PDM cache, and logs `val/score_epoch` plus component
diagnostics. The checkpoint callback monitors `val/score_epoch` and saves the
best epoch.

For the completed run:

- Best checkpoint: `best_epoch=19.ckpt`
- Final internal validation score: approximately `0.864`
- Separate sequential NAVSIM v2 `navtest` score: `0.8732976692` (`87.33` on
  the 0-100 EPDMS convention)
- Scenarios: 12,146 successful, 0 failed

The last logged epoch-19 batch provides a useful arithmetic check of the
objective: `trajectory_loss=0.796`, `final_score_loss=0.289`,
`pred_ce_loss=0.0751`, `pred_l1_loss=3.020`, and `pred_area_loss=0.0299` give
`0.796 + 0.289 + 0.0751 + 0.1*3.020 + 2*0.0299 = 1.522`, matching the logged
`train/loss=1.520` up to display rounding. This confirms which terms were
actually contributing to the final checkpoint.

The sequential evaluator adds two-frame extended comfort by comparing
overlapping simulated states of consecutive original frames. At evaluation
time, `AbstractAgent.compute_trajectory()` adds `past_ego_simulated_states`,
the metric cache, and the observation interval to the feature dictionary; the
model's calibration path uses those values to adjust proposal scores before
the final argmax. This path is absent during ordinary training batches.

## 12. Contract-level audit findings

These are behaviorally significant facts for anyone trying to reproduce or
extend the run:

1. **The visual input is narrower than the apparent sensor setup.** Four
   current cameras are preprocessed for metadata, but only two front frames
   are fed to V-JEPA. The current four-view `camera_feature` tensor is dead in
   the present forward path.
2. **Only the latest ego row is used.** Four ego rows are built and stacked,
   but `DriveJEPAModel.forward()` selects `ego_status[:, -1]`. Temporal ego
   dynamics are therefore not encoded except indirectly through the two front
   images.
3. **The future window and target horizon differ.** Ten future frames are
   loaded, but only eight are converted into the supervised target trajectory.
4. **Refinement stages share parameters.** `ModuleList([shared_refiner] * 4)`
   performs four recurrent applications of one refiner, not four independent
   refiners.
5. **PDM scoring is non-differentiable supervision.** Proposals are detached
   before simulator/scorer execution. Gradients come from the human/anchor
   trajectory distances and BCE/L1 auxiliary heads, not through the simulator.
6. **The training score is not the final extended-comfort score.** Two-frame
   extended comfort is added by the sequential evaluator, so `val/score_epoch`
   and official-style navtest EPDMS are related but not identical metrics.
7. **Several declared features are dormant.** LiDAR, semantic-map targets,
   BEV-agent targets, sub-score BCE, and diversity regularization do not
   contribute to the final checkpoint under the resolved configuration.
8. **Anchor indices are per sample.** `scores_index` must be iterated per batch
   item; using one index array for all proposals would silently corrupt the
   multimodal trajectory supervision.
9. **The metric-cache metadata may contain foreign absolute paths.** The local
   sequential evaluator includes path localization so the cache resolves on
   this machine; a fresh installation must perform equivalent path mapping or
   regenerate metadata.
10. **The human trajectory target is currently float64 in practice.** The
    target builder does not cast the NumPy float64 poses to float32. A port that
    assumes float32 should cast deliberately and verify that mixed-precision
    loss behavior remains unchanged.

## 13. Minimal implementation checklist

An implementation claiming compatibility with this checkpoint should verify:

- Four history and ten future log rows are available for every sample.
- Front camera files for history indices 2 and 3 exist, plus current `B0/F0/L0/R0`
  files and calibration.
- `camera_feature_1/2` are `[3,256,512]` and ImageNet-normalized exactly once.
- `ego_status` is `[4,11]`, local-frame converted, and latest-row selection is
  preserved.
- Target trajectory is `[8,3]` at `0.5 s`; proposals are `[32,8,3]`.
- Anchor trajectories use `8192 x 40 x 3` source data and `[:,4::5]` slicing.
- PDM proposal simulation uses 40 poses at `0.1 s` and the local metric cache.
- Active loss weights are `(trajectory=1, final_score=1, agent_cls=1,
  agent_box=0.1, area=2)`; disabled terms remain disabled unless the model is
  intentionally changed.
- The frozen V-JEPA checkpoint and the 4060 PyTorch deformable fallback are
  selected in the same way as the completed run.

## 14. Concrete sample checked on this machine

One materialized window was checked directly from the local NAVSIM files. Its
current token was `52dab8bede98583e` in log
`2021.05.12.19.36.12_veh-35_00005_00204`, with map
`us-nv-las-vegas-strip`. The resulting feature and target shapes were:

```text
camera_feature    torch.float32 (4, 3, 448, 768)
img_shape         torch.float32 (4, 3)
lidar2img         torch.float32 (4, 4, 4)
camera_feature_1  torch.float32 (3, 256, 512)
camera_feature_2  torch.float32 (3, 256, 512)
ego_status        torch.float32 (4, 11)
trajectory        torch.float64 (8, 3)
token             str
```

The source frame records in this example contained `1920 x 1080` camera
images and annotation arrays with `N=119` objects. This is an observed
implementation sample, not a synthetic placeholder. It also demonstrates why
shape and dtype checks should be part of a reproduction preflight.

## 15. Evidence map

The following source locations are the compact audit trail behind this
contract:

| Concern | Source location |
| --- | --- |
| History/future window and relative poses | `navsim_v2/navsim/common/dataclasses.py:164-235`, `:341-372`, `:653-698` |
| Sensor schedule | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:76-88` |
| Feature keys and camera preprocessing | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_features.py:30-89`; `.../bevformer/bev_feature_build.py:16-82` |
| Target dictionary | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_features.py:135-164` |
| V-JEPA input, projector, proposal selection | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_model.py:24-96` |
| Shared refinement and proposal shape | `navsim_v2/navsim/agents/drive_jepa_perception_based/traj_refiner.py:8-29`; `.../bevformer/bev_refiner.py:107-142` |
| Score/auxiliary heads | `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/scorer.py:8-68` |
| PDM scoring and anchor indices | `navsim_v2/navsim/agents/drive_jepa_perception_based/score_module/compute_navsim_score.py:25-139` |
| Losses and active weights | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:141-378` |
| Adam parameter groups | `navsim_v2/navsim/agents/drive_jepa_perception_based/drive_jepa_agent.py:366-378` |
| DataLoader, drop-last, checkpoint monitor | `navsim_v2/navsim/planning/script/run_training.py:125-163` |
| Final 4060 resolved settings | `/media/dell/Storage/Drive-JEPA-runs/full-navtrain-4060-bs32/drive_jepa_navsim_v2_4060/code/hydra/config.yaml:120438-120481` |
| Sequential two-frame comfort evaluation | `navsim_v2/navsim/planning/script/run_pdm_score_sequential.py:135-184`, `:341-461` |
