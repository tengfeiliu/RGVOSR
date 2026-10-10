# Observation constrained SR loop implementation plan

> **For agentic workers:** Use superpowers:executing-plans to implement task by task. The user approved implementation and execution on 2026-10-10.

**Goal:** Execute a larger paired experiment that tests spatial LR injection and a shared residual Transformer against the uploaded FLUX.2 F0.

**Architecture:** Freeze F0 and its condition8 MoE. Reuse verified first-round training images, generate independent heldout first rounds, compare matched weak/spatial FLUX adapters, then train shared visual correction arms with actual output supervision and recursive state. A resumable supervisor executes stages and records raw outputs, paired metrics and validation-only selections.

**Tech Stack:** Existing sr-flux2 environment, PyTorch BF16, existing Diffusers FLUX.2, PyIQA paired evaluation. No environment installation or driver changes.

**Spec:** /media/xd/data/code/RGVOSR/artifacts/multiround_diagnosis_20261010/optimization_plan.md

## Global Constraints

- F0 adapter SHA: bf8539cfd27dc5f43d9a202b6f68ab87cc2ae8c603ca37df2e0b19f909452fe0.
- Model: /media/xd/data/models/FLUX.2-klein-base-4B, BF16 backbone and FP32 VAE/text encoding/new optimizer weights.
- Training 1024 sources, validation 128, test 256; exclude all original 3000 training HR groups and previously inspected validation/test HR groups from new holdout.
- First round retains original personalized prompt/condition8. FLUX later rounds use cached 512x7680 generic prompt, previous SR as native image condition, original LR spatial branch.
- No IQA metric as training loss; paired Charbonnier, HR gradients and low-frequency reconstruction for correction; existing FM objective for matched FLUX adapter ablation.
- Never terminate other jobs. Resource admission, OOM microbatch reduction, bounded retries, immutable split and checkpoint provenance.
- Tiny data smoke tests do not constitute quality success. Final selection uses validation only; test remains excluded from model selection.

## Review Focus

- Cached images with a wrong producer, prompt, geometry or content hash must be rejected.
- Multiple crops of the same source HR must not cross training/validation/test.
- A zero-initialized spatial branch must receive a nonzero exit gradient and must later change actual output.
- Recursive training and evaluation must consume the same shared model and carry state within each image chain only.
- Failed, OOM or interrupted stages must resume without overwriting completed outputs or selecting on test metrics.

### Task 1: Independent data and verified reuse

**Files:** Create rl_sr/observation_loop_data.py and tests/test_observation_loop.py; add configs/rl_sr_observation_loop.yaml.

**Interfaces:** select_sources(rows, baseline_paths, excluded_groups, counts, seed) returns train/validation/test rows; prepare_run(config,run) verifies F0 and existing StateRecord identities and writes immutable split/provenance.

- [x] Test grouped exclusion and insufficient data, run to confirm failure, implement, rerun.
- [x] Run full prepare on 1024/128/256 and inspect reuse counts; missing holdout F0 paths remain explicitly pending.

### Task 2: Spatial FLUX injection and shared correction

**Files:** Create models/sr_observation_loop.py; extend tests/test_observation_loop.py.

**Interfaces:** SpatialLRResidual(hidden_width,latent_channels,attention_width,grid) returns residual image tokens with a single zero-initialized output projection. FluxSpatialHooks(actor,branch) injects into target image tokens after x_embedder, bypasses first round. ObservationLoop(current,original_lr,rounds,state=None,lr_mode='spatial',use_round=True,use_state=True) returns image/state/delta.

- [x] Test identity initialization, nonzero initial output gradient, actual LR/round influence after an optimizer step, arbitrary valid spatial sizes, per-image state and first-round bypass.
- [x] Implement normalized cross-attention and shared 96-width two-block Transformer with stride16 visual encoder, zero-initialized RGB residual head and explicit round/state controls.
- [x] Verify on CPU and real CUDA BF16 with a forward/backward batch and full512 output.

### Task 3: Resumable experiment stages

**Files:** Create tools/run_sr_observation_loop.py, tools/sr_observation_loop_flux.py, tools/sr_observation_loop_training.py.

**Interfaces:** Experiment(config,run) supplies sources/artist/metrics and stage dispatch; stages prepare,cache_text,f0,flux_weak,flux_spatial,correctors,report.

- [x] Test validation-only checkpoint ranking, checkpoint compatibility and actual multi-round output supervision.
- [x] Generate new heldout F0 only; cache generic prompt with existing signature and integrity checks.
- [x] Train weak/spatial FLUX branches with matched source/update schedules, frozen MoE and held routing for checkpoint backward; evaluate actual 25-NFE outputs.
- [x] Train C0 weak LR, C1 spatial, C2 round/state, C3 hidden-state unroll, and C4 periodic old-policy state refresh with 1200 updates each and validation checkpointing. Record update/sample/loop compute counts.
- [x] Evaluate all raw selected arms on fresh validation/test at full512, run LR/round/state ablations, produce paired intervals and inspection grids.

### Task 4: Durable execution and review

**Files:** Create tools/watch_sr_observation_loop.py and tools/start_sr_observation_loop_background.sh; update versioned experiment documentation.

- [x] Test stage failure/retry and OOM batch adaptation; implement process-owning supervisor with lock, source hashes, heartbeat and explicit failed/completed state.
- [x] Run focused existing/new tests, run real-GPU smoke and obtain fresh whole-branch review.
- [ ] Commit final reviewed fixes on RL-SR-observation-loop-v5, launch supervisor, verify live PID plus actual stage progress, and link log/status/report in the original workspace.

## Progress and decisions

Initial baseline: tests/test_round_state_quick.py 6 passed. CPU default Python has no YAML; use /home/xd/miniconda3/envs/sr-flux2/bin/python for every experiment/test. Larger sample counts are continuation holdouts, not certified heldout from F0 pretraining. Existing CUDA and NVML installations remain unchanged.

Final-review changes: checkpoint evaluation/refresh recovery; exact producer/tensor hashes; continuous RGB and BF16 hidden refresh; mixed rounds for C0–C2; immutable completed image chains; PID/start identity and stage lock. 34 focused tests pass. Actual CUDA injected phase recovery passed. Final background launch pending.

Ruling: add G_generic on all384 new heldout sources; V4 historical means use different sources and cannot serve as the fresh repeated-generation control. Runtime grows, predicted overall12–22h from measured F0 throughput, below next-evening target if resources remain available.
