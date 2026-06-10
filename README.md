# RG-FLUX-SR / VOSR Command Reference

这个 README 主要作为本地和服务器上的运行命令手册。当前重点是 `train_rg_flux_sr.py`、`inference_rg_flux_sr.py` 和 `eval_rg_flux_sr_metrics.py` 这条 RG-FLUX-SR 训练、推理、评测链路。

## 项目简介

RG-FLUX-SR 是在 FLUX.1-dev 上做超分适配的实验链路。训练时使用 FLUX VAE 编码 HQ/LQ-up 图像，在 latent space 中做 flow matching；FLUX transformer 作为主干，主要训练 LoRA 和退化/LR 条件 adapter。

目前推荐先用 2 卡 256 smoke 配置验证完整链路，再在更大显存或 8 卡环境下使用 512 正式配置训练。

## 环境与模型路径

安装依赖：

```bash
pip install -r requirements.txt
```

FLUX 模型目录需要是完整 Diffusers 格式，并包含 `transformer/`、`vae/`、text encoder、tokenizer 等子目录。当前配置默认路径：

```yaml
model:
  flux_model_path: /data/datasets/FLUX.1-dev
```

如果服务器路径不同，修改：

- `configs/train_rg_flux_sr_ms.yaml`
- `configs/train_rg_flux_sr_ms_smoke_256.yaml`
- `configs/train_rg_flux2_klein_sr_smoke_256.yaml`

FLUX.2-klein 后端需要单独的 Diffusers 格式基础模型目录。当前 FLUX2 smoke 配置默认使用：

```yaml
model:
  flux_backend: flux2_klein
  flux_model_path: /data/models/FLUX.2-klein-base-4B
```

该目录需要包含 FLUX.2-klein base 的 `transformer/`、`vae/`、Qwen tokenizer/text encoder 等完整组件。FLUX.2-klein 不是简单替换 FLUX.1 路径，必须显式设置 `model.flux_backend: flux2_klein`。

数据 JSONL 默认：

```yaml
data:
  jsonl_path: datasets/LSDIR_cache/valid.jsonl
```

每条 JSONL 需要包含 `hq_path`、`lq_path` 和 `result`。`result` 中的 reasoning、suggestions、degradation_vector 会用于 prompt 和退化条件。

## 配置文件说明

| 文件 | 用途 |
| --- | --- |
| `configs/train_rg_flux_sr_ms_smoke_256.yaml` | 2 卡低显存 smoke 配置，`crop_size=256`，用于验证训练链路。 |
| `configs/train_rg_flux_sr_ms.yaml` | 512 正式训练配置，推荐 8 卡或更大显存。 |
| `configs/train_rg_flux2_klein_sr_smoke_256.yaml` | FLUX.2-klein 2 卡 256 smoke 配置，默认 `model.flux_backend=flux2_klein`。 |
| `configs/accelerate/zero3_bf16_cpu_offload.yaml` | 2 卡 ZeRO-3 + CPU offload，用于 24GB 卡 smoke test。 |
| `configs/accelerate/zero3_bf16_param_offload.yaml` | 2 卡 ZeRO-3 配置，optimizer 不 offload；可按显存情况设置 `offload_param_device` 为 `cpu` 或 `none`。 |
| `configs/accelerate/zero3_bf16.yaml` | 8 卡 ZeRO-3，无 CPU offload，用于正式训练。 |

关键配置：

- `data.batch_size`：每卡 batch，不是 global batch。
- `training.grad_accum_steps`：梯度累积步数。
- 有效 batch：`batch_size * num_processes * grad_accum_steps`。
- `model.text_encoder_device: cpu`：text encoder 放 CPU，避免初始化 OOM。
- `model.vae_device: cpu`：VAE 放 CPU，继续降低显存压力。
- `evaluation.eval_every: 500`：训练中每 500 step 计算一次指标。

## RG-FLUX-SR 训练命令

### 2 卡 256 Smoke Dry-Run

这个命令用于验证 2x24GB 环境下 ZeRO-3 初始化、模型加载、forward、backward、optimizer step 和 checkpoint 保存是否能完整跑通。`--dry_run` 只跑 1 个优化 step，适合改代码或换环境后快速检查。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 2 \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux_sr_ms_smoke_256.yaml \
  --dry_run
```

### 2 卡 256 Smoke 正式训练

这个命令去掉了 `--dry_run`，会按 smoke 配置持续训练。它适合 2x24GB 上做流程验证、小规模实验或调试，不代表最终 512 正式训练效果。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 2 \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux_sr_ms_smoke_256.yaml
```

### 2 卡 FLUX.2-klein 256 Smoke 训练

这个命令用于启动 FLUX.2-klein base 后端的 256 smoke 训练。配置文件会走 `Flux2KleinSRArtist`，基础模型路径默认是 `/data/models/FLUX.2-klein-base-4B`。当前建议先用 2 卡 256 验证链路，后续再逐步放大 crop、token 数或迁移到更多显存。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16_param_offload.yaml \
  --num_processes 2 \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml
```

FLUX2 smoke 配置里的几个关键项：

```yaml
model:
  flux_backend: flux2_klein
  flux_model_path: /data/models/FLUX.2-klein-base-4B
  text_encoder_device: cpu
  vae_device: cuda

training:
  grad_accum_steps: 1
  resume_ckpt: null
  auto_resume: false
  suffix: "_flux2_klein_smoke256_v2"

evaluation:
  enabled: false
```

如果显存不足，优先把 `vae_device` 改回 `cpu`，并把 `configs/accelerate/zero3_bf16_param_offload.yaml` 中的 `offload_param_device` 改成 `cpu`。如果想恢复自动续训，把 `training.auto_resume` 改成 `true`，但 ZeRO-3 下旧 LoRA checkpoint 可能需要单独的安全加载逻辑。

### 8 卡 512 Dry-Run

这个命令用正式 512 配置先跑 1 个 step，适合在正式训练前检查 8 卡 ZeRO-3、模型路径、数据路径和显存是否正常。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16.yaml \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux_sr_ms.yaml \
  --dry_run
```

### 8 卡 512 正式训练

这是推荐的正式训练入口。512 crop 的 image token 数比 256 高 4 倍，显存压力明显更大，建议用 8 卡或更大显存环境。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16.yaml \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux_sr_ms.yaml
```

### 普通 DDP Dry-Run

这个命令主要用于调试 accelerate/DDP 行为，不推荐在 24GB 卡上做完整 512 训练。普通 DDP 会在每张卡上复制完整 FLUX transformer，显存压力比 ZeRO-3 更高。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --num_machines 1 \
  --num_processes 2 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux_sr_ms_smoke_256.yaml \
  --dry_run
```

## RG-FLUX-SR 推理命令

### 单张图片或文件夹推理

这个命令使用训练好的 LoRA/adapter checkpoint 对 LQ 图片或文件夹做超分。`--num_inference_steps` 控制 multi-step flow matching 的采样步数，默认建议 25。

```bash
python inference_rg_flux_sr.py \
  --input path/to/lq_or_folder \
  --output_dir outputs/rg_flux_sr \
  --checkpoint exp_rg_flux_sr/rg_flux_sr_ms_stageA_latent_adapter_size256_smoke256/checkpoints/checkpoint-00000001/rg_flux_adapters \
  --config configs/train_rg_flux_sr_ms_smoke_256.yaml \
  --jsonl_path datasets/LSDIR_cache/valid.jsonl \
  --num_inference_steps 25 \
  --upscale 4
```

### FLUX.2-klein Checkpoint 推理

FLUX.2-klein 推理仍使用 `inference_rg_flux_sr.py`，关键是 `--config` 指向 FLUX2 配置，`--checkpoint` 指向 FLUX2 实验保存的 `rg_flux_adapters` 目录。

```bash
python inference_rg_flux_sr.py \
  --input path/to/lq_or_folder \
  --output_dir outputs/rg_flux2_klein_sr \
  --checkpoint exp_rg_flux_sr/rg_flux2_klein_sr_ms_stageA_latent_adapter_size256_flux2_klein_smoke256_v2/checkpoints/checkpoint-00000001/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/LSDIR_cache/valid.jsonl \
  --num_inference_steps 25 \
  --upscale 4
```

参数说明：

- `--input`：输入 LQ 图片、文件夹，或 txt 列表。
- `--output_dir`：SR 图片输出目录。
- `--checkpoint`：训练保存的 adapter 目录，通常指向 `.../checkpoint-XXXXXXXX/rg_flux_adapters`。
- `--config`：训练时使用的配置。推理会用其中的模型路径、条件模式等设置。
- `--jsonl_path`：可选，用于读取 RG/VOSR 分析结果并构造 prompt 与 degradation vector。
- `--num_inference_steps`：flow matching 采样步数，越大通常越慢。
- `--upscale`：输入图先 bicubic 放大的倍率，默认超分倍率通常用 4。

## 指标评测命令

### 独立评测 SR 图片目录

这个命令对已经生成好的 SR 图片目录计算 OMGSR 同款 PyIQA no-reference 指标，并输出 per-image CSV 和 summary JSON。

```bash
python eval_rg_flux_sr_metrics.py \
  --dataset_dirs smoke=outputs/rg_flux_sr \
  --output_dir eval/rg_flux_sr_smoke \
  --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa
```

输出文件：

```text
eval/rg_flux_sr_smoke/
|-- per_image_scores.csv
|-- summary_scores.csv
`-- summary_scores.json
```

默认指标：

- `clipiqa`
- `clipiqa+`
- `nima`
- `niqe`
- `liqe`
- `musiq`
- `maniqa`

其中 `niqe` 是 lower better，其余通常是 higher better。

### 多个数据集一起评测

如果有多个 SR 输出目录，可以一次传入多个 `name=path`。

```bash
python eval_rg_flux_sr_metrics.py \
  --dataset_dirs smoke=outputs/smoke real=outputs/real \
  --output_dir eval/rg_flux_compare \
  --device cuda
```

### 训练期自动评测

训练脚本会读取 YAML 中的 `evaluation` 配置。默认每 500 step 运行一次评测，采样前 8 条 JSONL 记录生成 SR 图片，然后计算 PyIQA 指标。

```yaml
evaluation:
  enabled: true
  eval_every: 500
  num_samples: 8
  num_inference_steps: 25
  metrics: [clipiqa, clipiqa+, nima, niqe, liqe, musiq, maniqa]
  jsonl_path: null
  output_dir: eval
  device: cpu
```

训练期指标输出路径：

```text
eval/<exp_name>/step-XXXXXXXX/
|-- images/
`-- metrics/
    |-- per_image_scores.csv
    |-- summary_scores.csv
    `-- summary_scores.json
```

默认 `evaluation.device: cpu` 是为了避免 PyIQA 额外占用训练 GPU 显存。如果显存充足，可以改成 `cuda` 加速指标计算。

## 常用参数说明

### `--dry_run`

只跑 1 个优化 step，用于检查初始化、前向、反向、保存 checkpoint 是否正常。正式训练去掉即可。

### `crop_size`

训练 patch 大小。`256` 对应更低显存 smoke；`512` 是正式配置。由于 FLUX 会 pack latent tokens，512 的 image token 数约为 256 的 4 倍，显存压力明显更高。

### `lr_token_count`

LR latent adapter 产生的条件 token 数。token 越多，条件信息越丰富，但显存也更高。smoke 配置中是 16，正式配置中是 64。

### `num_inference_steps`

推理时 multi-step flow matching 的采样步数。常用值：

- `10`：更快，质量可能不稳定。
- `25`：默认推荐值。
- `50`：更慢，质量收益不一定线性。

### `resume_ckpt`

配置中可以显式指定 checkpoint：

```yaml
training:
  resume_ckpt: path/to/checkpoint-XXXXXXXX
```

当前训练脚本也会扫描实验目录下的最新 checkpoint。如果想完全从头开始，建议换一个新的 `training.suffix`，或清理对应实验目录中的旧 `checkpoints/`。

## 常见问题

### 1. 2 卡 24GB 能不能直接训练 512？

不推荐。512 crop + FLUX.1-dev 的 forward 显存压力很大。2 卡 24GB 建议先跑 `train_rg_flux_sr_ms_smoke_256.yaml` 验证链路；512 正式训练建议 8 卡或更大显存。

### 2. 为什么 text encoder 和 VAE 放 CPU？

FLUX.1-dev 本身很大。把 text encoder 和 VAE 放 GPU 会在初始化或 forward 前额外占用大量显存。当前训练只需要冻结的 text/VAE 编码结果，所以默认放 CPU 更稳。

### 3. ZeRO-3 下为什么禁用了 transformer gradient checkpointing？

Diffusers FLUX transformer 的 gradient checkpointing 和 DeepSpeed ZeRO-3 参数分片在 backward recompute 阶段会出现 metadata mismatch。当前默认在 ZeRO-3 下关闭 transformer gradient checkpointing，优先保证链路稳定。

### 4. `TOKENIZERS_PARALLELISM=false` 是必须的吗？

不是数学上必须，但建议保留。它可以减少 tokenizer 多进程并行相关 warning 和潜在卡顿。

### 5. 训练期指标很慢怎么办？

可以调小：

```yaml
evaluation:
  eval_every: 2000
  num_samples: 4
  device: cpu
```

也可以临时关闭：

```yaml
evaluation:
  enabled: false
```

## Legacy VOSR

原始 VOSR 多步/一步推理脚本仍在仓库中：

- `train_vosr.py`
- `train_vosr_distill.py`
- `inference_vosr.py`
- `inference_vosr_onestep.py`

当前 README 不再展开旧 VOSR 命令，主要维护 RG-FLUX-SR 实验链路。

## Profile Cleaner

`profile_cleaner` is a post-processing utility for UniPercept image understanding profiles. It cleans
`record.unipercept_raw.profile` in JSON or JSONL records while preserving every other record field.

### Install

```bash
pip install -r requirements.txt
```

The tool uses the existing `openai>=1.0.0` dependency and works with OpenAI-compatible chat completion APIs.

### Environment

```bash
export DASHSCOPE_API_KEY=...
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1  # optional default
export PROFILE_CLEANER_MODEL=qwen2.5-vl-72b-instruct                     # optional default
export PROFILE_CLEANER_TEMPERATURE=0                                      # optional
```

### Commands

Single JSON:

```bash
python -m profile_cleaner.cli --input input.json --output output.json --overwrite
```

JSONL batch:

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --output datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --jsonl \
  --model qwen2.5-vl-72b-instruct \
  --max-retries 2 \
  --limit 1
```

`--limit 1` is useful for testing a single paid API sample before launching a full batch. The CLI prints progress
for each file, record, and LLM prompt stage so long-running requests show where they are waiting.
If the JSONL output already exists, the CLI resumes by `hq_path`: records already present in the output are skipped,
and only missing records are cleaned and appended. Use `--overwrite` to force a full reclean and rewrite the output.

Directory batch:

```bash
python -m profile_cleaner.cli --input ./raw_profiles --output ./cleaned_profiles --recursive
```

Dry run validates structure and local IAA/IQA contamination without calling the model or writing output:

```bash
python -m profile_cleaner.cli --input input.jsonl --output output.jsonl --jsonl --dry-run --verbose
```

### Input And Output

The input record must contain a nested profile at:

```json
{
  "unipercept_raw": {
    "profile": {
      "iaa": {},
      "iqa": {},
      "ista": {}
    }
  }
}
```

Only `unipercept_raw.profile` is replaced. The cleaner does not change top-level fields, raw rewards, image paths,
degradation metadata, or `result`.

### IAA/IQA Boundary

IAA is limited to composition, framing, layout, balance, color harmony, mood, theme communication, originality,
artistic expression, viewer response, and overall gestalt.

IQA is limited to blur, sharpness, focus, resolution, pixelation, noise, compression artifacts, exposure problems,
detail loss, texture loss, fidelity, recognizability, and usability.

If model output still mixes these concepts after retries, the local fallback deletes contaminated bullet/sentence
items and fills empty fields with a short valid placeholder.

### Word Budgets And IQA Fallback

Prompt B asks the model to follow token budgets. The local cleaner uses an English word-count approximation instead
of character or byte counts:

- `profile.iaa`: no more than 50 words, summarized into `iaa.comprehensive`.
- `profile.iqa`: no more than 350 words, preserving the required IQA fields when possible.
- `profile.suggestion`: no more than 80 words.

By default, if `distortion_location`, `distortion_severity`, `distortion_type`, or `overall_quality` is empty after
LLM cleanup, the cleaner fills it from the original profile. To inspect whether the LLM itself returned all required
fields correctly, disable that fallback:

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --output datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --jsonl \
  --no-required-iqa-fallback \
  --overwrite
```

### Error Log

Single-record failures do not stop a batch. Errors are written as JSONL with:

```json
{
  "input_file": "input.jsonl",
  "item_index": 0,
  "error": "...",
  "profile_summary": {}
}
```

Use `--error-log path/to/errors.jsonl` to choose the log path.

### FAQ

- JSONL output resumes by `hq_path` when the output file already exists; use `--overwrite` to force a full reclean.
- Existing non-JSONL output files are not overwritten unless `--overwrite` is set.
- Missing `unipercept_raw.profile`, `iaa`, or `iqa` is logged and the record is kept unchanged.
- `profile.ista` is preserved from the original profile.
- JSON output uses `ensure_ascii=False`, so Chinese and other Unicode text are preserved.

## UniPercept Profile 生成与清洗流程

这一节记录从原始 HQ 图像生成 `unipercept_raw.profile`，再用 `profile_cleaner` 优化 IAA/IQA/profile suggestion 的推荐命令。生成阶段使用本地 UniPercept 权重，不会从 Hugging Face 下载模型。

### 1. 生成 UniPercept Raw Profile Cache

默认输入、输出和模型路径在 `tools/generate_unipercept_raw_cache.py` 中已经配置：

```text
--input configs/train_txt/train_dataset_txt.txt
--lq-output-dir datasets/LSDIR_unipercept_lq
--output datasets/LSDIR_unipercept_raw_cache/valid.jsonl
--invalid-output datasets/LSDIR_unipercept_raw_cache/invalid.jsonl
--unipercept-repo /data/code/UniPercept/
--unipercept-model-path /data/models/UniPercept/
--unipercept-backend profile
```

先只跑 1 条样本确认环境、模型路径和输出结构：

```bash
python tools/generate_unipercept_raw_cache.py \
  --input configs/train_txt/train_dataset_txt.txt \
  --lq-output-dir datasets/LSDIR_unipercept_lq \
  --output datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --invalid-output datasets/LSDIR_unipercept_raw_cache/invalid.jsonl \
  --unipercept-repo /data/code/UniPercept/ \
  --unipercept-model-path /data/models/UniPercept/ \
  --unipercept-backend profile \
  --device auto \
  --limit 1 \
  --resume
```

确认无误后去掉 `--limit 1` 跑完整数据：

```bash
python tools/generate_unipercept_raw_cache.py \
  --input configs/train_txt/train_dataset_txt.txt \
  --lq-output-dir datasets/LSDIR_unipercept_lq \
  --output datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --invalid-output datasets/LSDIR_unipercept_raw_cache/invalid.jsonl \
  --unipercept-repo /data/code/UniPercept/ \
  --unipercept-model-path /data/models/UniPercept/ \
  --unipercept-backend profile \
  --device auto \
  --resume
```

常用参数：

- `--unipercept-model-path`：本地 UniPercept 模型目录，必须存在；不会自动下载 HF 模型。
- `--unipercept-repo`：本地 UniPercept 仓库路径，`profile` / `conversation` backend 需要。
- `--unipercept-backend profile`：推荐默认值，组合 reward 分数和 per-aspect conversation profile。
- `--unipercept-backend reward`：只使用 `unipercept-reward` inferencer。
- `--unipercept-backend command`：使用自定义命令模板，需提供 `--unipercept-command`。
- `--limit N`：只处理前 N 条，适合调试。
- `--resume`：跳过已经写入 valid/invalid JSONL 的 HQ 路径。

生成后的每条 JSONL 记录会包含 `hq_path`、`lq_path`、`raw_degradation_params`、`unipercept_raw` 和 `result`。其中待清洗的 profile 位于：

```json
{
  "unipercept_raw": {
    "profile": {
      "iaa": {},
      "iqa": {},
      "ista": {}
    }
  }
}
```

### 2. 生成推理测试集 UniPercept Raw Profile

这个命令用于已经存在的 LR 推理测试集。它不会再调用 RealESRGAN 二次退化，而是直接把测试集 LR 图片作为 `lq_path` 做 UniPercept 分析。DRealSR 和 RealSR 有同名 HR 图片，所以会从 `test_HR` 自动匹配真实 `hq_path`；RealLR200 和 RealLQ250 没有 GT 时会写入 `hq_path == lq_path` 和 `has_gt: false`。

先用 `--limit 2` 跑小样本，确认路径、模型和输出结构正常：

```bash
python tools/generate_unipercept_raw_cache.py \
  --inference-lr-mode \
  --dataset-dirs \
    dreal=/data/datasets/omgsr_eval/DrealSR_CenterCrop-20260428T063453Z-3-001/DrealSR_CenterCrop/test_LR \
    realsr=/data/datasets/omgsr_eval/RealSR_CenterCrop-20260428T063513Z-3-001/RealSR_CenterCrop/test_LR \
    reallr200=/data/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
    reallq250=/data/datasets/omgsr_eval/RealLQ250/lq \
  --hq-dirs \
    dreal=/data/datasets/omgsr_eval/DrealSR_CenterCrop-20260428T063453Z-3-001/DrealSR_CenterCrop/test_HR \
    realsr=/data/datasets/omgsr_eval/RealSR_CenterCrop-20260428T063513Z-3-001/RealSR_CenterCrop/test_HR \
  --output datasets/inference_unipercept_raw.jsonl \
  --invalid-output datasets/inference_unipercept_invalid.jsonl \
  --unipercept-repo /data/code/UniPercept/ \
  --unipercept-model-path /data/models/UniPercept/ \
  --unipercept-backend profile \
  --device cuda \
  --limit 2 \
  --resume
```

确认无误后去掉 `--limit 2` 跑完整四个推理测试集：

```bash
python tools/generate_unipercept_raw_cache.py \
  --inference-lr-mode \
  --dataset-dirs \
    dreal=/data/datasets/omgsr_eval/DrealSR_CenterCrop-20260428T063453Z-3-001/DrealSR_CenterCrop/test_LR \
    realsr=/data/datasets/omgsr_eval/RealSR_CenterCrop-20260428T063513Z-3-001/RealSR_CenterCrop/test_LR \
    reallr200=/data/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
    reallq250=/data/datasets/omgsr_eval/RealLQ250/lq \
  --hq-dirs \
    dreal=/data/datasets/omgsr_eval/DrealSR_CenterCrop-20260428T063453Z-3-001/DrealSR_CenterCrop/test_HR \
    realsr=/data/datasets/omgsr_eval/RealSR_CenterCrop-20260428T063513Z-3-001/RealSR_CenterCrop/test_HR \
  --output datasets/inference_unipercept_raw.jsonl \
  --invalid-output datasets/inference_unipercept_invalid.jsonl \
  --unipercept-repo /data/code/UniPercept/ \
  --unipercept-model-path /data/models/UniPercept/ \
  --unipercept-backend profile \
  --device cuda \
  --resume
```

然后清洗成推理可直接读取的 JSONL：

```bash
python -m profile_cleaner.cli \
  --input datasets/inference_unipercept_raw.jsonl \
  --output datasets/inference_cleaned.jsonl \
  --jsonl \
  --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --verbose
```

清洗完成后，推理命令中的 `--jsonl_path` 指向 `datasets/inference_cleaned.jsonl`。例如对 DRealSR 跑 FLUX.2-klein 推理：

```bash
python inference_rg_flux_sr.py \
  --input /data/datasets/omgsr_eval/DrealSR_CenterCrop-20260428T063453Z-3-001/DrealSR_CenterCrop/test_LR \
  --output_dir outputs/rg_flux2_klein_dreal \
  --checkpoint path/to/checkpoint/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/inference_cleaned.jsonl \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16
```

推理测试集输出字段说明：
- `dataset_name`：数据集名称，例如 `dreal`、`realsr`、`reallr200`、`reallq250`。
- `lq_path`：原始 LR 测试图片路径，也是推理输入匹配的主要 key。
- `hq_path`：DRealSR/RealSR 为同名 HR 图片；无 GT 数据集则等于 `lq_path`。
- `has_gt`：是否存在真实 HR/GT，后续有参考指标只应对 `true` 的样本计算。
- `raw_degradation_params.degradation_generated: false`：表示该记录没有生成合成退化，只分析已有 LR 图。

### 3. 配置千问/OpenAI-Compatible API

`profile_cleaner` 使用 OpenAI-compatible Chat Completions 接口。当前默认指向阿里云 DashScope 兼容接口：

```bash
export DASHSCOPE_API_KEY=你的千问APIKey
export OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export PROFILE_CLEANER_MODEL=qwen2.5-vl-72b-instruct
export PROFILE_CLEANER_TEMPERATURE=0
```

也可以在命令行显式传入：

```bash
--api-key sk-xxx
--base-url https://dashscope.aliyuncs.com/compatible-mode/v1
--model qwen-plus
--temperature 0
```

### 4. 优化 Profile Cleaner 输出

`profile_cleaner` 只替换 `record.unipercept_raw.profile`，不会修改 `hq_path`、`lq_path`、`raw_degradation_params`、`unipercept_raw.raw_reward`、外层 `result` 等字段。JSONL 模式会完成一条写一条，长任务中断时更容易保留已完成输出。

先用 `--limit 1` 测一条付费 API 样本：

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --output datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --jsonl \
  --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --limit 1 \
  --verbose
```

确认输出后跑完整清洗：

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --output datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --jsonl \
  --model qwen-plus \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --verbose
```

如果 `valid.cleaned.jsonl` 已经存在，JSONL 模式会根据 `hq_path` 自动续跑：输出中已有的记录会跳过，只清洗并追加缺失记录。需要强制重新清洗并重写输出时再加 `--overwrite`。

目录批处理：

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache \
  --output datasets/LSDIR_unipercept_cleaned_cache \
  --recursive \
  --jsonl \
  --model qwen-plus
```

不调用大模型、只做结构和本地禁词检查：

```bash
python -m profile_cleaner.cli \
  --input datasets/LSDIR_unipercept_raw_cache/valid.jsonl \
  --output datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --jsonl \
  --dry-run \
  --verbose
```

当前 Prompt B 约束：

- `profile.iaa`：不超过 50 tokens；本地 cleaner 使用 50 words 近似控制。
- `profile.iqa`：约 350 tokens；本地 cleaner 使用 350 words 近似控制，并尽量保留四个 IQA 必填字段。
- `profile.suggestion`：不超过 80 tokens；本地 cleaner 使用 80 words 近似控制。
- `profile.ista`：保留原始结构和内容。

常用参数：

- `--jsonl`：按 JSONL 读取和写出。
- `--limit 1`：只清洗 1 条，适合测试 API key、模型名和费用。
- `--overwrite`：强制覆盖输出并重新清洗；JSONL 默认会按已有输出中的 `hq_path` 自动续跑。
- `--error-log profile_cleaner_errors.jsonl`：单条失败记录写入 JSONL，不中断批处理。
- `--max-retries`：当前单 Prompt B 流程下影响有限，主要保留兼容参数。
- `--verbose`：输出文件、记录和 LLM 阶段进度。
- `--no-required-iqa-fallback`：关闭四个 IQA 必填字段的原始 profile 兜底，便于检查 LLM 原始返回是否合规。
