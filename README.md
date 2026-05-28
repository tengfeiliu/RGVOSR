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
| `configs/accelerate/zero3_bf16_cpu_offload.yaml` | 2 卡 ZeRO-3 + CPU offload，用于 24GB 卡 smoke test。 |
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
  --limit 1 \
  --overwrite
```

`--limit 1` is useful for testing a single paid API sample before launching a full batch. The CLI prints progress
for each file, record, and LLM prompt stage so long-running requests show where they are waiting.

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

- Existing output files are not overwritten unless `--overwrite` is set.
- Missing `unipercept_raw.profile`, `iaa`, or `iqa` is logged and the record is kept unchanged.
- `profile.ista` is preserved from the original profile.
- JSON output uses `ensure_ascii=False`, so Chinese and other Unicode text are preserved.
