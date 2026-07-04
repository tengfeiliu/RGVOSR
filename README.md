# RG-FLUX-SR / VOSR Command Reference

## Run-contained Pipeline 常用命令

推荐新实验不传 `--inference_output_root`，这样训练、推理、评估和 bad case 会统一保存在同一个 run 目录下：

```text
exp_rg_flux_sr/<run_name>/
|-- checkpoints/
|-- inference/checkpoint-00020000/
|-- metrics/checkpoint-00020000/
|-- bad_cases/checkpoint-00020000/
|-- pipeline_manifest.json
`-- run_summary.json
```

如果仍想使用旧的外部推理目录，可以显式传入 `--inference_output_root eval/inference/<name>`。

### Single-LoRA：从训练到 bad case 一键完成

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python tools/run_rg_flux_pipeline.py \
  --train_config configs/train_rg_flux2_klein_sr_stage0b_512.yaml \
  --accelerate_config configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 1 \
  --checkpoint_steps 20000 30000 40000 \
  --dataset_dirs \
    RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda \
  --run_bad_cases \
  --bad_case_metrics clipiqa maniqa musiq \
  --bad_case_mode separate \
  --bad_case_worst_k 50
```

### 已训练好的 Single-LoRA：从推理到 bad case

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python tools/run_rg_flux_pipeline.py \
  --skip_train \
  --run_dir exp_rg_flux_sr/<single_lora_run_name> \
  --checkpoint_steps 20000 30000 40000 \
  --dataset_dirs \
    RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda \
  --run_bad_cases \
  --bad_case_metrics clipiqa maniqa musiq \
  --bad_case_mode separate \
  --bad_case_worst_k 50
```

如果旧 run 目录下没有 `pipeline_runtime_config.yaml`，补跑时额外传入：

```bash
--train_config configs/train_rg_flux2_klein_sr_stage0b_512.yaml
```

### MoE-LoRA：Stage1 初始化、MoE 训练到 bad case 一键完成

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python tools/run_rg_flux_moe_pipeline.py \
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml \
  --single_lora_run_dir exp_rg_flux_sr/<single_lora_run_name> \
  --single_lora_checkpoint_step 32000 \
  --accelerate_config configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 1 \
  --checkpoint_steps 20000 30000 40000 \
  --dataset_dirs \
    RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --prototype_num_samples 128 \
  --init_device cuda \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda \
  --run_bad_cases \
  --bad_case_metrics clipiqa maniqa musiq \
  --bad_case_mode separate \
  --bad_case_worst_k 50
```

### 已训练好的 MoE-LoRA：从推理到 bad case

```bash
CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false python tools/run_rg_flux_moe_pipeline.py \
  --skip_stage1 \
  --skip_train \
  --moe_run_dir exp_rg_flux_sr/<moe_run_name> \
  --checkpoint_steps 20000 30000 40000 latest \
  --dataset_dirs \
    RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda \
  --run_bad_cases \
  --bad_case_metrics clipiqa maniqa musiq \
  --bad_case_mode separate \
  --bad_case_worst_k 50
```

`--bad_case_mode separate` 会分别输出 `clipiqa/maniqa/musiq` 三组 bad cases；如果想看多指标联合排序，可以改成：

```bash
--bad_case_mode joint_mean
```

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
| `configs/train_rg_flux2_klein_sr_smoke_256.yaml` | FLUX.2-klein Single-LoRA 实验配置，默认使用原生 `flux2_image_concat` LR 图像条件。实际 crop 大小以 YAML 中的 `data.crop_size` 为准。 |
| `configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml` | FLUX.2-klein LoRA-MoE 实验配置，默认使用原生 `flux2_image_concat` LR 图像条件和 `model.lora_backend=moe`。 |
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

### FLUX.2-klein Single-LoRA 训练

这个命令用于启动 FLUX.2-klein base 后端的 Single-LoRA 训练。配置文件会走 `Flux2KleinSRArtist`，基础模型路径默认是 `/data/models/FLUX.2-klein-base-4B`，LR 条件默认使用 `flux2_image_concat`。配置文件名保留了早期的 `smoke_256` 命名，实际训练尺寸以 YAML 中的 `data.crop_size` 为准。

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

condition:
  lr_cond_mode: flux2_image_concat

training:
  grad_accum_steps: 8
  resume_ckpt: null
  auto_resume: false

evaluation:
  enabled: false
```

如果显存不足，优先把 `vae_device` 改回 `cpu`，并把 `configs/accelerate/zero3_bf16_param_offload.yaml` 中的 `offload_param_device` 改成 `cpu`。如果想恢复自动续训，把 `training.auto_resume` 改成 `true`，但 ZeRO-3 下旧 LoRA checkpoint 可能需要单独的安全加载逻辑。

#### FLUX.2 原生 LR 图像条件：`flux2_image_concat`

FLUX.2 Single-LoRA 和 LoRA-MoE 配置默认使用：

```yaml
condition:
  lr_cond_mode: flux2_image_concat
```

该模式复用 FLUX.2-Klein 预训练的图像条件布局：

- LR 图像通过 VAE posterior `mode()` 进行确定性编码。
- patchify 和 latent normalization 后的 LR latent 直接作为 image token，不经过 `concat_proj`、一维池化或额外 type embedding。
- Transformer 输入顺序为 `[生成 token, LR 条件 token]`。
- 生成 token 使用二维位置 ID `(T=0, H, W, L=0)`，LR 条件 token 使用 `(T=10, H, W, L=0)`。
- Transformer 输出只保留前面的生成 token，LR 条件 token 的输出不会进入 FM loss。

在该模式下，LR token 数由输入尺寸和 FLUX.2 VAE/patchify 过程自动决定。例如当前 512 crop 会产生 `32 × 32 = 1024` 个 LR 条件 token。配置中的 `condition.lr_token_count` 不参与 `flux2_image_concat`，它只对旧的 `latent_adapter` 和 `latent_concat` 模式生效。

旧模式仍然保留：

```yaml
condition:
  lr_cond_mode: latent_adapter  # LR token 加入 text/context 序列
  # 或
  lr_cond_mode: latent_concat   # 旧版投影后 image-token concat
```

旧 `latent_concat` checkpoint 是在不同的 token 顺序、投影方式和位置 ID 下训练的。虽然代码仍可加载这类 checkpoint，但不建议直接用于 `flux2_image_concat` 正式推理或继续训练；建议从 FLUX.2 base 重新训练 Single-LoRA，再据此初始化 LoRA-MoE。

### FLUX.2-klein LoRA-MoE 训练流程

LoRA-MoE 是 FLUX.2-klein 的可选后端。默认 single-LoRA 路径不变，只有配置中显式设置 `model.lora_backend: moe` 时才启用。MoE 版本会保持 FLUX.2 transformer 主干冻结，只训练 shared LoRA、routed LoRA experts、router 和现有 condition adapters。

#### Stage 0：训练 Single-LoRA Baseline

先用当前 FLUX.2-klein Single-LoRA 配置和相同的 `flux2_image_concat` 条件模式训练一个稳定 baseline。这个 checkpoint 后续用于初始化 shared LoRA 和 routed experts。

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16_param_offload.yaml \
  --num_processes 2 \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml
```

Stage 0 输出的 adapter 目录通常类似：

```text
exp_rg_flux_sr/<single_lora_exp>/checkpoints/checkpoint-XXXXXXXX/rg_flux_adapters
```

#### Stage 1：从 Single-LoRA 初始化 LoRA-MoE

这个命令会把 Stage 0 的 single-LoRA 权重复制到 shared LoRA，并用 `baseline + small perturbation` 初始化 routed experts。它还会从 JSONL 中抽样，按当前 `latent_branch` 提取 router feature，并用 K-means 初始化 prototypes。

```bash
python tools/init_flux2_lora_moe.py \
  --config configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml \
  --single_lora_checkpoint exp_rg_flux_sr/<single_lora_exp>/checkpoints/checkpoint-XXXXXXXX/rg_flux_adapters \
  --output exp_rg_flux_sr/init_flux2_moe/rg_flux_adapters \
  --prototype_num_samples 128 \
  --device cuda \
  --dtype bf16
```

输出目录中会包含：

```text
exp_rg_flux_sr/init_flux2_moe/rg_flux_adapters/
|-- rg_flux_checkpoint_meta.json
|-- flux2_klein_lora_moe_state.pt
`-- condition_adapters.pt
```

#### Stage 2/3：启动 LoRA-MoE 训练

MoE 训练使用 `configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml`。训练时会自动按进度切换：

- 前 `warmup_ratio` steps：full softmax routing，shared LoRA 可冻结，temperature 从 `2.0` 逐步到 `1.5`。
- warm-up 之后：Top-2 routing，shared LoRA 解冻，temperature 逐步降到 `0.7`。

如果使用 Stage 1 生成的 MoE checkpoint，可以把配置里的 `training.resume_ckpt` 改成 `exp_rg_flux_sr/init_flux2_moe`，或在命令前临时改 YAML。然后运行：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
TOKENIZERS_PARALLELISM=false \
accelerate launch \
  --config_file configs/accelerate/zero3_bf16_param_offload.yaml \
  --num_processes 2 \
  train_rg_flux_sr.py \
  --config configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml
```

也可以不使用 Stage 1 工具，而是在 MoE 配置里直接设置：

```yaml
model:
  lora_moe:
    init_from_single_lora: exp_rg_flux_sr/<single_lora_exp>/checkpoints/checkpoint-XXXXXXXX/rg_flux_adapters
```

这种方式会在训练启动时从 single-LoRA 初始化 MoE，但不会预先 K-means 初始化 prototypes。正式实验更推荐先跑 Stage 1 工具。

#### LoRA-MoE 关键配置

```yaml
model:
  lora_backend: moe
  lora_moe:
    num_routed_experts: 4
    top_k: 2
    latent_branch: stat_conv
    router_hidden_dim: 1024
    warmup_ratio: 0.1
    init_temperature: 2.0
    final_temperature: 0.7
    shared_always_active: true
    freeze_shared_during_warmup: true

loss:
  router_div_weight: 1.0e-3
  router_entropy_weight: 1.0e-4
  router_balance_weight: 1.0e-4
```

训练日志会额外记录 `loss_div`、`loss_entropy`、`loss_balance`、`router/temperature`、`router/entropy` 和每个 expert 的 usage/top1 占比。

#### LoRA-MoE 一键 Stage1→训练→推理→评估

如果 Stage0 Single-LoRA 已经训练完成，可以用 `tools/run_rg_flux_moe_pipeline.py` 一次完成 Stage1 MoE 初始化、Stage2/3 MoE 训练、多个 checkpoint 的多数据集推理和评估：

```bash
python tools/run_rg_flux_moe_pipeline.py \
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml \
  --single_lora_run_dir exp_rg_flux_sr/<single_lora_run> \
  --single_lora_checkpoint_step 32000 \
  --accelerate_config configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 1 \
  --checkpoint_steps 20000 40000 latest \
  --dataset_dirs \
    realLQ250=/data/datasets/omgsr_eval/RealLQ250/lq \
    realLR200=/data/datasets/omgsr_eval/RealLR200 \
  --inference_output_root eval/inference/rg_flux2_moe_cached_fixed_prompt \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda
```

也可以直接指定 Stage0 adapter：

```bash
python tools/run_rg_flux_moe_pipeline.py \
  --moe_config configs/train_rg_flux2_klein_sr_moe_stage0b_512.yaml \
  --single_lora_checkpoint exp_rg_flux_sr/<single_lora_run>/checkpoints/checkpoint-00032000/rg_flux_adapters \
  --checkpoint_steps 20000 40000 \
  --dataset_dirs realLQ250=/path/lq realLR200=/path/lq \
  --inference_output_root eval/inference/rg_flux2_moe_cached_fixed_prompt
```

脚本会在 MoE run 目录下生成：

```text
exp_rg_flux_sr/<moe_run>/
|-- pipeline_runtime_config.yaml
|-- stage1_init/
|   `-- rg_flux_adapters/
|-- checkpoints/
|-- moe_pipeline_manifest.json
`-- logs/
```

`pipeline_runtime_config.yaml` 会自动设置 `training.resume_ckpt` 指向 `stage1_init/rg_flux_adapters`，并设置 `training.resume_training_state: false`，因此 Stage2/3 训练会从 Stage1 初始化的 MoE adapter 加载模型权重，但 optimizer/scheduler 从头开始。

已有 MoE run 只补跑推理和评估时：

```bash
python tools/run_rg_flux_moe_pipeline.py \
  --skip_stage1 \
  --skip_train \
  --moe_run_dir exp_rg_flux_sr/<moe_run> \
  --checkpoint_steps 20000 40000 latest \
  --dataset_dirs realLQ250=/path/lq realLR200=/path/lq \
  --inference_output_root eval/inference/rg_flux2_moe_cached_fixed_prompt \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda
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

## Text Embedding 离线缓存

离线缓存会预先生成 `prompt_embeds`、`pooled_prompt_embeds` 和 `text_ids`。训练或推理设置为 `cached` 后不会加载 text encoder、tokenizer 或 text pipeline，可减少 CPU/GPU 内存占用。

`condition.use_prompt` 和 `condition.use_suggestions` 会改变实际 prompt，并且属于 text embedding cache signature 的一部分。生成缓存、训练和推理时三者必须保持一致；修改这两个参数后，需要使用新的缓存目录或重新生成缓存。

### 场景一：`use_prompt=true`、`use_suggestions=true`

该模式会根据每张图片的 IQA、IAA 和 restoration suggestion 构造不同 prompt，因此需要遍历完整 JSONL，为各图片生成对应的 text embedding。

训练配置：

```yaml
condition:
  use_prompt: true
  use_suggestions: true

text_encoding:
  mode: cached
  cache_dir: datasets/text_embed_cache/flux2_klein_train_full_prompt
  strict: true
  dtype: bf16
  validate_prompt_hash: true
```

生成训练集缓存：

```bash
python tools/cache_rg_flux_text_embeddings.py \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --output_dir datasets/text_embed_cache/flux2_klein_train_full_prompt \
  --device cuda \
  --dtype bf16 \
  --resume \
  --skip-existing
```

推理集通常包含不同图片和不同 prompt，需要针对推理 JSONL 单独生成缓存：

```bash
python tools/cache_rg_flux_text_embeddings.py \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/inference_cleaned.jsonl \
  --output_dir datasets/text_embed_cache/flux2_klein_inference_full_prompt \
  --device cuda \
  --dtype bf16 \
  --resume \
  --skip-existing
```

推理命令：

```bash
python inference_rg_flux_sr.py \
  --input path/to/lq_or_folder \
  --output_dir outputs/rg_flux2_cached_full_prompt \
  --checkpoint path/to/checkpoint/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/inference_cleaned.jsonl \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_inference_full_prompt \
  --use_prompt \
  --use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16
```

### 场景二：`use_prompt=false`、`use_suggestions=false`

该模式忽略每张图片的 profile，所有样本统一使用 `models/prompt_builder.py` 中的 `DEFAULT_SR_PROMPT`。由于 prompt 和 encoder signature 完全相同，只需要在线编码一条合法 JSONL 记录，训练、推理、Single-LoRA 和 LoRA-MoE 都可以复用同一个 embedding 文件。

训练配置：

```yaml
condition:
  use_prompt: false
  use_suggestions: false

text_encoding:
  mode: cached
  cache_dir: datasets/text_embed_cache/flux2_klein_fixed_sr_prompt
  strict: true
  dtype: bf16
  validate_prompt_hash: true
```

使用 `--limit 1` 只生成一次固定 prompt embedding：

```bash
python tools/cache_rg_flux_text_embeddings.py \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/LSDIR_unipercept_raw_cache/valid.cleaned.jsonl \
  --output_dir datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --device cuda \
  --dtype bf16 \
  --limit 1 \
  --overwrite
```

缓存 resolver 会先按图片路径查询，未命中时再按 `prompt_hash + encoder_signature` 查询，因此其他训练或推理图片可以读取同一份固定 prompt embedding，不需要逐图生成或逐图注册 manifest。Single-LoRA 与 LoRA-MoE 配置只要基础模型、文本长度、文本 dtype 和上述两个 condition 开关一致，也可以指向同一个缓存目录。

独立推理脚本的 `--use_prompt` 和 `--use_suggestions` 默认值均为 `true`，会覆盖 YAML，所以固定 prompt 模式必须显式传入两个否定参数：

```bash
python inference_rg_flux_sr.py \
  --input path/to/lq_or_folder \
  --output_dir outputs/rg_flux2_cached_fixed_prompt \
  --checkpoint path/to/checkpoint/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16
```

如果推理仍需要读取 JSONL 中的 degradation vector，可以继续传入 `--jsonl_path datasets/inference_cleaned.jsonl`；它不会改变固定的 `DEFAULT_SR_PROMPT`。

强制全部重新生成时使用 `--overwrite`。逐图 prompt 模式建议先加 `--limit 2` 做小样本检查。`online` 保持在线编码；`auto` 优先读取缓存，未命中时在线编码，因此仍会加载 text encoder。`cached` 模式下如果 prompt hash、encoder signature 或 embedding 文件不匹配，程序会直接报错，不会静默使用错误缓存。

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
  --checkpoint exp_rg_flux_sr/rg_flux2_klein_sr_ms_stageA_flux2_image_concat_size512_flux2_klein_smoke256_v0621/checkpoints/checkpoint-XXXXXXXX/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --jsonl_path datasets/LSDIR_cache/valid.jsonl \
  --lr_cond_mode flux2_image_concat \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4
```

通常无需显式传入 `--lr_cond_mode`，因为配置文件已经设置为 `flux2_image_concat`；这里保留该参数是为了强调推理条件模式必须与训练一致。当前配置的 `use_prompt=false`、`use_suggestions=false` 会被独立推理 CLI 的默认值覆盖，因此必须显式传入两个 `--no-*` 参数。

#### FLUX.2-klein 推理 dtype 说明

FLUX.2-klein 推理阶段以 YAML 中的 `model.dtype` 作为 FLUX transformer / LoRA adapter 的主 dtype。`--dtype` 仍保留作为 CLI 参数，但如果它和 `model.dtype` 不一致，脚本会打印 warning，并使用 `model.dtype`，避免输入 hidden states、base transformer 权重、MoE LoRA 权重出现 `fp32` / `bf16` 混用导致的 linear dtype mismatch。

当前推荐保持配置中的 `model.dtype: bf16`，推理命令也使用 `--dtype bf16`。MoE router 的统计和 gating 可以保留 `fp32`，但进入 transformer 与 LoRA linear 的张量和 adapter 参数会在 checkpoint 加载后对齐到推理主 dtype。

#### FLUX.2-klein 多数据集批量推理

`inference_rg_flux_sr.py` 也支持一次加载模型后连续推理多个 LR 文件夹。多数据集模式使用 `--dataset_dirs name=folder_path ...`，并且 `--output_dir` 表示总输出根目录；脚本会自动在下面创建每个数据集自己的子目录。

```bash
python inference_rg_flux_sr.py \
  --dataset_dirs \
    realLQ250=/data/datasets/omgsr_eval/RealLQ250/lq \
    realLR200=/data/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --run_dir exp_rg_flux_sr/rg_flux2_klein_sr_ms_stageA_flux2_image_concat_size512_flux2_klein_smoke256_v0621_26062810 \
  --checkpoint_step 32000 \
  --output_root eval/inference/rg_flux2_cached_fixed_prompt \
  --config configs/train_rg_flux2_klein_sr_smoke_256.yaml \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16
```

输出目录会是：

```text
eval/inference/rg_flux2_cached_fixed_prompt/
`-- rg_flux2_klein_sr_ms_stageA_flux2_image_concat_size512_flux2_klein_smoke256_v0621_26062810/
    `-- checkpoint-00032000/
        |-- realLQ250/
        |   |-- *.png
        |   `-- inference_failures.jsonl
        |-- realLR200/
        |   |-- *.png
        |   `-- inference_failures.jsonl
        `-- inference_manifest.json
```

推荐使用 `--run_dir + --checkpoint_step + --output_root`，这样只需要在一个地方修改 checkpoint step。脚本会自动推导 `checkpoint-00032000/rg_flux_adapters` 和输出目录，并写出 `inference_manifest.json`。后续指标评估可以直接读取该 manifest：

```bash
python eval_rg_flux_sr_metrics.py \
  --inference_manifest eval/inference/rg_flux2_cached_fixed_prompt/rg_flux2_klein_sr_ms_stageA_flux2_image_concat_size512_flux2_klein_smoke256_v0621_26062810/checkpoint-00032000/inference_manifest.json \
  --device cuda \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa
```

旧的 `--checkpoint ... --output_dir ...` 方式仍然兼容。使用旧方式时，不要把 `--output_dir` 写成 `.../RealLQ250` 这类单个数据集目录；多数据集模式会自动追加 `dataset_name`，否则路径会变成 `.../RealLQ250/realLQ250` 和 `.../RealLQ250/realLR200`。

### 一键训练-推理-评估 Pipeline

如果希望从训练到多个 checkpoint 推理、再到多个数据集评估都无需手动串命令，可以使用：

```bash
python tools/run_rg_flux_pipeline.py \
  --train_config configs/train_rg_flux2_klein_sr_stage0b_512.yaml \
  --accelerate_config configs/accelerate/zero3_bf16_cpu_offload.yaml \
  --num_processes 1 \
  --checkpoint_steps 20000 40000 \
  --dataset_dirs \
    realLQ250=/data/datasets/omgsr_eval/RealLQ250/lq \
    realLR200=/data/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --inference_output_root eval/inference/rg_flux2_cached_fixed_prompt \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda
```

脚本会先复制训练 YAML 到本次 run 目录下的 `pipeline_runtime_config.yaml`，并写入固定的 `training.exp_name`，所以原始配置文件不会被修改。训练完成后，它会依次处理 `checkpoint-00020000`、`checkpoint-00040000`，每个节点都会先对所有 `--dataset_dirs` 推理，再用对应的 `inference_manifest.json` 自动评估。

输出结构类似：

```text
eval/inference/rg_flux2_cached_fixed_prompt/
`-- <run_name>/
    |-- checkpoint-00020000/
    |   |-- realLQ250/
    |   |-- realLR200/
    |   |-- inference_manifest.json
    |   `-- metrics/
    `-- checkpoint-00040000/
        |-- realLQ250/
        |-- realLR200/
        |-- inference_manifest.json
        `-- metrics/
```

每次 pipeline 还会写出：

```text
exp_rg_flux_sr/<run_name>/pipeline_manifest.json
```

其中记录 runtime config、checkpoint path、每个推理 manifest、metrics 输出目录和子命令返回状态。

如果训练已经完成，只想对已有 run 补跑多个 checkpoint 的推理和评估：

```bash
python tools/run_rg_flux_pipeline.py \
  --skip_train \
  --run_dir exp_rg_flux_sr/<existing_run> \
  --checkpoint_steps 20000 40000 latest \
  --dataset_dirs \
    realLQ250=/data/datasets/omgsr_eval/RealLQ250/lq \
    realLR200=/data/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
  --inference_output_root eval/inference/rg_flux2_cached_fixed_prompt \
  --text_encoding_mode cached \
  --text_embedding_cache datasets/text_embed_cache/flux2_klein_fixed_sr_prompt \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16 \
  --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa \
  --metric_device cuda
```

`latest` 会解析为 `<run_dir>/checkpoints/` 下最新的 `checkpoint-*` 目录。若指定 checkpoint 或 `rg_flux_adapters` 不存在，脚本会直接报错，避免静默跳过导致后续评估混乱。v1 版本默认等训练完全结束后再推理和评估，不会在训练中并发抢 GPU 显存。

### FLUX.2-klein LoRA-MoE Checkpoint 推理

MoE 推理仍然使用同一个 `inference_rg_flux_sr.py`。只要 `--config` 指向 MoE 配置，`--checkpoint` 指向包含 `flux2_klein_lora_moe_state.pt` 的 `rg_flux_adapters` 目录，脚本会自动启用 Top-2 routing。

```bash
python inference_rg_flux_sr.py \
  --input path/to/lq_or_folder \
  --output_dir outputs/rg_flux2_klein_lora_moe_sr \
  --checkpoint exp_rg_flux_sr/rg_flux2_klein_sr_ms_stageA_flux2_image_concat_size512_flux2_klein_lora_moe_smoke256/checkpoints/checkpoint-XXXXXXXX/rg_flux_adapters \
  --config configs/train_rg_flux2_klein_sr_moe_smoke_256.yaml \
  --jsonl_path datasets/inference_cleaned.jsonl \
  --lr_cond_mode flux2_image_concat \
  --no-use_prompt \
  --no-use_suggestions \
  --num_inference_steps 25 \
  --upscale 4 \
  --dtype bf16
```

如果 `use_prompt=true`，MoE 推理依赖 cleaned profile 构造 prompt，推荐使用 `datasets/inference_cleaned.jsonl` 或同结构 JSONL。当前配置中 `use_prompt=false`、`use_suggestions=false` 时，所有图片使用固定 `DEFAULT_SR_PROMPT`；此时 JSONL 仅在还需要 degradation vector 或其他记录字段时才需要。若传入了 `--jsonl_path` 但找不到输入图片对应记录，推理脚本会跳过该图片并写入 `inference_failures.jsonl`。

参数说明：

- `--input`：输入 LQ 图片、文件夹，或 txt 列表。
- `--output_dir`：SR 图片输出目录。
- `--checkpoint`：训练保存的 adapter 目录，通常指向 `.../checkpoint-XXXXXXXX/rg_flux_adapters`。
- `--config`：训练时使用的配置。推理会用其中的模型路径、条件模式等设置。
- `--lr_cond_mode`：LR 条件模式；FLUX.2 新训练推荐 `flux2_image_concat`，并且必须与 checkpoint 的训练模式一致。
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

该参数只影响 `latent_adapter` 和旧版 `latent_concat`：

- `latent_adapter`：控制加入 text/context 序列的 LR 条件 token 数。
- `latent_concat`：控制经过池化和 `concat_proj` 后加入 image 序列的 LR token 数。
- `flux2_image_concat`：忽略该参数，直接使用完整的 FLUX.2 LR 空间 token；token 数由输入尺寸自动决定。

对于当前推荐的 `flux2_image_concat`，不要通过增大 `lr_token_count` 来增强 LR 条件。

### `num_inference_steps`

推理时 multi-step flow matching 的采样步数。常用值：

- `10`：更快，质量可能不稳定。
- `25`：默认推荐值。
- `50`：更慢，质量收益不一定线性。

### `training.add_datetime_suffix` / `training.run_id`

默认情况下，如果没有显式设置 `training.exp_name`，训练实验目录会自动在原有名字后追加小时级 run id，例如 `_26062810`，避免同一个 `suffix` 重复启动时覆盖旧实验：

```yaml
training:
  add_datetime_suffix: true
  run_id: null
```

如果希望手动固定 run id，可以设置：

```yaml
training:
  run_id: 26062810
```

如果目标实验目录已经存在，脚本会自动追加 `_r02`、`_r03`。如果需要严格恢复某个旧实验目录，请显式设置完整的 `training.exp_name`。

### Loss 数值记录与曲线图

训练主进程会在日志目录里记录 loss 数值：

```text
exp_rg_flux_sr/<run_name>/logs/
|-- loss_history.jsonl
|-- loss_history.csv
|-- loss_summary.json
`-- loss_curves.png
```

`loss_curves.png` 会把 `loss_total`、`loss_fm`、`loss_latent`、`loss_charb`、`loss_lpips`、`loss_down` 以及可用的 MoE router loss 画在同一张图上，横坐标为 step，纵坐标为 loss。默认绘图频率跟随 `training.save_every`，也就是每次保存 checkpoint 时更新一次；同时会保留对应 step 的快照：

```text
loss_curves_step-00032000.png
```

如需单独调整绘图频率，可以设置：

```yaml
training:
  loss_plot_every: 5000
```

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
