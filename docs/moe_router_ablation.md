# LoRA-MoE Router 消融实验

该工具在已有 LoRA-MoE checkpoint 上运行 A-F 路由消融，用于区分 SR 提升来自专家参数、动态 Router，还是 Condition8。它不会启动训练，也不会修改 checkpoint、训练配置或现有推理目录。

## 实验模式

| 模式 | 路由 |
|---|---|
| `learned_top2` | 当前学习到的 Top-2，作为基准 |
| `fixed_mean` | checkpoint 前最后 N 个唯一训练 step 的平均 Router 权重 |
| `shuffle_condition8` | 仅在 Router 分支按数据集错排 Condition8；Prompt、Caption 和 timestep 不变 |
| `uniform` | 所有专家使用 `1/E` 权重 |
| `dense_soft` | 使用当前 Router 的 `clean_dense_alpha`，不执行 Top-k 截断 |
| `onehot` | 自动展开成 `onehot_e0 ... onehot_e{E-1}` |

Shared LoRA 在全部模式下保持激活。当前 `SharedRoutedMoELoRALinear` 会计算全部 routed expert 后再按权重融合，因此 `uniform`/`dense_soft` 主要改变路由语义，不应被解释为当前实现中的额外专家计算量。

`fixed_mean` 默认读取 `router/expert_i_usage`，而不是 Teacher 混合后的 `used`。CSV 会先按 `global_step` 去重，只保留断点恢复后同一步的最后一条记录，再截取不晚于目标 checkpoint 的最后 N 步。

## 兼容性与隔离

- 原有 `train_rg_flux_sr.py`、`inference_rg_flux_sr.py` 和 pipeline 不变。
- 诊断推理通过新进程内的临时 Router wrapper 覆盖 `alpha`，子进程退出后自动消失。
- 每个模式使用独立子进程、相同 seed、相同图片顺序、相同 Prompt 和相同推理步数。
- 默认拒绝写入非空模式目录，避免断点残留造成配对随机性不一致。
- 所有结果写入 `router_ablation_outputs`，不会覆盖已有 inference/evaluation/bad-case。
- 默认在每个 checkpoint 的整组消融前后计算 adapter SHA-256；不一致会直接报错。
- `learned_top2` 只记录原始 Router 输出，不改动 `alpha`。首次部署时仍应在服务器用2张图片与原推理入口做一次同 seed 像素/SHA-256核验；本地无完整模型环境时不能把这种一致性当作已经实测。

## 推荐先做单 checkpoint 实验

服务器仓库目录：

```bash
cd /root/autodl-tmp/code/RGVOSR
mkdir -p router_ablation_logs
```

后台运行 checkpoint 24000 的完整 A-F 对比：

```bash
LOG_FILE="router_ablation_logs/router-ablation-24000-$(date +%Y%m%d-%H%M%S).log"

nohup env \
  HF_ENDPOINT=https://hf-mirror.com \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  conda run --no-capture-output -n sr-flux2 \
  python tools/run_rg_flux_moe_router_ablation.py \
    --run_dir exp_rg_flux_sr/rg_flux2_klein_sr_ms_stage0B_flux2_image_concat_size512_s3_condition8_prompt_iqa_suggestion_26080421_r02 \
    --checkpoint_steps 24000 \
    --ablation_modes learned_top2 fixed_mean shuffle_condition8 uniform dense_soft onehot \
    --dataset_dirs \
      RealLR200=/root/autodl-tmp/datasets/omgsr_eval/RealLR200-20260418T151906Z-3-001/RealLR200 \
      RealLQ250=/root/autodl-tmp/datasets/omgsr_eval/RealLQ250/lq \
    --jsonl_path datasets/inference.iqa_caption_suggestion.jsonl \
    --text_encoding_mode online \
    --lr_cond_mode flux2_image_concat \
    --prompt_variant condition8_text \
    --use_prompt \
    --use_suggestions \
    --include_caption \
    --no-use_degradation_vector \
    --full_frame_inference \
    --restore_input_size \
    --num_inference_steps 25 \
    --upscale 4 \
    --dtype bf16 \
    --device cuda \
    --seed 42 \
    --fixed_weight_last_n 1000 \
    --fixed_weight_field usage \
    --condition_shuffle_seed 3407 \
    --metrics clipiqa clipiqa+ nima niqe liqe musiq maniqa-pipal \
    --metric_device cuda \
    --run_bad_cases \
    --bad_case_metrics clipiqa maniqa-pipal musiq \
    --bad_case_mode separate \
    --bad_case_worst_k 50 \
    --bad_case_font_size 40 \
    --ablation_output_root router_ablation_outputs \
  > "$LOG_FILE" 2>&1 < /dev/null &

echo "PID=$!"
echo "LOG=$LOG_FILE"
```

查看日志：

```bash
tail -f "$LOG_FILE"
```

训练图片是否为 512x512 与本工具无关：本工具只做 checkpoint 推理。`--full_frame_inference` 会覆盖训练配置中的 `pre_cropped`，因此推理使用原图全图；`--restore_input_size` 将输出恢复到原图宽高乘以 upscale。

## 多 checkpoint 对比

先确认单 checkpoint 成功后，将命令修改为：

```bash
--checkpoint_steps 16000 24000 40000
```

完整实验每个 checkpoint 包含 9 次推理：A-E 五组，加四个 one-hot 专家。建议先用小型图片列表或单数据集冒烟测试，再运行全部 RealLR200/RealLQ250。

如果只指定 B-F 的某些模式，runner 也会自动补充 `learned_top2`，确保配对比较始终有A基准。

只生成命令和 manifest、不执行模型：

```bash
python tools/run_rg_flux_moe_router_ablation.py \
  ... \
  --dry_run
```

为保证相同图片顺序与随机噪声，模式输出目录默认必须为空。任务中断后请换一个新的 `--ablation_output_root`，或确认不需要旧结果后自行处理旧目录；诊断入口不会自动删除任何内容。

## 输出

```text
router_ablation_outputs/
└── <run_name>/
    ├── router_ablation_manifest.json
    └── checkpoint-00024000/
        ├── learned_top2/
        ├── fixed_mean/
        ├── shuffle_condition8/
        ├── uniform/
        ├── dense_soft/
        ├── onehot_e0/
        ├── onehot_e1/
        ├── onehot_e2/
        ├── onehot_e3/
        ├── router_ablation_comparison.csv
        ├── router_ablation_comparison.json
        └── onehot_output_diversity.csv
```

每个模式目录包含：

- `inference_manifest.json`
- `router_trace.jsonl`
- 各数据集 SR 图片
- `metrics/per_image_scores.csv`
- `metrics/summary_scores.json`
- 可选 `bad_cases/`

`router_trace.jsonl` 逐图片、逐 flow-matching timestep 保存：原始 Condition8、mask、confidence、dense alpha、learned Top-2 alpha、实际使用 alpha，以及 Condition8 打乱时的 donor。

## 结果解释

- `learned_top2 ≈ fixed_mean`：动态 Router 贡献有限。
- `learned_top2 ≈ shuffle_condition8`：Condition8 与图片正确匹配的贡献有限。
- `dense_soft > learned_top2`：硬 Top-2 可能丢弃了当前 checkpoint 中仍有价值的专家输出。
- `uniform > learned_top2`：当前 Router 排序可能不合理；不代表均匀路由一定适合作为最终模型。
- one-hot 图片彼此接近：专家功能分化不足。
- one-hot 图片有差异但 learned 与 fixed 接近：专家有差异，但 Router 没有有效利用差异。

比较器会把 higher-better/lower-better 统一为“`baseline_advantage > 0` 表示 learned_top2 更好”，并报告配对均值、中位数、胜率和 bootstrap 95% 置信区间。

NR-IQA 只能说明感知指标差异。如果评估集存在 GT，应额外加入 PSNR、SSIM、LPIPS、DISTS。one-hot 输出有差异也不等于形成了退化专精；最终还应按 Condition8 退化维度分组比较各专家胜率。
