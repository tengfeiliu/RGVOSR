# RL-SR：V5 有效方案集成（C3）

本分支新增独立的共享递归修正训练/推理入口。固定的 FLUX.2 Klein F0 负责第 1 轮；第 2～4 轮使用同一个视觉 Transformer，对前轮 SR 做有界残差修正。每轮输入同时包含前轮 SR 和**原始 LR 的 1024 个空间位置**，轮次 embedding 与跨轮隐藏状态由共享网络传递。训练按 1→2→3 次展开逐步增加轮数，混入 10% HR 身份输入。配对 HR 只参与训练损失和离线评价，不进入推理。

V5 的 256 张测试图上，C3 第 1→4 轮 PSNR 为 18.132→18.882→18.966→19.002 dB；关闭/打乱原始 LR 后第 4 轮验证 PSNR 跌至约 12.7/12.5 dB。以上是已完成 V5 实验的证据，不代表在新数据上重训必然得到相同数值。C3 比 C2 的增益同时包含展开训练与 hidden state，暂不能独立归因。C4 old-policy 刷新、FLUX 后轮重复生成与 FLUX LR 注入在 V5 未展示额外收益，均未放入此默认入口。

## 输入

需要先用现有 FLUX.2 Klein `checkpoint-00036000` 生成首轮 F0 图像。训练工具**不重新生成 F0**。传入 JSON 文件，顶层包含 `train`、`validation`、`test` 数组；每行至少包含：

```json
{"lq_path": "/path/original_lr.png", "hq_path": "/path/hr.png", "_f0_path": "/path/f0_round1.png", "_sample_key": "unique-id", "source_hq_path": "/path/original-source.png"}
```

所有图像均为对应的 512×512 RGB；`lq_path` 是**同一张图的原始退化观测**，不是前轮 SR。前轮 SR 由 `_f0_path` 开始，后续各轮使用自身生成结果。`source_hq_path` 在同源裁块存在时用于阻止跨 split 泄漏。推理真实图像可省略 `hq_path`，并加 `--no-hr`。

本机已有可直接用来核验入口的 V5 split：

`/media/xd/data/code/RGVOSR-self-refine-v5/exp_rg_flux_self_refine/observation_loop_v5_20261010/split.json`

迁移服务器时需要把 split 中的绝对路径改到新机器对应位置。原 F0 首轮、LR、HR 图片需完整保留。建议先运行 `infer --max-samples 2` 检查路径与几何。

## 训练和推理

在 RL-SR 仓库根目录，激活原有 `sr-flux2` 环境：

```bash
conda activate sr-flux2
python tools/run_rl_sr_observation_loop.py train \
  --split-json /path/to/split.json \
  --output exp_rg_flux_rl/observation_c3 \
  --updates 1200 --batch-size 256 --micro-batch 256
```

80GB A100 可先用 `--micro-batch 256`；显存不够时只降 micro-batch（例如 64），保留有效 batch 256。默认每 200 次更新保存一次 checkpoint 并在 validation 做完整 4 轮 PSNR 验证，`metrics.jsonl` 记录结果。训练从中断点继续：

```bash
python tools/run_rl_sr_observation_loop.py train \
  --split-json /path/to/split.json --output exp_rg_flux_rl/observation_c3 \
  --resume exp_rg_flux_rl/observation_c3/checkpoint-00600.pt
```

选择 checkpoint **只能看 validation**；选定之后再执行 test。建议同时检查逐轮 PSNR、SSIM、LPIPS、DISTS 和生成图像；当前脚本直接写出每图 PSNR 与四轮原始图像，完整指标可接项目原有评估工具。推理命令：

```bash
python tools/run_rl_sr_observation_loop.py infer \
  --split-json /path/to/split.json \
  --checkpoint exp_rg_flux_rl/observation_c3/checkpoint-01200.pt \
  --eval-split test --output exp_rg_flux_rl/observation_c3_test
```

输出 `round_01` 是原 F0，`round_02`～`round_04` 是连续浮点图像递归后的 PNG。`summary.json` 与 `per_image_psnr.csv` 记录评价。对真实无 HR 数据，使用 `--no-hr`，此时仅导出四轮图片，不计算 PSNR。该入口也可加载原 V5 的 `C3_recursive/checkpoint-01200.pt`，便于对照。原 V5 输入的 FLUX/MoE/prompt 作用于 F0 生成；C3 后轮是独立视觉修正器，没有文本或 MoE 输入。

本集成不替代原 RL-SR 的 C/E/RL 阶段。原训练脚本与配置保留；论文主张需另行验证 F0 预训练数据与测试数据独立，并在新的 heldout/真实退化数据上检查视觉质量。
