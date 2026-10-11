# RL-SR 分支修复与验证记录

目标：修复本分支已确认的初始化、路由、数值精度、原 LR 锚点和多轮训练覆盖问题，并提供直接的 SR 质量监督。按用户要求在原 RL-SR 工作区修改，保留已有 YAML 路径和数据修改。原审计与历史26k/小原型实验仍保留，不能改称36k效果。

## 实施顺序与验收

1. 回归测试复现被冻结 router 仍使用 soft/2.0，以及 BF16 概率丢失弱偏好。修复 configure_rl_trainables 固定 rollout 路由；NFT operands/probability/confidence/reduction 使用 FP32，主干仍 BF16。
2. 新配置 context_dim=7680、显式36k初始化；C 缺初始化时提前报错。保持 iqa_suggestion+caption 和512文本长度，不缩短原生图像token。
3. spatial 锚点：原 LR 二维池化成8×8网格，加二维位置特征，用文本查询对空间锚点做注意力，再通过现有zero gate调制文本。默认旧接口保留 legacy 模式；新checkpoint保存/检查锚点模式元数据，防止学过的旧adapter无意换语义。验证空间差异可区分、零gate等价、梯度、旧F0加载及新checkpoint往返。
4. C 和 E 的配对replay接入RGB Charbonnier与LPIPS，复用现有可微VAE解码、对齐随机crop与LPIPS实现。默认新配置权重0.1/0.05，256像素监督crop；E乘replay图像权重0.25。不引入错误的严格bicubic→LR约束。验证实际decoder梯度与开关，不将权重称为已验证最优。
5. round-balanced batch sampler：每个batch至少含两种可用轮次，轮次循环覆盖，小轮次桶允许重新打乱采样；可关闭恢复普通shuffle。C/E update/replay均接入。新配置batch=4；冻结F0 bootstrap生成第1至第3轮输出，构建目标第2至第4轮的C状态。旧单轮缓存需重建，不能伪造多轮数据。
6. CPU回归、原项目全套、A100真实36k加载及训练步验证。保留日志、损失分项、梯度、显存、冻结参数检查。一次独立代码审查后修复重要发现。

## 设计边界

- 64空间锚点仍压缩原LR；不是完整原生LR流，也不声称已经证明质量提升。legacy可用于8-token对照，spatial可用256-token消融。
- 多轮混合离线状态训练不是全链展开反向传播，也不等于已经实现EMA/on-policy刷新。
- 新loss与更大batch是可配置实现；一次训练步只验证可执行性、梯度和数值，不证明收敛或论文质量。
- 用户原有datasets/LSDIR_cache两项修改不属于本次变更，不覆盖、不提交。

## 进度与证据

- 已在原工作区 `RL-SR` 分支完成以下源码修改与训练步验证，基线提交为 `e5a7da4`。改动目前留在工作区，未提交 Git；用户原有数据修改保持原样。
- 修改前全套：275 passed / 7 failed；修改后：299 passed / **同样的7个失败**。新增24个测试用例覆盖此次行为修复，相关测试集合62 passed。详细失败名与日志见下文。
- A100 80GB、`sr-flux2`：真实 C 两个更新步与 E 一个更新步分别在 cached 和 online 文本模式通过；默认 online 模式峰值分配显存 C=66.406 GiB、E=66.904 GiB。batch=4，replay batch=4，未开启 gradient checkpointing。
- 训练检查：每阶段180个注册的冻结参数均无梯度、版本未变；可训练adapter参数均为有限值；router为topk/0.7/2；C之后anchor_gate非零。冻结VAE为单独模块，图像loss会经过其可微解码，但不更新其参数；Qwen3为冻结文本编码器。

## 实际修改位置

| 问题 | 修改 | 位置 |
|---|---|---|
| 初始化可能缺少36k；配置维度易误读 | 明确FLUX.2 Klein base-4B BF16、36k adapter和context_dim=7680；C缺少初始化时提前报错；一键脚本默认读取同一配置的F0路径 | `configs/rl_sr_refinement_flux2_klein_moe.yaml`、`train_rg_flux_refiner_sft.py`、`tools/run_rl_sr_stage_ae.sh` |
| router参数冻结但训练/rollout策略不同 | `configure_rl_trainables(train_router=False)`同时锁定推理路由；训练summary记录实际模式 | `rl_sr/snapshots.py`、C/E入口 |
| BF16概率舍入吞掉弱偏好 | 概率、confidence在入口保持FP32；NFT velocity/residual/weight/reduction和reference残差用FP32 | `losses/output_nft.py`、`train_rg_flux_output_rl.py` |
| 原LR的8个一维token丢失空间信息 | 新配置使用64个二维锚点，位置编码+文本查询cross-attention+zero gate；保留legacy对照 | `models/sr_refinement_condition.py`、`models/flux2_klein_sr_artist.py` |
| C/E replay缺少直接SR质量监督 | 新增共享可微RGB Charbonnier+LPIPS，随机对齐256crop；像素loss直接对配对HR监督 | `rl_sr/training.py`、C/E入口 |
| batch=1/随机shuffle不能保证多个轮次；C只有一轮状态 | round-balanced batch采样；F0实际生成3轮，构建目标第2/3/4轮的C状态；bootstrap轮数独立于推理轮数 | `dataloaders/sr_refinement_state_dataset.py`、一键脚本、`tools/resolve_rl_sr_run_dir.py` |
| 新旧锚点接口可能无意混载 | 新checkpoint记录并验证锚点模式/长度/attention_dim/max_round；声明refinement但文件或参数缺失时拒绝加载 | `models/flux2_klein_sr_artist.py` |
| CUDA autocast下topk scatter dtype报错 | scatter目标张量采用softmax实际dtype，允许CUDA AMP下的FP32概率 | `models/lora_moe.py` |
| E两份可训练计算图同时存在，batch4显存不足 | 先backward策略loss，再forward/backward replay，最后统一clip和一次optimizer step；数学目标及梯度累积缩放不变 | `rl_sr/training.py`、`train_rg_flux_output_rl.py` |

### 编码和锚点

保持36k原有的 `iqa_suggestion + caption`，512文本位置、7680特征维度。512×512图片仍保留原生32×32=1024图像token，待恢复图与当前SR分别1024；没有把原生图像流压成64个token。原审计已核对42045条训练提示和450条推理提示，最大长度396/437，均小于512；本次不改提示模板。

新的原LR支路是 `128×32×32 latent → 8×8池化 → 64锚点`。二维坐标参与key，512个文本位置作为query，得到对各文本位置的原LR残差，再经zero gate加入文本embedding；额外attention的scores为512×64。它避免旧1D池化的空间混淆，但仍是压缩支路，不能当作完整原LR原生图像流。anchor_gate与round_gate初始为零，从36k开始时新增支路不改变F0输出。

### 训练loss与显存调度

```text
z_t = (1 - sigma) * z_HR + sigma * eps
v_target = eps - z_HR
z0_pred = z_t.float() - sigma.float() * v_pred.float()

L_image = 0.10 * Charbonnier(decode(z0_pred_crop), HR_crop)
        + 0.05 * LPIPS(decode(z0_pred_crop), HR_crop)

L_C = 1.00 * L_FM + 0.10 * L_latent + L_image

L_E = 1.00 * L_NFT + 0.10 * L_reference
    + 0.25 * L_replay_FM + 0.05 * L_replay_latent
    + 0.25 * L_replay_image
```

`L_latent`保持SmoothL1。解码监督crop默认256×256，LPIPS输入clamp到[-1,1]并resize到256；可用 `rl_sr.image_loss` 控制权重、crop、频率和warmup。省略配置或两项权重均为0时不执行解码，兼容旧训练目标。VAE、LPIPS被冻结，但对预测latent保留梯度。没有加入把复杂退化LR当作严格bicubic结果的约束。以上权重是工程起点，未完成质量消融。

E原来先构造两个可训练前向，再对总loss反向，实际在replay前向出现显存分配失败。现在对policy部分先反向，释放其保存的激活，再构建replay图反向。两部分通过同一个 `Accelerator.backward`，保持AMP/梯度累积缩放，之间没有optimizer step；统一clip后更新一次。回归分别检验累积步数1和2，梯度等于loss之和且第一份图没有retain。默认不依靠开启gradient checkpointing解决该显存问题。

### 真实多轮状态与采样

新的bootstrap顺序为 `F0: LR→y1→y2→y3`；C使用 `(原LR,y1,round2,HR)`、`(原LR,y2,round3,HR)`、`(原LR,y3,round4,HR)`。y2/y3是真实递归输出，不能复制y1再改轮次。已有y1复用逻辑保留，新增bootstrap不重复计算其IQA。默认 `sft_bootstrap_rounds=3`，与 `iterations=4` 分开控制。

新sampler按round分桶，每batch至少包含配置要求的不同轮次；三轮、batch4时每batch包含第2/3/4轮。各轮均衡循环，稀少桶重新打乱采样；一个epoch是floor(N/B)个完整batch，并非每条样本恰好经过一次。轮次多于batch时跨batch/epoch连续覆盖，避免后期轮次永远漏采。E更新buffer与配对replay分别应用该规则。单轮缓存、batch小于min_rounds、数据不足一batch时明确报错；关闭 `round_balanced_batches` 可恢复普通shuffle。

## 兼容、启动与旧缓存处理

- 指定36k F0没有新增refinement权重，可以直接初始化新的spatial模型。
- 已训练的旧8-token refiner使用 `condition.refinement.anchor_mode: legacy` 和原来的 `anchor_tokens: 8`；不能把它直接加载成spatial并声称语义不变。spatial训练需从F0重新开始。新spatial checkpoint必须使用保存时的锚点设置。
- 旧的一键运行目录中 `02_states_for_c.jsonl` 通常只有目标round2，不能直接用于新多轮采样；应使用新的运行目录重新执行f0/c，或者显式关闭round-balanced作旧基线。不要覆盖旧实验并混用其C checkpoint。
- `artifacts/gain_validation_20261005/flux2_36k_audit/recommended_identity_config.yaml`是修复前的历史审计配置；当前分支应使用下面的生产配置。历史审计的“尚未修复”描述保留，不能当作当前源码状态。

使用当前配置从头准备少量真实bootstrap数据，再训练C（命令仅供复用，本次没有启动长期训练）：

```bash
cd /media/xd/data/code/RGVOSR
conda activate sr-flux2
TRAIN_MAX_SAMPLES=32 F0_TRAIN_IQA_SAMPLES=0 bash tools/run_rl_sr_stage_ae.sh f0
# 用上一条输出的实际运行目录继续，避免创建另一个空目录：
RL_SR_RUN_DIR=/path/to/the/printed/run SFT_MAX_STEPS=100 bash tools/run_rl_sr_stage_ae.sh c
```

默认从YAML读取本机36k目录，仍支持 `F0_CHECKPOINT` 显式覆盖。`f0`会生成 `02_f0_bootstrap_trajectory/` 与多轮状态。完整 `all` 流程和原来的恢复阶段命令仍可用，本次只验证了脚本的阶段调用顺序和真实C/E训练步，未执行全部IQA/reward评价链。

## A100实际验证结果

探针在原分支调用真实训练入口、真实4B Transformer、真实36k adapter和VAE/LPIPS；仅包装model factory，以保留训练后模型引用并检查冻结参数。使用两个已有验证源图，实际F0按25 NFE各生成第1/2/3轮，共6个中间状态，覆盖目标round2/3/4。文本缓存由本机FP32 Qwen3生成；online试验则由训练入口实时编码同一提示。

| 文本模式 | C更新步 / batch | C峰值分配GiB | C调用秒 | E更新步 / batch+replay | E峰值分配GiB | E调用秒 |
|---|---|---:|---:|---|---:|---:|
| cached | 2 / 4 | 51.318 | 13.428 | 1 / 4+4 | 51.820 | 15.032 |
| online（默认） | 2 / 4 | 66.406 | 23.803 | 1 / 4+4 | 66.904 | 26.034 |

时间包括各次trainer的模型加载、预处理、训练和保存，不能直接当作纯训练吞吐。显存为torch.max_memory_allocated，不是整个服务器/所有进程显存。A100可见总量79.14GiB，测试前可用约68GiB，已有其他显存占用未改动。batch4实际可运行；没有测试最大batch，不能称batch8/16已可行。缓存可释放约15GiB常驻Qwen3占用，但本次没有把全数据集改成cached，避免无准备缓存造成默认启动失败。

在线模式loss分项：

| 阶段/步 | 总loss | FM或replay FM | latent（C） | Charbonnier | LPIPS | NFT | reference |
|---|---:|---:|---:|---:|---:|---:|---:|
| C / 1 | 0.910970 | 0.875766 | 0.146274 | 0.131154 | 0.149225 | — | — |
| C / 2 | 0.892767 | 0.866447 | 0.090083 | 0.117886 | 0.110470 | — | — |
| E / 1 | 0.900992 | 1.016318 | — | 0.092039 | 0.093122 | 0.640467 | 0.000000 |

E开始时G_phi=G_old=G_ref，因此reference为0合理。E动作是训练后的C模型真实采样latent，但为了验证更新链只使用1 NFE，概率人为设为0.4995、confidence=1；没有进行真实IQA reward评分。这验证弱偏好精度与训练执行，**不是多轮SR质量、reward有效性、收敛或论文效果验证**。C两步loss降低也不能作质量结论。源图取自既有验证manifest，仅供调试训练步，生成checkpoint不应进入正式验证结果。

## 回归与审查

新增测试先复现失败，再修复：固定路由、NFT弱偏好、2D锚点区分/梯度/no-op、checkpoint兼容/完整性、可微图像监督、多轮batch、实际shell启动参数、CUDA AMP topk dtype、E顺序反向。源码compileall、bash语法、git diff --check通过。

```bash
/home/xd/miniconda3/envs/sr-flux2/bin/python -m pytest -q \
  tests/test_rl_sr_training_fixes.py tests/test_rl_sr_core.py \
  tests/test_rl_sr_f0_reuse.py tests/test_rl_sr_sample_limit.py tests/test_lora_moe.py
# 62 passed
/home/xd/miniconda3/envs/sr-flux2/bin/python -m pytest -q
# 299 passed / 7 failed / 12 subtests passed
```

全套以下7项在修改前后均失败，未扩大范围去改Stage-0配置/已有IQA接口：

1. `test_degradation_analysis.py::DegradationAnalysisTests::test_semantic_validation_rejects_risky_suggestion_with_zero_risk`
2. `test_precropped_prompt_curriculum.py::PrecroppedPromptCurriculumTests::test_new_single_and_moe_configs_share_precropped_curriculum`
3. `test_rg_flux_sr_components.py::RGFluxSRComponentTests::test_flux2_klein_smoke_config_is_isolated`
4. `test_rg_flux_sr_components.py::RGFluxSRComponentTests::test_smoke_256_config_keeps_main_config_unchanged`
5. `test_rg_flux_sr_components.py::RGFluxSRComponentTests::test_stage0b_configs_cover_single_and_moe_backends`
6. `test_rg_flux_sr_components.py::RGFluxSRComponentTests::test_stage0b_image_loss_can_crop_decode_to_reduce_memory`
7. `test_rg_sr_metrics.py::RGSrMetricsTests::test_evaluate_metrics_writes_each_metric_for_each_image`

独立代码审查发现的3项重要问题也已增加失败测试并修复：轮次多于batch时后期轮次漏采；checkpoint metadata声明refinement但整个condition文件缺失时漏验；将bootstrap限制错误地施加给所有推理/恢复模式。其后GPU还发现并修复了topk AMP dtype与E双图显存问题。审查发生于GPU新增修复之前，后两项另有真实GPU及回归证据；不把早先审查说成覆盖最终所有改动。

## 保存的证据

目录：`artifacts/rl_sr_fixes_20261005/`。

- `results.csv`：cached/online逐阶段loss、时间、显存。
- `source_provenance.csv`：当前相关源码、配置、36k权重与两个原有用户数据文件SHA256；数据文件仅记录身份，未修改。
- `review.md`：审查发现、修复及验证范围。
- `gpu_verify.py`：本服务器可复跑的真实训练探针，依赖历史manifest和模型路径；输出会覆盖该诊断目录，不能用它启动正式质量实验。
- `gpu/verification_cached.json`、`gpu/verification.json`（online）、`gpu/config.yaml`、`gpu/states.jsonl`、`gpu/scored_rollouts.jsonl`、C/E checkpoint和summary。
- `red_*.log` / `green_*.log`：各组失败→通过证据；`baseline_pytest.log` / `final_pytest.log`：全套前后对照。
- `gpu_verify_memory_fixed.log`、`gpu_verify_online.log`：两次最终成功；更早失败日志保留，分别包含探针API/payload修正、实际AMP dtype失败和E显存失败，不能混作最终成功日志。

原始模型、生成图片、checkpoint、JSON及完整日志保留在服务器，artifact目录的.gitignore只将说明、CSV和诊断脚本纳入版本管理。后续需要独立进行36k真实SR质量实验：固定源图隔离划分、配对1/2/3/4轮指标和可视化、匹配计算预算的单轮基线，再消融anchor/loss/策略刷新；本次完成的是分支缺陷修复与执行验证。
