# 3000源图后台验证启动记录

启动于 2026-10-06 00:05:09 +0800，分支RL-SR。用户要求后台一键运行，并明确选择“先完成3000张的监督多轮训练与450张评测，再继续RL”。

## 进程与入口

- 宿主机后台PID：3525735，session/进程组同PID；nohup+独立session，stdin为/dev/null，日志写入文件，退出当前IDE会话后继续运行。
- 实际运行目录：`/media/xd/data/code/RGVOSR/exp_rg_flux_rl/rlsr_f0-036000_flux2-klein_flux2-image-concat_iqa-suggestion_ccondition8_s512_n3000-ss42_r4_k4_sft1000_nft250_eval-reallq250-reallr200_261005-235953`。
- 完整命令计划：运行目录中的`background_plan.sh`，快捷入口`logs/rlsr_n3000_20261006.start.sh`。
- 配置：运行目录`launch_config.yaml`，启动时冻结当前主配置，将本次train_max_samples写为3000；主配置保留原值，后台各阶段读取快照。
- 总日志：`logs/rlsr_n3000_20261006.log`（指向运行目录launcher.log）。
- PID：`logs/rlsr_n3000_20261006.pid`；目录：`logs/rlsr_n3000_20261006.run`。
- 正常或异常退出会写运行目录`launcher.exit_code`；完整流程成功写0。日志按阶段分别写入运行目录logs/。

## 固定设置及顺序

3000张唯一训练源图，seed42，各阶段复用同一子集；训练原JSONL42045条、评测原JSONL450条均不修改。subset input SHA256：`b3aacdced6a0199fe99f5bae4992e82e4fe487e50c6393faef6a59376cb8a09c`。运行快照SHA256：`099e44a3b141c6c15157686a6c61c4487cdddd47f92713fbbe330c6d1cd77086`。

1. 3000源图F0第1轮；冻结36k继续bootstrap第2/3轮，构建9000个C状态（目标round2/3/4）。
2. C训练1000更新步，batch4，原LR spatial64，FM+latent+Charbonnier+LPIPS。
3. 提前完成RealLQ250+RealLR200全部450图的4轮评测，round1是F0，round2至4用C共享refiner。完成后写`C_VALIDATION_COMPLETED`，无需等待RL候选生成即可查看06目录的metric_trends.csv及图片。
4. 继续3000源图C策略的多轮训练状态采集，显式复用已经生成的F0 y1，避免重复推理。建立9000个RL状态；from05对已经完成的06采用resume，不重复生成评测图。
5. 每状态4个候选，共约36000候选；保持每轮25NFE，自动校准/评分，E训练250步，update/replay batch各4，最后完整450图评测与C/RL对比。

F0训练集过程IQA和后续训练多轮IQA固定抽样64图；正式450图评测仍全量7指标。此设置减少过程诊断开销，不改变3000训练图、HR监督、模型或采样25NFE。所有阶段使用本地FLUX.2 Klein base-4B BF16及指定36k。

初始实测F0约9.6s/图，3000图首轮约8h、三轮约24h；C训练和评测另计。评测源图最大432×430、全部小于训练512×512。目标是1–2天拿到第一份监督多轮验证结果，非保证；完整RL还会更久。时间估计基于首批数据，不是完整阶段实测。

## 依赖和核验

主环境`/home/xd/miniconda3/envs/sr-flux2/bin/python`。工作区之外的必要包安装均通过工具自动审批，不改GPU驱动或现有torch/torchvision。

- 原先缺少项目要求的pyiqa，补装pyiqa==0.1.15。
- LIQE需要clip/ftfy，补装openai-clip==1.0.1、ftfy==6.3.1。
- clip依赖pkg_resources，现有setuptools不提供，调整为setuptools==80.9.0以恢复导入。
- 全部安装使用--no-deps；torch保持2.7.1+cu118、torchvision保持0.22.1+cu118。
- 配置7个IQA模型在CPU提前缓存并评分，结果见iqa_check.csv；初次探针字段修正及偶发HF TLS EOF保留在原日志，修正探针对同一权重重试，官方GitHub备源仅用于下载缓存，不改生产模型或指标。
- 具体版本：`{"torch": "2.7.1+cu118", "torchvision": "0.22.1+cu118", "diffusers": "0.38.0", "transformers": "4.57.1", "accelerate": "1.10.1", "pyiqa": "0.1.15", "openai-clip": "1.0.1", "ftfy": "6.3.1", "setuptools": "80.9.0"}`。

本次正式450图只有NR-IQA指标，不能将它们等同于配对PSNR/LPIPS/DISTS、结构保真或论文结论。后台先交付多轮趋势与图像，后续质量判断需结合对应配对评测和视觉检查。

## 常用命令

```bash
cd /media/xd/data/code/RGVOSR
# 下列为本次启动入口；当前已经启动。
nohup bash logs/rlsr_n3000_20261006.start.sh > logs/rlsr_n3000_20261006.log 2>&1 < /dev/null &
# 总流程
 tail -f logs/rlsr_n3000_20261006.log
# 当前F0逐图进度
 tail -f "$(cat logs/rlsr_n3000_20261006.run)/logs/01_f0_round1_train_state.log"
# 宿主机检查进程
 ps -p "$(cat logs/rlsr_n3000_20261006.pid)" -o pid,ppid,etime,stat,args
# 若用户要停止本次流程，信号只发给记录的独立进程组：
 kill -- -"$(cat logs/rlsr_n3000_20261006.pid)"
```

最终启动核验：宿主机PID3525735存活，SID/PGID同PID，已产生61张实际F0 PNG；7/7正式IQA模型CPU前向评分通过（所有分数有限）。启动核验不等于全部训练/评测已完成。

## 2026-10-06：训练目标与评测指标重合的设计核对

用户提出评估指标被用于loss是否不妥。源码核对结果：

- C仅使用FM、latent SmoothL1、配对HR Charbonnier与LPIPS；当前450图的7个NR-IQA指标没有直接进入C loss。训练损失也作为测试指标并非天然无效（例如MSE训练/PSNR测试），但必须在未参与训练/调参的数据上检验泛化，且不能只凭同一感知网络的分数证明真实质量。
- E的质量奖励默认使用CLIP-IQA、MUSIQ、MANIQA-PIPAL，score_sr_rollouts将奖励映射为optimality_probability/confidence进入NFT训练。IQA evaluator虽为no_grad，仍通过偏好标签成为优化目标。三个指标也在正式评测中；它们的涨分是目标达成证据，不能单独证明视觉质量和结构保真改善。新增加的LPIPS也进入E配对replay。
- 当前正式450图评价只有NR-IQA。CLIP-IQA+与CLIP-IQA相关；换指标名称不意味着独立。NIQE、NIMA、LIQE等未直接用于当前reward，可作为辅助证据，但仍不能取代配对HR指标或人类盲评。
- tools/summarize_rl_sr_evaluation.py还计算不同量纲raw oriented delta的平均；该字段不能作为可靠的论文综合效果分数，应按各指标分别比较。

建议在论文评估协议中：按原始HR源图隔离train/validation/test；在独立配对测试集报告每轮PSNR/SSIM、LPIPS（标注参与loss）、未参与训练的DISTS与局部文字/结构错误；真实域保留全NR-IQA但标注reward指标，并加入随机盲评；用validation选择轮数/停止策略、test仅做最终评测；比较匹配总NFE/耗时的单轮基线，独立消融C和E。当前运行按已启动计划继续，以上是审计建议，尚未新增这些配对/盲评实验，也未得到reward hacking已经发生的证据。

参考原始研究：[LPIPS/CVPR2018](https://arxiv.org/abs/1801.03924)、[RFSR的reward hacking约束](https://arxiv.org/abs/2412.03268)、[FinPercep-RM对SR全局IQA奖励局部缺陷的分析](https://arxiv.org/abs/2512.22647)。

## 2026-10-06：共享精修器的递归稳定性与 old/new policy 建议

用户报告另一服务器第2轮显著提升、第3/4轮下降。尚未读取那台服务器的配置、逐图指标及checkpoint，因此下面区分本地源码事实与待实验验证的原因假设。本次是只读审查与方案记录，没有修改后台实验的配置、模型或执行顺序。

### 已确认的实现事实

- C已经混合目标round2/3/4，不能说当前只训练了第2轮。但缓存输入y1/y2/y3均由冻结F0生成；部署第3/4轮输入由G_sft自己生成。后轮producer不同，存在训练与部署状态分布失配的可能。证据：`tools/run_rl_sr_stage_ae.sh:204–218`、`tools/run_rg_flux_iterative_inference.py:581–630`。
- 当前SR已有完整图像条件，轮次已有round embedding，原LR已有二维spatial64锚点。共享参数不等于缺少轮次条件。轮次主要通过文本残差进入；condition8和prompt沿用原LQ描述，未显式表征当前剩余缺陷。证据：`train_rg_flux_refiner_sft.py:157–169`、`models/sr_refinement_condition.py:66–81,113–126`。
- 每轮默认pure_noise、sigma_start=1、25NFE，重新生成完整输出；不是显式的小幅残差纠正。C的完整HR FM/latent/image监督没有实际轨迹增益、近HR保持或停止约束。文本condition adapter的residual不等于SR输出residual。证据：`rg_flux_fm.py:133–145`、`train_rg_flux_refiner_sft.py:151–176`。
- E固定G_ref/G_old，使用一份G_sft生成的buffer更新一次；尚无多周期策略晋升/轨迹刷新。router的EMA usage统计不等于模型policy EMA。证据：`train_rg_flux_output_rl.py:231–254`、`tools/run_rl_sr_stage_ae.sh:296–325`。

因此，共享权重不是已经证实的故障原因。应优先检验自身状态分布覆盖、完整再生成的改动幅度、后轮目标是否学习保留正确细节。更强的模型也可能更快累积幻觉；简单增加轮数或训练步数不能替代这些验证。

### 推荐训练外循环：先监督聚合，再考虑RL

1. 使用已有C模型作为起点，将近期new模型复制为不可变old快照。这里的训练周期t与图像增强轮次k是两个独立索引。
2. 保持F0首轮不变；用old连续生成自己的后轮状态：y1=F0(x)，y2=G_old(y1,x,2)，y3=G_old(y2,x,3)。逐个保存当前图、原LR、轮次、HR、producer hash及采样设置。
3. new从old初始化，用真实old轨迹上的HR监督更新；batch混合各轮，混入历史F0和近HR状态，避免遗忘。new由梯度优化更新。采集阶段no_grad，先不跨整个多轮链反传，以控制A100显存开销。
4. 短程更新后，在固定验证集验证闭环轨迹，再冻结新快照重新采集。先用一轮数据聚合做消融；效果成立后才扩展2–3个刷新周期。
5. EMA可平滑梯度更新后的new，作为下一周期的候选teacher/actor；采集前仍须冻结快照。不要用EMA代替new的学习。多步更新使用old buffer属于近期策略数据上的近似在线训练，不宣称每次梯度都严格on-policy。
6. 若加入NFT/RL：G_ref保留固定参照；每次old晋升必须同步刷新自身状态、候选actions、reward标签和buffer，保留已有policy hash一致性检查。old/new不是推理时第2轮与第3轮的两个不同模型。

这个监督聚合流程借鉴[DAgger](https://proceedings.mlr.press/v15/ross11a.html)对学习器自身访问状态进行训练的思想；本项目使用配对HR监督，不直接继承DAgger的理论保证，也不需要为了实现状态刷新而先加入RL。

### 推荐模型与目标：保留共享主体，学习后轮的保守更新

- 共享FLUX.2主体及主要精修权重，强化轮次/当前残余缺陷的轻量条件调制。当前SR已输入，不应重复称其尚未接入。若消融证明存在轮次任务冲突，可比较共享主体+早期/后期小LoRA或门控专家，暂不复制三份4B模型。新增condition应使用训练/推理都可获得的信号，不能在推理依赖HR。
- 设计真正的局部纠正形式，例如`y_next = y_current + alpha * mask * delta`，支持alpha=0保持当前结果。alpha和mask根据当前状态学习；后轮更保守是否有效应验证。该公式是新方案，当前模型尚未实现，不能把现有完整HR预测直接当delta相加。
- 保留绝对HR监督，增加近HR/已恢复状态的保持训练，以及`max(0, d(y_next,HR)-d(y_current,HR))`这样的变差惩罚。parent固定；不可用恒定正margin要求接近HR的图像仍严格提升。单独使用变差惩罚会允许总是不修改的退化解。这些训练约束不保证测试图上数学意义的单调提升。
- 真实多步输出与训练的随机时间步z0重建代理不同；最终必须验证完整采样的闭环轨迹。若改成从当前latent的小噪声起点/新flow bridge，需要匹配训练路径、速度目标和部署采样，不能只切换sigma_start就宣称修复。
- 先2轮稳定，再逐渐引入3/4轮。允许困难样本继续、已收敛样本保持或停止。停止器可用训练HR构造真实增益标签，但部署不能使用测试HR选最佳轮；单一NR-IQA涨分也不能代替保真约束。
- 中期可研究latent/hidden-state递归并保留记忆，减少RGB输出再VAE编码的反复往返。这更接近looped transformer，但需要单独的架构训练与对照；简单复用latent也会改变当前训练分布，尚未验证。

### 最小消融和判定标准

| 对照 | 相对上一项只新增什么 | 回答的问题 |
|---|---|---|
| A | 当前F0多轮状态+共享精修器 | 基线在第3/4轮是否退化 |
| B | 混入G_old自身轨迹，短程监督更新并刷新 | 状态分布差异是否是主要原因 |
| C | 在B上加入保守更新/保持约束，分别消融组件 | 后轮是否需要减小改动并保留已有细节 |
| D | 在同样数据和目标下比较早/晚两套轻量adapter | 共享参数是否真的造成阶段任务冲突 |

先在按原HR源图隔离的100–200张配对验证图上比较，固定退化、种子和预算，再扩大最终独立测试。保持有效训练样本数/更新预算可比，并报告参数量；不能用新增数据/算力优势解释为架构收益。记录每轮PSNR/SSIM、LPIPS（已参与loss）、DISTS、逐图变差比例和局部文字/结构样例，分开比较感知提升与保真损失；不要把异量纲指标直接平均。

当前450张真实图只有NR-IQA，需另外补配对评测；它们可反映真实域趋势，无法独立验证上述保真稳定性。论文同时比较相同总NFE/耗时下的单次更长采样与多轮方案；用验证集选择停止策略，测试HR选最佳轮只能作为明确标注的oracle上限。

### 论文方向的判断与已有研究

[DRCN/CVPR2016](https://arxiv.org/abs/1511.04491)说明共享权重的递归SR可以有效，但它的特征递归与当前完整RGB生成级联不同，不能据此保证本项目的后轮增益。[IAFS](https://arxiv.org/abs/2512.23532)已经研究迭代SR、结构纠偏和频率融合；[OARS](https://arxiv.org/abs/2603.12811)已经研究过程感知奖励与on-policy在线SR对齐。因此“多轮+old/new policy”本身不是充分的创新证据。这两篇此处按原始预印本引用，不额外宣称录用状态。

更值得验证的主线是：共享递归SR为何后轮退化；自身轨迹训练与状态感知的保守更新能否使增加计算预算带来可重复的质量收益，并在接近收敛时稳定停止。阶段LoRA、EMA、停止器是候选组件，应通过消融保留有效且必要的部分。CVPR潜力取决于相对已有工作的明确差异、真实闭环多轮收益、计算公平性和独立保真/人评证据，当前没有新实验结果可证明这些目标已经达成。

## 团队汇报流程图

2026-10-06应用户要求绘制，表达的是上述建议方案，不代表当前后台代码已实现策略周期刷新。图均为16:9；PNG为2400×1350，SVG为矢量图，PDF为两页。

- [图1：训练外循环](diagrams/01_training_outer_loop.png)：冻结old → 采集自身状态 → 构建监督数据 → 梯度训练new → 验证new真实自递归 → 冻结下一周期old并重新采集。
- [图2：状态采集与HR监督](diagrams/02_rollout_and_supervision.png)：固定F0生成y1，共享old生成y2/y3；y1/y2/y3分别作为第2/3/4轮训练输入，监督来自配对HR，梯度只更新new。
- [两页PDF](diagrams/old_new_sr_workflow.pdf)、[图1 SVG](diagrams/01_training_outer_loop.svg)、[图2 SVG](diagrams/02_rollout_and_supervision.svg)。
- 本地可复现绘图脚本：`diagrams/draw_policy_workflow.py`；使用sr-flux2的matplotlib及系统Noto CJK字体，不加载训练模型或占用GPU。

30秒汇报说明：固定36k模型先生成首轮结果，冻结的old精修器继续生成真实后轮状态。这些状态只作为new的训练输入，监督仍是HR。new从old初始化，混合训练各轮状态；短程更新后，让new真正连续精修做验证，再将选定快照冻结为下一周期old并重新采集。图像轮次k和训练周期t是两个循环；EMA是可选的参数平滑，不能替代梯度训练。

## 2026-10-06：保守纠正的具体接入方案（尚未实现）

用户要求解释alpha/M/residual、近HR保持与变差惩罚的具体实现。本次核对源码及方案，不修改训练代码，不运行新GPU实验，不调整后台基线。

### 第一阶段：真实候选图 + RGB接受控制器

当前artist输出的是FM速度场，不能直接当作图像残差；采样器`rg_flux_fm.py:105`有no_grad，C训练`z0_pred`只是随机时间步的一次重建代理。最小接入不改现有25NFE flow路径，令：

```
p = G_old(current, original_lr, round, seed)  # 实际完整采样并解码的候选
delta = p - current
alpha, mask = Controller(current, original_lr_up, p, delta, round)
executed = current + alpha * mask * delta
```

这一步是受控候选融合，不是已经训练了直接残差生成器。权重alpha*mask限制在[0,1]时，每像素只能在当前RGB与候选RGB的连线上选择，不能创造候选没有提供的纠正方向。它可先检验“接受过量”是否导致后轮退化，不能保证每轮有新增收益。

控制器可用小CNN：输入12个RGB通道（current、原LR上采样、p、p-current）和轮次embedding；下采样4–8倍提取特征，全局池化head输出alpha[B,1,1,1]，空间head输出M[B,1,H/8,W/8]并平滑上采样。M在RGB三个通道共享，避免独立色彩门控产生色偏。head通过sigmoid约束范围，alpha初始偏小、M保持可打开，避免两者同时硬归零阻断梯度。alpha和M分解不唯一，应记录有效接受权重w=alpha*M，不能自动称其为校准置信度。

完整RGB融合必须保留current_rgb_fp32旁路，供VAE/FLUX的BF16条件副本另存。sigmoid不能提供精确零：可用验证集选定的noop阈值或另行训练的stop分支；执行noop时直接复用原始RGB/uint8图像。不能用再次VAE编码解码或已BF16量化的current声称精确保持。

### 数据与训练目标

采集并保存(original_lr,current,p,HR,target_round,seed,producer,采样配置)，先冻结G，只训练控制器。可从已生成的相邻真实输出复用候选；一个候选即可启动验证，不需要先生成每状态4个RL候选。训练输入与HR使用完全对齐的crop、相同颜色范围和几何变换，HR不能进入控制器的输入。

以逐样本的d(a,h)=lambda_pix*Charbonnier(a,h)+lambda_perc*LPIPS(a,h)为例：

```
L_HR = mean(d(executed, HR))
L_regress = mean(relu(d(executed, HR) - stopgrad(d(current, HR))))
L_total = L_HR + lambda_regress*L_regress + lambda_id*L_identity + lambda_tv*TV(mask)
```

逐图做relu后再平均，避免好样本抵消坏样本；比较双方必须同crop、同权重、同归一化。该regress项实质上在退步样本上增加HR误差的惩罚强度，不是数学上的单调性保证。不能单独训练它，否则全部noop即可达到零regress loss；也不应要求接近HR的状态仍有恒定正margin的提升。

最干净的identity样本是令current=HR，原LR取该HR对应的退化图，p仍由冻结G真实生成；L_identity=|executed-current|。覆盖多个轮次。只对这类精确正确状态明确要求不变；接近HR的真实SR仍有可纠正误差，应保留HR重建目标，不宜一律强制复制current。少量identity混入比例应通过验证集调节，尚无实测最优值。

可离线缓存真实p，控制器训练不保留25NFE的激活图；在线生成时也可以no_grad，但Controller forward、融合、loss必须在no_grad之外。若使用inference_mode，须在退出后复制为普通tensor再给可训练控制器。冻结G时保持eval与部署采样设置一致。原FM loss继续用于原flow；不能把原速度场乘alpha/M后继续假设原FM目标成立。

### 推理闭环与old/new一致性

每次执行后，将融合后的executed作为下一轮current，再为这个新current重新生成候选；不能继续复用原始未门控轨迹的后轮候选。先用原G轨迹启动控制器，随后冻结整个G+Controller快照，收集真实融合轨迹再训练。old/new的策略身份应包含控制器参数、noop阈值及采样配置；当前round_index已表示下一次目标轮次，不可重复加1。

RGB门控发生在候选生成之后，因此没有自动节省25NFE。只有在候选生成前预测停止，或另行实现匹配训练的小步纠正采样，才可能显著减少生成计算。

第一版先在监督训练/推理上验证。接入现有NFT前需明确action语义：raw_sr和保存的z0仍对应原flow候选，executed_sr应是融合结果。融合结果的reward不会自动训练控制器，也不能直接把融合图当作原flow latent动作；任何RL扩展必须单独验证策略身份、动作与reward的一致性。

### 何时升级为直接残差或flow bridge

若控制器明显减少退步，但第3/4轮仍只停留在第2轮水平，说明止损与额外恢复能力是两个问题。可以进一步训练RGB residual head预测新的纠正方向，或构造由current/其加噪版本通往HR的匹配flow路径；二者都需要新目标与采样定义。[ResShift](https://arxiv.org/abs/2307.12348)提供了利用HR/LR残差设计转移路径的相关先例，但不是直接兼容当前36k权重的替换实现。

从纯噪声开始时将FM速度乘零会留下噪声，无法保留current；仅降低sigma_start也不能替代匹配训练。即使新latent路径能在关闭更新时返回E(current)，D(E(current))也不保证RGB恒等，因此精确noop仍应有RGB旁路。

### 必要的最小消融

固定输入和seed，比较noop、完整候选、验证集选择的固定alpha、学习alpha、学习alpha+M；随后分别消融identity/regress/TV。真正质量结论以完整递归链为准，不能只测旧轨迹上的单轮融合。

在配对验证集可额外用HR选取固定alpha网格的逐图oracle上限，判断现有候选是否有可利用的纠正方向；这个oracle只作诊断，不能作为部署方法或测试主成绩。记录各轮保真/感知指标、退步比例及尾部幅度、有效接受权重、修改面积、noop比例，并检查文字/脸部/平坦区域。单纯不再变差不能宣称证明了第三四轮继续恢复细节。
