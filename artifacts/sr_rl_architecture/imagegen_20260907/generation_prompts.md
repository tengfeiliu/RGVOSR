# SR output reinforcement architecture diagrams

Generated using the built-in image_gen tool. Both images were visually checked for labels, feedback direction, and training constraints.

## Scheme 1 prompt

Use case: infographic-diagram. Create a polished Chinese scientific architecture diagram as a standalone high-resolution raster image, landscape 16:9, approximately 2560x1440 or higher. White background, precise flat vector-like shapes, dark navy Chinese sans-serif typography, generous whitespace, professional paper/presentation figure. Large readable labels; no tiny footnotes, no decorative illustrations, no gradients, no 3D, no watermark. Use restrained gray for permanently frozen modules, cobalt blue for trainable model / training modules, amber for reward evaluation, teal for constraints. Simple data-image symbols and stacked rectangles for candidate images. This must be a very clear simplified architecture, NOT a complex dependency graph. Exactly three horizontal bands: upper band is the left-to-right generation flow, middle band is reward feedback and training, bottom band is three evenly spaced constraint items. Numbered model badges ① and ② correspond to the numbered training cards. DO NOT draw arrows from training cards back up to model boxes: use matching numbers and text to indicate which parameters are updated, so there are no crossing edges or tangled loops. All arrows are orthogonal, short, thin, with unambiguous arrowheads. No hidden internal attention/transformer/flow-step components. Render all Chinese text correctly. A small but clearly readable single sentence at the bottom explains sampling and gradient boundaries. Main title left-aligned, short subtitle. Make every label readable when shown as a large conversation image.
IMAGE 1 ONLY. Title exactly: "方案一｜固定首轮，强化第二轮". Subtitle: "学习如何进一步修好首轮结果".

Upper band section label: "生成过程". Arrange exactly six visual objects left-to-right:
1. small input image icon labeled "原始 LR" and "x".
2. gray model rectangle with numbered badge ①, labeled "首轮 SR" and "F₀ · 全程冻结".
3. simple image icon labeled "首轮结果" and "y₁".
4. blue model rectangle with numbered badge ②, labeled "第二轮精修" and "Gφ · 训练 LoRA", with small but readable additional line "条件：x + y₁".
5. stacked image rectangles labeled "K 个候选" and "y₂¹ … y₂ᴷ".
6. amber evaluation rectangle labeled "输出奖励 R" with two short lines "质量 + 忠实度".
Connect only adjacent objects with straight left-to-right arrows. Above second model/candidates add short text "同一 y₁，不同噪声".

Middle band section label: "学习反馈". From the reward box at right, draw one clean downward arrow then leftward arrow into a small amber card labeled "候选组内比较" and "相对 y₁ 的改进". From this card a leftward arrow to a larger blue training card centered below the second-round model, numbered ② and labeled "第二轮参数更新" with lines "NFT 输出偏好目标" and "仅更新 φ". The first round has no training card or update arrow. Small neutral label below the first-round model: "第一轮始终不更新". No arrows loop back to upper band.
Middle band can read from right to left; arrows visibly point reward -> comparison -> training. These represent feedback data, NOT differentiating through the sampled images.

Bottom band section label: "训练约束". Three simple teal items side by side, with no web of edges: "原始 LR 条件与一致性", "配对 HR 监督回放（仅训练）", "固定 reference 行为约束". A subtle small label connects the band conceptually: "约束作用于第二轮训练".
Bottom sentence verbatim: "候选由冻结快照采样；奖励与候选停止梯度；参数更新后重新采样。"
Ensure no other content, invented metrics, score values, or equations.

## Scheme 2 prompt

Use case: infographic-diagram. Create a polished Chinese scientific architecture diagram as a standalone high-resolution raster image, landscape 16:9, approximately 2560x1440 or higher. White background, precise flat vector-like shapes, dark navy Chinese sans-serif typography, generous whitespace, professional paper/presentation figure. Large readable labels; no tiny footnotes, no decorative illustrations, no gradients, no 3D, no watermark. Use restrained gray for permanently frozen modules, cobalt blue for trainable model / training modules, amber for reward evaluation, teal for constraints. Simple data-image symbols and stacked rectangles for candidate images. This must be a very clear simplified architecture, NOT a complex dependency graph. Exactly three horizontal bands: upper band is the left-to-right generation flow, middle band is reward feedback and training, bottom band is three evenly spaced constraint items. Numbered model badges ① and ② correspond to the numbered training cards. DO NOT draw arrows from training cards back up to model boxes: use matching numbers and text to indicate which parameters are updated, so there are no crossing edges or tangled loops. All arrows are orthogonal, short, thin, with unambiguous arrowheads. No hidden internal attention/transformer/flow-step components. Render all Chinese text correctly. A small but clearly readable single sentence at the bottom explains sampling and gradient boundaries. Main title left-aligned, short subtitle. Make every label readable when shown as a large conversation image.
IMAGE 2 ONLY. Title exactly: "方案二｜终局回报，协同训练两轮". Subtitle: "学习怎样的中间结果更有利于最终恢复".

Upper band section label: "生成过程 · 冻结快照采样". Arrange exactly six visual objects left-to-right:
1. small input image icon labeled "原始 LR" and "x".
2. blue model rectangle with numbered badge ①, labeled "首轮策略" and "Fθ".
3. stacked image symbols labeled "K 个首轮候选" and "y₁ⁱ".
4. blue model rectangle with numbered badge ②, labeled "第二轮精修" and "Gφ", with readable line "条件：x + y₁ⁱ".
5. stacked image rectangles labeled "每个分支续采样 M 次" and "y₂ⁱʲ".
6. amber evaluator box labeled "终局回报" and "Rᵢⱼ".
Connect only adjacent objects left-to-right. This summarizes branches, DO NOT explicitly draw K times M branches. Short small label above second model: "所有分支使用同一后续策略".

Middle band section label: "按轮分配反馈". Put two wide blue training cards side by side, left card badge ① corresponding to the first-round model, right card badge ② corresponding to the second-round model. From terminal reward box draw one clean short downward connection to an amber horizontal feedback rail, split into two noncrossing downward arrows into the two cards; these are reward signals, not backpropagation. No arrows back up to the model row.
LEFT card exact lines: "① 更新首轮 θ"; "Q₁ⁱ = meanⱼ Rᵢⱼ"; "按后续平均回报比较首轮候选"; "NFT 输出偏好目标".
RIGHT card exact lines: "② 更新第二轮 φ"; "固定 y₁ⁱ，计算组内优势 A₂ⁱʲ"; "按最终结果比较精修候选"; "NFT 输出偏好目标".
Between or directly below the training cards add a prominent short statement "交替更新：训练一轮时，另一轮冻结". Do not show a gradient running from the second model into first model.

Bottom band section label: "两轮共同约束". Three simple teal items side by side, with no web of edges: "原始 LR 条件与各轮一致性", "配对 HR 监督回放（仅训练）", "固定初始 reference 行为约束".
Bottom sentence verbatim: "终局回报分配到各轮；无跨轮反传；每次更新后刷新快照并重新采样。"
Ensure reward travels to each training card, Q first averages continuations within each first-round candidate and then compares different first-round candidates; second-round advantage is computed only within a fixed first-round state. Do not pool all second-round outputs into one group. No arbitrary metric scores or decorative text.

