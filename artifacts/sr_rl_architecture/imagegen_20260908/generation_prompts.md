# 多轮输出强化超分架构图

使用内置 image_gen 生成并编辑。最终图已检查跨轮输入箭头、共享参数、冻结模块和训练约束。

## 初始生成提示词

Use case: infographic-diagram.
Generate ONE high-resolution Chinese scientific architecture image for a research implementation plan. It must be clean, exceptionally legible, professionally composed, and much less cluttered than a Mermaid dependency graph. Landscape 3:2 composition, aim for 2560x1700 or higher resolution. White background, dark navy Chinese sans-serif labels, flat vector-like design, generous whitespace. Restrained palette: gray for frozen first-round SR, purple for frozen VLM diagnosis, cobalt blue for shared trainable refinement and parameter-update card, amber for reward feedback, teal for constraints. No decorative photos, no gradients, no 3D, no fake quantitative quality curves, no invented scores, no watermark. Use simple small image glyphs as symbolic outputs. Essential technical correctness is mandatory.

Exact title: "多轮输出强化超分架构"
Exact subtitle: "目标：让第 3、4 轮继续带来恢复收益"
A small readable tag near title: "2 → 3 → 4 轮课程训练 · 可扩展至更多轮"

LAYOUT: three spacious horizontal zones. Avoid crossing edges and avoid long backward arrows. All arrows have clear arrowheads. The same Gφ appears in rounds 2, 3, 4: these are applications of ONE SHARED refiner, NOT independent networks. Do not show separate φ2, φ3, φ4.

ZONE 1: "多轮生成"
A thin teal conditioning rail at the top labeled exactly "原始 LR x：始终保留，接入每轮条件". A small LR input image glyph on the left begins the rail. Short vertical branches of the rail enter all four stage columns. Do not draw a dense web of LR-to-every-box connections.

Below the rail, FOUR evenly spaced stage columns from left to right:
COLUMN 1 heading "第 1 轮".
A gray module "初始 SR：F₀" with clear label "冻结".
Below it a small output image glyph labeled "首轮结果 y₁".
COLUMN 2 heading "第 2 轮".
A purple small module "重新诊断 H" with label "冻结 VLM".
A short downward arrow labeled "修复指令 d₁" to a blue module "共享精修器 Gφ".
Below it small output image glyph labeled "第二轮结果 y₂".
COLUMN 3 heading "第 3 轮".
A purple module "重新诊断 H" with label "冻结 VLM".
A short downward arrow labeled "修复指令 d₂" to a blue module "共享精修器 Gφ".
Below it small output glyph labeled "第三轮结果 y₃".
COLUMN 4 heading "第 4 轮".
A purple module "重新诊断 H" with label "冻结 VLM".
A short downward arrow labeled "修复指令 d₃" to a blue module "共享精修器 Gφ".
Below it small output glyph labeled "第四轮结果 y₄".
Connect the four stage columns with clean left-to-right arrows: output y₁ passes to the second-stage column; y₂ passes to third; y₃ to fourth. Arrows should clearly carry previous output into the next stage, not directly from one model's parameters to another. The stage grouping can summarize the paired inputs to diagnosis and refiner.
After the fourth stage a small right-facing arrow and ellipsis labeled "更多轮".
Place a shared blue bracket below stages 2–4 with exact label "同一组参数 φ · 根据当前图像、指令与轮次精修". This is a weight-sharing annotation, not gradient arrows.
Add one short sentence under the stage headings or at zone foot: "每轮重新观察当前 SR，不复用过时诊断".
Do NOT show quality progressively increasing as a claimed result. This is a proposed architecture.

ZONE 2: "输出强化训练"
Use just TWO main cards side by side and one clear forward arrow between them.
Left wide amber card titled "冻结奖励评估 R".
Inside, two concise aligned rows:
"逐轮增益：ΔR₂、ΔR₃、ΔR₄"
"后续终局回报：评估当前修改的长期贡献"
And a short smaller line: "质量 + 原始 LR 忠实度".
A simple single collector connection from the round-output region above down into this amber card, labeled "各轮输出与后续采样"; do NOT explicitly draw every branch of K×M.
Right blue card titled "更新共享精修参数 φ".
Inside exact lines:
"混合第 2、3、4 轮状态"
"NFT 输出偏好目标 + 约束"
"更新后刷新快照，重新采样"
The arrow from amber to blue reads "奖励权重".
Do NOT draw return arrows from this training card to all three blue upper modules. Matching symbol φ and shared-weight bracket sufficiently show which parameters are updated.
Readable note within or below this zone: "采样与奖励停止梯度 · 不跨生成链反传".
This note must not accidentally imply training parameters never get gradients.

ZONE 3: "训练约束"
Three equal teal items in a single row with short text:
"原始 LR 与各轮内容一致性"
"配对 HR 监督回放（仅训练）"
"固定 reference 行为约束"
No arrows from these three items into every model. Put a short label "共同作用于精修训练" if needed.
Final readable bottom sentence: "验收：第 3 轮相对第 2 轮、第 4 轮相对第 3 轮是否有真实增益，同时保持早轮质量。"

Typography and accuracy:
All Chinese must be correctly spelled and large. Math limited to x, F₀, H, Gφ, d₁–d₃, y₁–y₄, R, ΔR₂–ΔR₄, φ; no lengthy equations. Do not invent components. No internal FM steps, no attention blocks, no separate per-round large models. First-round stays frozen for this main version; later first-round unfreezing is intentionally omitted. The VLM and reward remain frozen while shared image refiner trains. Keep the overall figure airy and readable, with crisp labels and no overlapping arrows or clipped text.

## 最终修订提示词

Edit the provided architecture image. Preserve the title, all existing content, Chinese text, color palette, three-band composition, four stage columns, reward and training panels, all bottom constraints, and image size. Make ONLY this technical routing correction in the upper multi-round generation zone:
The current three horizontal arrows wrongly connect y1 directly to the displayed y2 image, y2 directly to y3 image, and y3 directly to y4 image. Replace those three arrows with tidy orthogonal navy input paths. For each previous output image, a line exits its right edge, travels a short distance right into the gap between stage columns, then bends UP through that gap. From that vertical input path, branch two short right-pointing arrows: one enters the LEFT edge of the NEXT stage's purple '重新诊断 H' box, and the other enters the LEFT edge of the NEXT stage's blue '共享精修器 Gφ' box. Thus y1 visibly feeds BOTH H and G in round 2, y2 feeds BOTH H and G in round 3, y3 feeds BOTH H and G in round 4. Keep the existing H -> instruction d -> G downward arrows and G -> own output downward arrows. Do NOT add a direct arrow from one displayed output image to the next output image. Use thin crisp paths confined to the gaps, never crossing text or cards. Keep the fourth-output-to-more-rounds arrow unchanged.
If space is tight, slightly narrow the internal purple and blue boxes while keeping text readable, but do not move the whole sections. Do not add new explanations or nodes. Everything else remains unchanged.

