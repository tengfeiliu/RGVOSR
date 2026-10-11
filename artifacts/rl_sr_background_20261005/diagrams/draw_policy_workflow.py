"""Render the proposed old/new supervised SR workflow for a team presentation.

Only produces local figures; does not import or modify training code.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent
REGULAR = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
BOLD = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
plt.rcParams.update({"pdf.fonttype": 42, "svg.fonttype": "path"})
NAVY = "#172B45"
GREY = "#526174"
OLD = "#2877BA"
NEW = "#168263"
DATA = "#B77915"
HR = "#8553B0"
BACKGROUND = "#F6F8FB"


def label(ax, x, y, text, size=14, color=NAVY, bold=False, ha="center", va="center"):
    return ax.text(x, y, text, fontsize=size, color=color,
                   fontproperties=BOLD if bold else REGULAR,
                   ha=ha, va=va, linespacing=1.45, zorder=5)


def card(ax, x, y, w, h, title, body="", edge=OLD, fill="#EFF6FC", title_size=17, body_size=13):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.015,rounding_size=0.10",
                           facecolor=fill, edgecolor=edge, linewidth=1.4, zorder=2)
    ax.add_patch(patch)
    if body:
        label(ax, x + w / 2, y + h * 0.76, title, title_size, edge, True)
        label(ax, x + w / 2, y + h * 0.35, body, body_size)
    else:
        label(ax, x + w / 2, y + h / 2, title, title_size, edge, True)
    return patch


def arrow(ax, start, end, color=GREY, dashed=False, width=1.8):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16,
                                linewidth=width, color=color,
                                linestyle="--" if dashed else "-", zorder=3))


def line(ax, points, color=GREY, dashed=False, width=1.6):
    ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
            linewidth=width, linestyle="--" if dashed else "-", zorder=3)


def canvas(title, subtitle, number):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 16), ylim=(0, 9), aspect="equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")
    label(ax, 0.6, 8.43, title, 25, bold=True, ha="left")
    label(ax, 0.6, 7.92, subtitle, 13, GREY, ha="left")
    label(ax, 15.4, 8.42, number, 13, GREY, ha="right")
    label(ax, 0.6, 0.28, "建议方案 · 尚未接入当前运行；3000张后台基线保持原计划", 10.5, GREY, ha="left")
    label(ax, 15.4, 0.28, "RL-SR / FLUX.2 Klein · 2026-10-06", 10.5, GREY, ha="right")
    return fig, ax


def outer_loop():
    fig, ax = canvas("图1｜训练外循环：采集自身状态，再更新模型",
                     "训练周期 t：每完成一次模型更新，就用近期模型重新采集递归状态。", "01 / 02")
    x1, x2, x3 = 0.6, 5.8, 11.0
    width, height, top, bottom = 4.4, 1.65, 5.55, 2.95
    card(ax, x1, top, width, height, "① 冻结采集策略 old",
         "来自近期训练模型的参数快照\n采集期间保持冻结", edge=OLD)
    card(ax, x2, top, width, height, "② 采集真实递归状态",
         "固定 F0 首轮 → old 连续精修\n采集全程不计算梯度", edge=OLD)
    card(ax, x3, top, width, height, "③ 构建监督训练数据",
         "原LR + 当前SR + 目标轮次 + HR\n混合各轮及历史状态", edge=DATA, fill="#FFF7E8")
    card(ax, x3, bottom, width, height, "④ 梯度训练 new",
         "初始参数复制 old\n使用配对 HR 监督短程更新", edge=NEW, fill="#ECF8F2")
    card(ax, x2, bottom, width, height, "⑤ 验证 new 的闭环效果",
         "F0 → new第2轮 → new第3/4轮\n检查每轮质量与变差比例", edge=NEW, fill="#ECF8F2")
    card(ax, x1, bottom, width, height, "⑥ 冻结下一周期的 old",
         "选择通过验证的 new 快照\n可选：使用其 EMA 快照", edge=OLD)
    center_top, center_bottom = top + height / 2, bottom + height / 2
    arrow(ax, (x1 + width + 0.10, center_top), (x2 - 0.10, center_top))
    arrow(ax, (x2 + width + 0.10, center_top), (x3 - 0.10, center_top))
    arrow(ax, (x3 + width / 2, top - 0.10), (x3 + width / 2, bottom + height + 0.10))
    arrow(ax, (x3 - 0.10, center_bottom), (x2 + width + 0.10, center_bottom))
    arrow(ax, (x2 - 0.10, center_bottom), (x1 + width + 0.10, center_bottom))
    arrow(ax, (x1 + width / 2, bottom + height + 0.10), (x1 + width / 2, top - 0.10), OLD)
    label(ax, 8, 5.02, "t → t + 1：交接参数快照，并重新采集数据", 16, bold=True)
    card(ax, 0.6, 1.85, 14.8, 0.64,
         "验证中：new 上一轮的输出，必须真正成为下一轮的输入", edge=NEW,
         fill="#ECF8F2", title_size=16)
    label(ax, 0.8, 1.30, "两个循环：t 是模型训练周期；k 是同一张图的精修轮次。", 15, bold=True, ha="left")
    label(ax, 0.8, 0.83, "EMA 平滑已经训练的参数；new 的学习仍来自梯度优化。已有 HR 时可先完成监督版本。",
          13, GREY, ha="left")
    return fig


def inner_details():
    fig, ax = canvas("图2｜展开一次训练周期：状态如何进入 HR 监督",
                     "图像轮次 k：同一个冻结 old 连续精修；new 从这些状态学习修正，HR 始终是真值。", "02 / 02")
    label(ax, 0.6, 7.44, "① 冻结采集 / no_grad：每轮均保留原始 LR 锚点；两处 old 共享参数", 14, OLD, True, ha="left")
    y, h = 6.17, 0.95
    chain = [
        (0.6, 1.25, "原LR", "x", GREY, BACKGROUND),
        (2.25, 1.8, "F0", "固定36k", GREY, BACKGROUND),
        (4.45, 1.25, "y1", "首轮结果", DATA, "#FFF7E8"),
        (6.1, 2.0, "共享 old", "第2轮 k=2", OLD, "#EFF6FC"),
        (8.5, 1.25, "y2", "old输出", DATA, "#FFF7E8"),
        (10.15, 2.0, "共享 old", "第3轮 k=3", OLD, "#EFF6FC"),
        (12.55, 1.25, "y3", "old输出", DATA, "#FFF7E8"),
    ]
    for x, w, title, body, color, fill in chain:
        card(ax, x, y, w, h, title, body, edge=color, fill=fill, title_size=16, body_size=11.5)
    for a, b in zip(chain[:-1], chain[1:]):
        arrow(ax, (a[0] + a[1] + 0.06, y + h / 2), (b[0] - 0.06, y + h / 2))
    label(ax, 14.7, 6.64, "已有 y3\n即可训练\n第4轮", 12, GREY)
    label(ax, 0.6, 5.82, "② 各轮状态 → 监督样本", 14, DATA, True, ha="left")
    centers = [5.075, 9.125, 13.175]
    state_y, state_h, state_w = 4.45, 1.03, 3.6
    for center, k in zip(centers, [2, 3, 4]):
        card(ax, center - state_w / 2, state_y, state_w, state_h,
             "第" + str(k) + "轮训练样本",
             "输入 y" + str(k - 1) + " + 原LR x\n轮次 k=" + str(k) + "；监督目标 HR",
             edge=DATA, fill="#FFF7E8", title_size=15.5, body_size=12)
        arrow(ax, (center, y - 0.05), (center, state_y + state_h + 0.08), DATA)
    new_x, new_y, new_w, new_h = 5.0, 2.25, 4.25, 1.45
    new_center = new_x + new_w / 2
    for center in centers:
        line(ax, [(center, state_y - 0.04), (center, 4.04)], DATA)
    line(ax, [(centers[0], 4.04), (centers[-1], 4.04)], DATA)
    arrow(ax, (new_center, 4.04), (new_center, new_y + new_h + 0.07), DATA)
    label(ax, 0.6, 4.10, "batch 混合多个轮次\n可加入历史状态", 12.5, DATA, ha="left")
    label(ax, 0.6, 3.30, "③ 梯度学习\n只更新 new", 15, NEW, True, ha="left")
    card(ax, new_x, new_y, new_w, new_h, "共享 G_new",
         "参数从 old 初始化\n学习各轮输入状态的纠正", edge=NEW, fill="#ECF8F2")
    card(ax, 9.95, new_y, 2.1, new_h, "训练预测",
         "第2 / 3 / 4轮\n重建预测", edge=NEW, fill="#ECF8F2", title_size=16, body_size=13)
    card(ax, 12.75, new_y, 2.45, new_h, "监督 Loss",
         "FM / latent / 图像\n目标来自配对 HR", edge=HR, fill="#F5EFFA", title_size=16, body_size=12)
    arrow(ax, (new_x + new_w + 0.10, new_y + new_h / 2), (9.85, new_y + new_h / 2), NEW)
    arrow(ax, (12.15, new_y + new_h / 2), (12.65, new_y + new_h / 2), NEW)
    card(ax, 12.75, 0.70, 2.45, 0.80, "HR：真值目标", edge=HR, fill="#F5EFFA", title_size=14)
    arrow(ax, (13.975, 1.57), (13.975, new_y - 0.08), HR)
    line(ax, [(13.35, new_y - 0.04), (13.35, 1.90), (new_center, 1.90)], NEW, dashed=True)
    arrow(ax, (new_center, 1.90), (new_center, new_y - 0.04), NEW, dashed=True)
    label(ax, 10.3, 1.64, "反向传播只到 new", 12, NEW)
    card(ax, 0.6, 0.70, 10.75, 0.77,
         "old 的输出是输入状态，HR 才是监督目标；不跨整条采集链反传",
         edge=GREY, fill=BACKGROUND, title_size=13.5)
    return fig


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    figures = [("01_training_outer_loop", outer_loop()),
               ("02_rollout_and_supervision", inner_details())]
    with PdfPages(ROOT / "old_new_sr_workflow.pdf") as pdf:
        for name, fig in figures:
            fig.savefig(ROOT / (name + ".png"), dpi=150, facecolor="white")
            fig.savefig(ROOT / (name + ".svg"), facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    print("Saved 2 PNGs, 2 SVGs and a 2-page PDF to " + str(ROOT))


if __name__ == "__main__":
    main()
