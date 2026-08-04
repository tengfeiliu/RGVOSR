from __future__ import annotations

import math
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUT_DIR = Path(__file__).resolve().parent
FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BG = "#F7F9FC"
INK = "#172033"
MUTED = "#526079"
LINE = "#70809A"
BLUE_FILL = "#E7F0FF"
BLUE = "#4777C4"
GREEN_FILL = "#E5F6EC"
GREEN = "#3F9565"
ORANGE_FILL = "#FFF1DD"
ORANGE = "#CC7A20"
PURPLE_FILL = "#EEE8FF"
PURPLE = "#7658C2"
GRAY_FILL = "#EEF2F7"
GRAY = "#778398"
RED_FILL = "#FDEBEC"
RED = "#C55A64"
WHITE = "#FFFFFF"


class Canvas:
    def __init__(self, width: int, height: int, title: str):
        self.width = width
        self.height = height
        self.title = title
        self.image = Image.new("RGB", (width, height), BG)
        self.draw = ImageDraw.Draw(self.image)
        self.svg: list[str] = []

    @staticmethod
    def font(size: int, bold: bool = False):
        return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)

    def rounded_rect(self, xy, fill, outline, width=3, radius=24):
        x1, y1, x2, y2 = xy
        self.draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)
        self.svg.append(
            f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
            f'rx="{radius}" fill="{fill}" stroke="{outline}" stroke-width="{width}"/>'
        )

    def line(self, points, fill=LINE, width=5, dashed=False):
        if dashed:
            for p1, p2 in zip(points[:-1], points[1:]):
                self._dashed_segment(p1, p2, fill, width)
            dash = ' stroke-dasharray="14 10"'
        else:
            self.draw.line(points, fill=fill, width=width, joint="curve")
            dash = ""
        pts = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{fill}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"{dash}/>'
        )

    def _dashed_segment(self, p1, p2, fill, width, dash=14, gap=10):
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        ux, uy = dx / dist, dy / dist
        pos = 0.0
        while pos < dist:
            end = min(pos + dash, dist)
            self.draw.line(
                [(x1 + ux * pos, y1 + uy * pos), (x1 + ux * end, y1 + uy * end)],
                fill=fill,
                width=width,
            )
            pos += dash + gap

    def arrow(self, points, fill=LINE, width=5, head=18, dashed=False):
        self.line(points, fill=fill, width=width, dashed=dashed)
        (x1, y1), (x2, y2) = points[-2], points[-1]
        angle = math.atan2(y2 - y1, x2 - x1)
        left = (
            x2 - head * math.cos(angle) + head * 0.55 * math.sin(angle),
            y2 - head * math.sin(angle) - head * 0.55 * math.cos(angle),
        )
        right = (
            x2 - head * math.cos(angle) - head * 0.55 * math.sin(angle),
            y2 - head * math.sin(angle) + head * 0.55 * math.cos(angle),
        )
        poly = [(x2, y2), left, right]
        self.draw.polygon(poly, fill=fill)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in poly)
        self.svg.append(f'<polygon points="{pts}" fill="{fill}"/>')

    def text_line(self, xy, text, size=34, color=INK, bold=False, anchor="mm"):
        x, y = xy
        font = self.font(size, bold)
        self.draw.text((x, y), text, font=font, fill=color, anchor=anchor)
        weight = 500 if bold else 400
        svg_anchor = {"mm": "middle", "lm": "start", "rm": "end"}.get(anchor, "middle")
        self.svg.append(
            f'<text x="{x}" y="{y}" fill="{color}" font-family="Microsoft YaHei, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" text-anchor="{svg_anchor}" '
            f'dominant-baseline="middle">{escape(text)}</text>'
        )

    def text_block(self, xy, lines, size=32, color=INK, bold_first=False, spacing=12):
        x1, y1, x2, y2 = xy
        prepared = []
        for idx, line in enumerate(lines):
            prepared.append((line, self.font(size, bold_first and idx == 0), bold_first and idx == 0))
        heights = []
        for line, font, _ in prepared:
            bbox = self.draw.textbbox((0, 0), line, font=font)
            heights.append(bbox[3] - bbox[1])
        total = sum(heights) + spacing * max(0, len(lines) - 1)
        y = y1 + (y2 - y1 - total) / 2
        for (line, font, is_bold), h in zip(prepared, heights):
            self.draw.text(((x1 + x2) / 2, y + h / 2), line, font=font, fill=color, anchor="mm")
            weight = 500 if is_bold else 400
            self.svg.append(
                f'<text x="{(x1+x2)/2}" y="{y+h/2}" fill="{color}" '
                f'font-family="Microsoft YaHei, sans-serif" font-size="{size}" font-weight="{weight}" '
                f'text-anchor="middle" dominant-baseline="middle">{escape(line)}</text>'
            )
            y += h + spacing

    def box(self, xy, lines, fill=WHITE, outline=GRAY, width=3, radius=22, size=30, bold_first=True):
        self.rounded_rect(xy, fill, outline, width, radius)
        self.text_block(xy, lines, size=size, bold_first=bold_first)

    def section(self, xy, title, fill, outline):
        self.rounded_rect(xy, fill, outline, width=3, radius=30)
        x1, y1, _, _ = xy
        pill_w = max(360, 32 * len(title))
        self.rounded_rect((x1 + 26, y1 + 20, x1 + 26 + pill_w, y1 + 78), WHITE, outline, width=2, radius=20)
        self.text_line((x1 + 26 + pill_w / 2, y1 + 49), title, size=31, bold=True)

    def save(self, stem: str):
        png_path = OUT_DIR / f"{stem}.png"
        svg_path = OUT_DIR / f"{stem}.svg"
        self.image.save(png_path, dpi=(200, 200), optimize=True)
        content = "\n".join(self.svg)
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" '
            f'viewBox="0 0 {self.width} {self.height}">\n'
            f'<title>{escape(self.title)}</title>\n'
            f'<rect width="100%" height="100%" fill="{BG}"/>\n{content}\n</svg>\n'
        )
        svg_path.write_text(svg, encoding="utf-8")
        return png_path, svg_path


def overall_architecture():
    c = Canvas(3400, 2350, "当前代码中的 LoRA-MoE 总体架构")
    c.text_line((1700, 72), "当前代码中的 LoRA-MoE 总体架构", size=68, bold=True)
    c.text_line(
        (1700, 132),
        "Single-LoRA 初始化  ·  Prompt/LR 全局路由  ·  Shared LoRA + Top-2 Routed Experts",
        size=32,
        color=MUTED,
    )

    # Stage 1 initialization
    c.section((70, 185, 3330, 680), "阶段一：从 Single-LoRA 初始化 MoE", BLUE_FILL, BLUE)
    c.box((140, 320, 680, 560), ["Single-LoRA checkpoint", "A_single，B_single"], WHITE, BLUE, size=33)
    c.box((820, 320, 1390, 560), ["Shared LoRA", "A_s ← A_single", "B_s ← B_single"], WHITE, GREEN, size=33)
    c.box(
        (1530, 295, 2240, 585),
        ["4 个 Routed Experts", "A_e ← A_s + 0.01 × 噪声", "B_e ← 0", "初始 routed 输出为 0"],
        WHITE,
        ORANGE,
        size=31,
    )
    c.box(
        (2400, 275, 3250, 605),
        ["Router prototypes", "128 个训练样本", "提取 Prompt/LR Router 特征", "K-Means → 4 个中心"],
        WHITE,
        PURPLE,
        size=31,
    )
    c.arrow([(680, 440), (820, 440)], fill=BLUE)
    c.arrow([(1390, 440), (1530, 440)], fill=BLUE)

    # Input column
    c.section((70, 750, 770, 1810), "输入与条件编码", GRAY_FILL, GRAY)
    c.box((135, 880, 705, 1025), ["Prompt", "Caption + IQA + Suggestion"], WHITE, PURPLE, size=29)
    c.box((135, 1100, 705, 1245), ["LR 全图／512 crop", "推理／训练输入"], WHITE, BLUE, size=29)
    c.box((135, 1320, 705, 1465), ["HR 512 crop", "仅训练时使用"], WHITE, GREEN, size=29)
    c.box((135, 1535, 705, 1710), ["编码结果", "Prompt embeddings", "z_lr 及训练时的 z_hr、z_t"], WHITE, GRAY, size=28)
    c.arrow([(420, 1025), (420, 1535)], fill=PURPLE)
    c.arrow([(420, 1245), (420, 1535)], fill=BLUE)
    c.arrow([(420, 1465), (420, 1535)], fill=GREEN)

    # Router column
    c.section((850, 750, 1870, 1810), "全局 ProfileLatentRouter", PURPLE_FILL, PURPLE)
    c.box((915, 875, 1345, 1045), ["Prompt 分支", "序列维 Mean Pool", "无 attention mask"], WHITE, PURPLE, size=28)
    c.box((1390, 875, 1805, 1045), ["LR 统计分支", "mean / std", "min / max"], WHITE, BLUE, size=28)
    c.box((1390, 1100, 1805, 1270), ["LR Conv 分支", "Conv2d", "全局池化"], WHITE, BLUE, size=28)
    c.box((915, 1100, 1345, 1270), ["融合 MLP", "得到全局特征 f"], WHITE, PURPLE, size=29)
    c.box((915, 1340, 1805, 1505), ["Router logits", "Linear Head(f) + cosine(f, prototypes)"], WHITE, PURPLE, size=29)
    c.box((915, 1570, 1805, 1735), ["路由调度 → α ∈ R^(B×4)", "Warmup：4-way Softmax  ·  后期：Top-2", "Temperature：2.0 → 0.7"], WHITE, ORANGE, size=28)
    c.arrow([(1130, 1045), (1130, 1100)], fill=PURPLE)
    c.arrow([(1598, 1045), (1598, 1100)], fill=BLUE)
    c.arrow([(1598, 1270), (1598, 1305), (1360, 1305), (1360, 1340)], fill=BLUE)
    c.arrow([(1130, 1270), (1130, 1340)], fill=PURPLE)
    c.arrow([(1360, 1505), (1360, 1570)], fill=PURPLE)
    c.arrow([(2825, 605), (2825, 705), (1760, 705), (1760, 1340)], fill=PURPLE, dashed=True)

    # Inputs to router
    c.arrow([(705, 952), (810, 952), (810, 960), (915, 960)], fill=PURPLE)
    c.arrow([(705, 1172), (815, 1172), (815, 1070), (1598, 1070), (1598, 1045)], fill=BLUE)
    c.arrow([(705, 1172), (825, 1172), (825, 1295), (1598, 1295), (1598, 1270)], fill=BLUE)

    # Flux column
    c.section((1950, 750, 3330, 1810), "Frozen FLUX.2-Klein Transformer", GREEN_FILL, GREEN)
    c.box((2025, 875, 3255, 1035), ["Transformer 条件输入", "z_t + LR condition tokens + Prompt embeddings"], WHITE, GREEN, size=31)
    c.box((2025, 1100, 3255, 1245), ["Transformer Blocks", "Attention + FFN；基础权重保持冻结"], WHITE, GREEN, size=31)
    c.box(
        (2025, 1315, 3255, 1545),
        ["目标 Attention Linear 被替换", "SharedRoutedMoELoRALinear", "y = W0 x + ΔW_shared x + Σ a_e ΔW_e x"],
        WHITE,
        ORANGE,
        size=31,
    )
    c.box((2025, 1615, 3255, 1745), ["输出", "Flow / Velocity prediction"], WHITE, GREEN, size=30)
    c.arrow([(2640, 1035), (2640, 1100)], fill=GREEN)
    c.arrow([(2640, 1245), (2640, 1315)], fill=GREEN)
    c.arrow([(2640, 1545), (2640, 1615)], fill=GREEN)
    c.arrow([(705, 1620), (790, 1620), (790, 715), (2640, 715), (2640, 875)], fill=GREEN)

    # Router broadcast note and connection
    c.rounded_rect((2110, 1258, 3170, 1305), ORANGE_FILL, ORANGE, width=3, radius=18)
    c.text_line((2640, 1282), "同一组 α 广播到所有层、所有 token 和所有空间位置", size=25, bold=True)
    c.arrow([(1805, 1652), (1905, 1652), (1905, 1282), (2110, 1282)], fill=ORANGE, width=7)

    # Losses
    c.section((70, 1890, 3330, 2275), "训练目标与可训练参数", RED_FILL, RED)
    c.box((150, 2010, 1030, 2195), ["SR 主损失", "Flow Matching + Latent + Charbonnier", "Downsample consistency"], WHITE, RED, size=29)
    c.box((1260, 2010, 2140, 2195), ["MoE 辅助损失", "Router balance + entropy", "Expert A/B parameter diversity"], WHITE, PURPLE, size=29)
    c.box((2370, 2010, 3250, 2195), ["参数更新范围", "Shared LoRA + 4 Routed LoRAs + Router", "FLUX 基础权重与 VAE/Text Encoder 冻结"], WHITE, GREEN, size=29)
    c.arrow([(2640, 1745), (2640, 1850), (590, 1850), (590, 2010)], fill=RED)
    c.arrow([(1360, 1735), (1360, 2010)], fill=PURPLE)

    return c.save("lora_moe_overall_architecture")


def layer_architecture():
    c = Canvas(3400, 2050, "SharedRoutedMoELoRALinear 单层结构")
    c.text_line((1700, 75), "SharedRoutedMoELoRALinear：单层结构放大", size=68, bold=True)
    c.text_line(
        (1700, 138),
        "一个冻结 Base Linear + 一个始终启用的 Shared LoRA + 四个由全局 Router 加权的 Routed Experts",
        size=32,
        color=MUTED,
    )

    c.box((90, 810, 430, 1050), ["输入特征 x", "B × Tokens × D"], WHITE, INK, size=34)

    # Base branch
    c.section((540, 235, 2670, 590), "冻结基础分支", GRAY_FILL, GRAY)
    c.box((660, 350, 1330, 510), ["原始 Linear", "W0：冻结参数"], WHITE, GRAY, size=31)
    c.box((1750, 350, 2460, 510), ["基础输出", "y_base = W0 x"], WHITE, GRAY, size=31)
    c.arrow([(1330, 430), (1750, 430)], fill=GRAY)

    # Shared branch
    c.section((540, 660, 2670, 1045), "Shared LoRA：所有样本始终启用", GREEN_FILL, GREEN)
    c.box((625, 785, 1085, 935), ["LoRA A_s", "D → Rank 32"], WHITE, GREEN, size=30)
    c.box((1250, 785, 1710, 935), ["LoRA B_s", "Rank 32 → D"], WHITE, GREEN, size=30)
    c.box((1880, 785, 2540, 935), ["共享残差", "Δy_s = scale · B_s A_s x"], WHITE, GREEN, size=30)
    c.arrow([(1085, 860), (1250, 860)], fill=GREEN)
    c.arrow([(1710, 860), (1880, 860)], fill=GREEN)

    # Experts
    c.section((540, 1115, 2670, 1715), "Routed LoRA：4 个专家，训练后期 Top-2 非零", ORANGE_FILL, ORANGE)
    expert_x = [625, 1090, 1555, 2020]
    labels = ["Expert 1", "Expert 2", "Expert 3", "Expert 4"]
    for idx, (x, label) in enumerate(zip(expert_x, labels), start=1):
        c.box((x, 1240, x + 390, 1410), [label, f"B_{idx} A_{idx} x"], WHITE, ORANGE, size=29)
    c.box((1080, 1500, 2130, 1645), ["专家加权混合", "Δy_r = scale · Σ a_e B_e A_e x"], WHITE, PURPLE, size=31)
    centers = [x + 195 for x in expert_x]
    for cx in centers:
        c.arrow([(cx, 1410), (cx, 1460), (1605, 1460), (1605, 1500)], fill=ORANGE, width=4)

    # Router
    c.box(
        (2750, 1115, 3310, 1435),
        ["全局 Router", "α = [a1, a2, a3, a4]", "一张图一组权重", "广播到全部 token / 层", "后期仅 Top-2 非零"],
        PURPLE_FILL,
        PURPLE,
        size=29,
    )
    c.arrow([(2750, 1330), (2590, 1330), (2590, 1572), (2130, 1572)], fill=PURPLE, width=7)

    # Input split
    c.arrow([(430, 930), (485, 930), (485, 430), (660, 430)], fill=GRAY)
    c.arrow([(430, 930), (500, 930), (500, 860), (625, 860)], fill=GREEN)
    c.arrow([(430, 930), (515, 930), (515, 1325), (625, 1325)], fill=ORANGE)

    # Final sum/output
    c.box((2750, 545, 3310, 880), ["残差求和", "y = y_base + Δy_s + Δy_r", "", "LoRA scale = α_lora / rank"], BLUE_FILL, BLUE, size=31)
    c.box((2855, 200, 3205, 390), ["最终输出 y"], WHITE, BLUE, size=36)
    c.arrow([(2460, 430), (2720, 430), (2720, 650), (2750, 650)], fill=GRAY)
    c.arrow([(2540, 860), (2675, 860), (2675, 760), (2750, 760)], fill=GREEN)
    c.arrow([(2130, 1572), (2700, 1572), (2700, 830), (2750, 830)], fill=ORANGE)
    c.arrow([(3030, 545), (3030, 390)], fill=BLUE, width=7)

    # Initialization note
    c.section((90, 1800, 3310, 1970), "当前初始化保证初始函数不变", BLUE_FILL, BLUE)
    c.text_line(
        (1700, 1895),
        "Shared 继承 Single-LoRA；A_e = A_s + 0.01 × noise，B_e = 0，因此训练开始时 Routed Experts 的输出严格为 0",
        size=31,
        color=INK,
    )

    return c.save("lora_moe_layer_detail")


if __name__ == "__main__":
    for paths in (overall_architecture(), layer_architecture()):
        print(" | ".join(str(path) for path in paths))
