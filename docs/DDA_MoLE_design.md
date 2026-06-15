# DDA-MoLE: Decomposed Dual-Axis Mixture-of-LoRA Experts for FLUX.2-klein SR

本设计文档面向 Codex / 实现工程师。读完本文档应当能直接落地代码，无需再额外推断设计意图。
所有路径均以仓库根 `d:/work/code/RGVOSR/` 为基准。

---

## 0. TL;DR

在现有 `Flux2KleinSRArtist`（`models/flux2_klein_sr_artist.py`）的 LoRA 训练链路上，把"单一 LoRA"替换为"**两组正交的 LoRA 专家**"：

- **退化轴** (Degradation Axis)：`K_d = 4` 个 LoRA expert，由 **profile 文本** 路由（不使用 `degradation_vector`）。
- **场景轴** (Scene Axis)：`K_s = 4` 个 LoRA expert，由 **LR latent** 路由（场景簇通过 K-means + EM 自发现）。

主干 FLUX.2 transformer 保持冻结，每层 LoRA 权重在 forward 期间动态合成：

```
ΔW(x) = Σ_i g_d_i(text) · ΔW_d_i  +  Σ_j g_s_j(z_lr) · ΔW_s_j
W_eff = W_frozen + ΔW(x)
```

加 3 个 routing 损失（text-contrastive、scene-EM-soft-CE、load-balance），加 1 个 flow-matching 损失。
本文档定义所有 module 接口、文件路径、config 字段、训练 hook 顺序。

---

## 1. 与现有 baseline 的差异速览

| 现状（文件） | 改动 |
|---|---|
| `models/flux2_klein_sr_artist.py::Flux2KleinSRArtist._apply_lora()` | 改造为创建 `K_d + K_s` 个 LoRA adapter（命名 `deg_i`, `scn_j`），并注册一个 `MoLERouterRegistry`。 |
| `models/flux2_klein_sr_artist.py::Flux2KleinSRArtist.forward()` | forward 前调用 `set_runtime_routing(g_d, g_s)`；推理结束 reset。 |
| `models/flux2_klein_sr_artist.py::Flux2KleinSRArtist._build_condition_modules()` | 新增 `self.deg_router`、`self.scene_router`、`self.text_router_encoder`。 |
| 新文件 `models/mole_lora.py` | 多 adapter LoRA 权重合成 layer（核心）。 |
| 新文件 `models/profile_text_router.py` | profile 文本 → router 嵌入；T5 句向量 pooling。 |
| 新文件 `models/scene_router.py` | LR latent → softmax weights，含 EM K-means 更新 hook。 |
| 新文件 `models/mole_losses.py` | routing 三类 loss 实现。 |
| `dataloaders/rg_flux_jsonl_dataset.py` | `__getitem__` 增加返回 `raw_profile_text`（cleaned IQA/IAA 字符串拼接，不含 system instructions），不影响现有 prompt 字段。 |
| `train_rg_flux_sr.py` | 在主 training step 中拼接 routing 损失；在主进程上每 N step 触发 `scene_router.update_centers()`。 |
| `configs/train_rg_flux2_klein_sr_smoke_256.yaml` | 新增 `mole:` 节，详见 §10。 |

不动主干，不动 ZeRO-3 分片逻辑，不动 VAE / text encoder 路径。

---

## 2. 数据流总图

```
batch = {
  "hq", "lq_up", "prompt", "raw_profile_text", ... (现有 + 一个新字段)
}
                                │
                                ▼
            ┌────────────────── Flux2KleinSRArtist.forward ──────────────────┐
            │                                                                 │
   z_hr, z_lr ← VAE.encode (frozen)                                            │
   prompt_embeds, text_ids ← text_pipeline.encode_prompt (frozen)              │
                                                                               │
   ┌── profile_text ─► T5SentenceEncoder (frozen) ─► text_emb [B, D_t]         │
   │                                                                           │
   ├── z_lr ──► SceneRouter.pool ─► scene_feat [B, D_s] ─► softmax g_s [B,K_s] │
   │                                                                           │
   └── text_emb ─► DegradationRouter ─► softmax g_d [B, K_d]                   │
                                                                               │
   set_runtime_routing(g_d, g_s)                                               │
                                                                               │
   z_t (noisy latent) ─► transformer (LoRA 合成被 MoLERouterRegistry hijack)   │
                                                                               │
   v_pred ──► loss_FM = MSE(v_pred, v_target)                                  │
   text_emb + g_d ─► L_route_text                                              │
   z_lr + g_s ──► L_route_scene (软标签来自 K-means cluster)                   │
   g_d, g_s ──► L_balance                                                      │
            └────────────────────────────────────────────────────────────────┘
```

---

## 3. 新文件 module 设计

### 3.1 `models/mole_lora.py`（核心 - 多 LoRA 加权合成）

#### 3.1.1 设计原则

- **peft 的多 adapter** 默认是 "set_adapter(name)" 一次选一个，**不支持**在 forward 内做 **per-batch / per-sample 加权**。
- 我们要在 `Linear` 层上做：`W_eff = W + Σ_k g_k · (A_k @ B_k)`，其中 `g_k` 是 `[B]` 形 routing 权重。
- 实现路径：自定义一个 `MoLELoraLinear`，**包装** 原始 `Linear`，内部持有 `K` 组 `(A, B)` 参数；forward 时用全局 registry 读取当前 routing。
- 在 `_apply_lora()` 中遍历 transformer 的 LoRA target modules（沿用现有 `_resolve_lora_targets()`），用 `MoLELoraLinear` **手工替换**对应 `nn.Linear`，绕开 `peft.PeftModel`。
- 这样保持 ZeRO-3 兼容（只新增 `nn.Parameter`），并且 save / load 完全可控。

#### 3.1.2 接口

```python
# models/mole_lora.py

class MoLERouterRegistry:
    """Process-wide singleton holding the current batch's routing weights.

    Forward thread of MoLELoraLinear reads from here. Cleared on context exit.
    """
    _current: dict | None = None

    @classmethod
    def set(cls, g_deg: torch.Tensor, g_scene: torch.Tensor) -> None: ...
    @classmethod
    def get(cls) -> dict | None: ...  # {"g_deg": [B,K_d], "g_scene": [B,K_s]}
    @classmethod
    def clear(cls) -> None: ...

    @classmethod
    @contextlib.contextmanager
    def routing(cls, g_deg, g_scene): ...   # convenient `with` block

class MoLELoraLinear(nn.Module):
    """Wraps an nn.Linear; injects MoE LoRA delta read from MoLERouterRegistry."""

    def __init__(
        self,
        base_linear: nn.Linear,
        num_deg_experts: int,
        num_scene_experts: int,
        rank: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ):
        super().__init__()
        self.base = base_linear           # frozen, requires_grad_(False)
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.K_d = num_deg_experts
        self.K_s = num_scene_experts
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        # LoRA A: [K, rank, in], B: [K, out, rank]
        self.deg_A = nn.Parameter(torch.zeros(self.K_d, rank, self.in_features))
        self.deg_B = nn.Parameter(torch.zeros(self.K_d, self.out_features, rank))
        self.scn_A = nn.Parameter(torch.zeros(self.K_s, rank, self.in_features))
        self.scn_B = nn.Parameter(torch.zeros(self.K_s, self.out_features, rank))
        # init A ~ Gaussian(0, init_scale); B = 0 (so initial delta is 0)
        nn.init.normal_(self.deg_A, std=init_scale)
        nn.init.normal_(self.scn_A, std=init_scale)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        routing = MoLERouterRegistry.get()
        if routing is None:
            return base_out
        g_d = routing["g_deg"]      # [B, K_d]
        g_s = routing["g_scene"]    # [B, K_s]
        # x can be [B, L, in] or [B, in]; treat per-sample weight
        delta_d = self._mole_delta(x, g_d, self.deg_A, self.deg_B)
        delta_s = self._mole_delta(x, g_s, self.scn_A, self.scn_B)
        return base_out + self.scaling * (delta_d + delta_s)

    @staticmethod
    def _mole_delta(x, g, A, B):
        # x: [B, L, in] or [B, in]; g: [B, K]; A: [K, r, in]; B: [K, out, r]
        # 1) project: per-expert AB: out_k = (x @ A_k^T) @ B_k^T
        # 2) weighted sum over K with g
        x_has_seq = x.dim() == 3
        if not x_has_seq:
            x = x.unsqueeze(1)
        # einsum-friendly path
        # x: [B, L, in], A: [K, r, in] -> z: [B, K, L, r]
        z = torch.einsum("bli,kri->bklr", x.to(A.dtype), A)
        # z: [B, K, L, r], B_w: [K, out, r] -> y: [B, K, L, out]
        y = torch.einsum("bklr,kor->bklo", z, B)
        # g: [B, K] -> [B, K, 1, 1]
        y = (y * g.to(y.dtype).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
        if not x_has_seq:
            y = y.squeeze(1)
        return self.dropout(y)
```

**实现注意**

- 用 `einsum` 是为了易读；若实测显存爆，可以改成 `(x @ A_k^T) @ B_k^T` 循环 + `torch.compile`，**但 K=4 时 einsum 完全足够**。
- `init_scale` 必须保证训练开头 delta ≈ 0，避免一上来扰乱 baseline 已能收敛的状态。
- 在 ZeRO-3 下，`deg_A / deg_B / scn_A / scn_B` 会自动分片。

#### 3.1.3 替换函数

```python
def convert_linear_to_mole(
    module: nn.Module,
    target_suffixes: tuple[str, ...],
    num_deg_experts: int,
    num_scene_experts: int,
    rank: int,
    alpha: int,
    dropout: float,
) -> list[str]:
    """In-place replace nn.Linear submodules whose qualified name ends with any suffix.

    Returns the list of replaced qualified names (for debug/log).
    """
    replaced = []
    for name, child in module.named_modules():
        if not isinstance(child, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        # Set base linear frozen
        child.requires_grad_(False)
        wrapped = MoLELoraLinear(child, num_deg_experts, num_scene_experts, rank, alpha, dropout)
        # navigate and setattr
        parent, leaf = _split_qualified(module, name)
        setattr(parent, leaf, wrapped)
        replaced.append(name)
    return replaced
```

---

### 3.2 `models/profile_text_router.py`（退化轴 Router）

#### 3.2.1 数据准备

- 从 `cleaned profile`（位于 `unipercept_raw.profile`）中**拼接 IQA + IAA + suggestion** 成一个紧凑的"退化描述串"，不含 system instructions。
- 字符串构造规则：

```
"[IQA] blur: ...; noise: ...; jpeg: ...; ringing: ... [IAA] ... [SUGG] ..."
```

- 由 `dataloaders/rg_flux_jsonl_dataset.py` 在 `__getitem__` 中新增 `"raw_profile_text": str`。

#### 3.2.2 编码 backbone

- **不复用** FLUX 的 T5（输出维度太高、需要存全序列）。我们用一个**轻量**句向量模型：`sentence-transformers/all-MiniLM-L6-v2` (384-d) 或 `Qwen2-0.5B` 的最后 hidden state mean-pool。
- 推荐 `MiniLM-L6-v2`：CPU-friendly，可一次性 batch 编码，**完全冻结**。

```python
# models/profile_text_router.py

class FrozenSentenceEncoder(nn.Module):
    """Wraps sentence-transformers MiniLM-L6-v2; outputs [B, 384]."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        super().__init__()
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.out_dim = self.model.get_sentence_embedding_dimension()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> torch.Tensor:
        emb = self.model.encode(texts, convert_to_tensor=True, normalize_embeddings=True)
        return emb.float()  # [B, 384] on self.device


class DegradationRouter(nn.Module):
    """text_emb -> g_d (softmax/top-k) over K_d experts."""

    def __init__(self, in_dim: int, num_experts: int,
                 hidden_dim: int = 256, top_k: int | None = None,
                 temperature: float = 1.0, noisy_topk_std: float = 0.0):
        super().__init__()
        self.K = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.noisy_std = noisy_topk_std
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        # text_emb: [B, in_dim]
        logits = self.net(text_emb)
        if self.training and self.noisy_std > 0:
            logits = logits + torch.randn_like(logits) * self.noisy_std
        logits = logits / max(self.temperature, 1e-4)
        if self.top_k is None or self.top_k >= self.K:
            return F.softmax(logits, dim=-1)
        # noisy top-k (Switch-Transformer style)
        topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        return F.softmax(mask, dim=-1)
```

**关键决策**

- `top_k = 2`（用于退化轴）：兼顾稀疏与表达力，避免 collapse。
- `temperature` 训练初期 = 1.5 (软)，逐渐退火到 1.0；可放 config。

---

### 3.3 `models/scene_router.py`（场景轴 Router + EM K-means）

#### 3.3.1 场景特征

- 用 `z_lr` (FLUX.2 patchified latent，shape `[B, 128, H/16, W/16]`) 全局 mean-pool 得 `[B, 128]`。
- 再过一个 `Linear(128, 256)` MLP 投影到 `scene_feat`。
- **不**用 DINO（避免引入新外部模型）；后续可作为 ablation。

#### 3.3.2 K-means 中心（EM-style）

- 维护 `self.centers ∈ R^{K_s × 256}`，作为 buffer（不参与反向传播）。
- 启动方式：训练前 0 步，全局采样 `N=4096` 样本算一次 KMeans 初始中心（`sklearn.cluster.MiniBatchKMeans`）。
- **EM 更新**：每隔 `update_every` 个 train step（默认 1000），用最近 buffer 中的 scene_feat 重新跑 MiniBatchKMeans，更新 `self.centers`。

#### 3.3.3 软标签生成

```
d_i = ||scene_feat - centers_i||^2
p = softmax(-d / τ)        # τ = 0.5
```

#### 3.3.4 接口

```python
# models/scene_router.py

class SceneRouter(nn.Module):
    def __init__(self, latent_channels: int, num_experts: int,
                 hidden_dim: int = 256, top_k: int | None = None,
                 temperature: float = 1.0, soft_label_temperature: float = 0.5,
                 buffer_size: int = 8192, update_every: int = 1000):
        super().__init__()
        self.K = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.tau_label = soft_label_temperature
        self.update_every = update_every

        self.proj = nn.Sequential(
            nn.Linear(latent_channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.router_head = nn.Linear(hidden_dim, num_experts)

        self.register_buffer("centers", torch.zeros(num_experts, hidden_dim))
        self.register_buffer("centers_inited", torch.zeros(1, dtype=torch.bool))

        # ring buffer for EM refresh
        self.register_buffer("feat_buffer", torch.zeros(buffer_size, hidden_dim))
        self.register_buffer("buffer_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("buffer_filled", torch.zeros(1, dtype=torch.bool))
        self.buffer_size = buffer_size

    # ---- forward ----
    def pool(self, z_lr: torch.Tensor) -> torch.Tensor:
        # z_lr: [B, C, H, W] -> [B, C]
        return z_lr.mean(dim=(-2, -1)).float()

    def forward(self, z_lr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (g_s [B,K], soft_label [B,K], scene_feat [B, hidden])."""
        feat = self.proj(self.pool(z_lr))
        logits = self.router_head(feat) / max(self.temperature, 1e-4)
        if self.top_k is None or self.top_k >= self.K:
            g_s = F.softmax(logits, dim=-1)
        else:
            topk_vals, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            g_s = F.softmax(mask, dim=-1)

        soft_label = self._soft_label(feat)
        # push into buffer (no grad)
        with torch.no_grad():
            self._push_buffer(feat.detach())
        return g_s, soft_label, feat

    # ---- EM ----
    @torch.no_grad()
    def _soft_label(self, feat: torch.Tensor) -> torch.Tensor:
        if not bool(self.centers_inited.item()):
            # before init: uniform label (no scene supervision yet)
            return feat.new_full((feat.shape[0], self.K), 1.0 / self.K)
        d2 = (feat[:, None, :] - self.centers[None, :, :]).pow(2).sum(-1)  # [B, K]
        return F.softmax(-d2 / max(self.tau_label, 1e-4), dim=-1)

    @torch.no_grad()
    def _push_buffer(self, feat: torch.Tensor) -> None:
        n = feat.shape[0]
        ptr = int(self.buffer_ptr.item())
        end = ptr + n
        if end <= self.buffer_size:
            self.feat_buffer[ptr:end] = feat
        else:
            split = self.buffer_size - ptr
            self.feat_buffer[ptr:] = feat[:split]
            self.feat_buffer[: end - self.buffer_size] = feat[split:]
            self.buffer_filled.fill_(True)
        self.buffer_ptr.fill_((end) % self.buffer_size)
        if end >= self.buffer_size:
            self.buffer_filled.fill_(True)

    @torch.no_grad()
    def init_centers(self, n_min: int = 2048) -> bool:
        """Try to initialize centers from buffer; return True on success."""
        size = self.buffer_size if bool(self.buffer_filled.item()) else int(self.buffer_ptr.item())
        if size < n_min:
            return False
        from sklearn.cluster import MiniBatchKMeans
        feats = self.feat_buffer[:size].cpu().numpy()
        km = MiniBatchKMeans(n_clusters=self.K, n_init=4, batch_size=256, random_state=0).fit(feats)
        self.centers.copy_(torch.from_numpy(km.cluster_centers_).to(self.centers))
        self.centers_inited.fill_(True)
        return True

    @torch.no_grad()
    def update_centers(self) -> None:
        size = self.buffer_size if bool(self.buffer_filled.item()) else int(self.buffer_ptr.item())
        if size < self.K * 8:
            return
        from sklearn.cluster import MiniBatchKMeans
        feats = self.feat_buffer[:size].cpu().numpy()
        km = MiniBatchKMeans(
            n_clusters=self.K, n_init=1, batch_size=256, random_state=0,
            init=self.centers.detach().cpu().numpy() if bool(self.centers_inited.item()) else "k-means++",
        ).fit(feats)
        self.centers.copy_(torch.from_numpy(km.cluster_centers_).to(self.centers))
        self.centers_inited.fill_(True)
```

**实现注意**

- `centers` 用 buffer 不参与训练，避免和 ZeRO-3 参数管理冲突。
- KMeans 跑在 CPU；DDP 下只在 **rank 0** 跑，然后 `dist.broadcast` 到其他 rank（由 train loop 负责）。
- `top_k_scene` 推荐 `1`（hard scene routing），让场景 expert 强专门化。

---

### 3.4 `models/mole_losses.py`

```python
# models/mole_losses.py

import torch
import torch.nn.functional as F


def info_nce_text_routing(
    g_d: torch.Tensor,            # [B, K_d] softmax weights
    text_emb: torch.Tensor,       # [B, D_t] normalized
    temperature: float = 0.1,
) -> torch.Tensor:
    """Pull together routing distributions of samples with similar profile texts.

    Idea: cosine-sim of text_emb defines positives. For each anchor, positives are
    top-T closest others in the batch; we minimize JS / KL between their g_d.
    Lightweight implementation: weighted symmetric KL with similarity-derived weights.
    """
    sim = text_emb @ text_emb.t()                           # [B, B]
    sim = sim.masked_fill(torch.eye(sim.size(0), dtype=torch.bool, device=sim.device), float("-inf"))
    weights = F.softmax(sim / temperature, dim=-1)           # [B, B]
    # symmetric KL on routing
    log_p = (g_d.clamp_min(1e-8)).log()
    # E_i [ sum_j w_ij * KL(p_i || p_j) ]
    kl = (g_d.unsqueeze(1) * (log_p.unsqueeze(1) - log_p.unsqueeze(0))).sum(-1)  # [B,B]
    loss = (weights * kl).sum(-1).mean()
    return loss


def scene_soft_ce(
    g_s: torch.Tensor,            # [B, K_s] router output
    soft_label: torch.Tensor,     # [B, K_s] soft labels from EM centers
) -> torch.Tensor:
    log_p = g_s.clamp_min(1e-8).log()
    return -(soft_label * log_p).sum(-1).mean()


def load_balance_loss(g: torch.Tensor) -> torch.Tensor:
    """Switch-Transformer style aux loss.

    L = K * sum_i (f_i * P_i)
    where f_i = fraction of tokens routed to expert i (top-1 vote on g),
          P_i = mean routing prob for expert i.
    """
    # Approximation: use g as "fraction" too (since softmax >= 0 sums to 1).
    P = g.mean(dim=0)
    f = (g == g.max(dim=-1, keepdim=True).values).float().mean(dim=0)
    K = g.size(-1)
    return K * (f * P).sum()
```

---

## 4. 改造 `Flux2KleinSRArtist`

### 4.1 新增字段（`__init__`）

```python
# inside __init__ AFTER _build_condition_modules()
self.use_mole = bool(_cfg(config, "mole.enabled", False))
if self.use_mole:
    self._build_mole_modules()
```

### 4.2 `_build_mole_modules`

```python
def _build_mole_modules(self):
    mole = _cfg(self.config, "mole", {}) or {}
    self.K_d = int(mole.get("num_deg_experts", 4))
    self.K_s = int(mole.get("num_scene_experts", 4))
    self.lora_rank = int(mole.get("lora_rank", 8))
    self.lora_alpha = int(mole.get("lora_alpha", 8))
    self.lora_dropout = float(mole.get("lora_dropout", 0.0))
    self.top_k_d = int(mole.get("top_k_deg", 2))
    self.top_k_s = int(mole.get("top_k_scene", 1))

    # 1) text router
    self.text_router_encoder = FrozenSentenceEncoder(
        model_name=str(mole.get("text_encoder_name", "sentence-transformers/all-MiniLM-L6-v2")),
        device=str(mole.get("text_encoder_device", "cpu")),
    )
    self.deg_router = DegradationRouter(
        in_dim=self.text_router_encoder.out_dim,
        num_experts=self.K_d,
        hidden_dim=int(mole.get("router_hidden_dim", 256)),
        top_k=self.top_k_d,
        temperature=float(mole.get("router_temperature_deg", 1.0)),
        noisy_topk_std=float(mole.get("router_noise_std_deg", 0.3)),
    )

    # 2) scene router
    self.scene_router = SceneRouter(
        latent_channels=self.latent_channels,
        num_experts=self.K_s,
        hidden_dim=int(mole.get("router_hidden_dim", 256)),
        top_k=self.top_k_s,
        temperature=float(mole.get("router_temperature_scene", 1.0)),
        soft_label_temperature=float(mole.get("scene_soft_label_tau", 0.5)),
        buffer_size=int(mole.get("scene_buffer_size", 8192)),
        update_every=int(mole.get("scene_update_every", 1000)),
    )

    # 3) replace LoRA-target Linears with MoLELoraLinear
    target_suffixes = tuple(self._resolve_lora_targets_as_suffixes())
    replaced = convert_linear_to_mole(
        self.transformer,
        target_suffixes=target_suffixes,
        num_deg_experts=self.K_d,
        num_scene_experts=self.K_s,
        rank=self.lora_rank,
        alpha=self.lora_alpha,
        dropout=self.lora_dropout,
    )
    self._mole_replaced_names = replaced
```

> `_resolve_lora_targets_as_suffixes()` 复用现有 [`_resolve_lora_targets`](models/flux2_klein_sr_artist.py#L246) 的 suffix 集合（去掉完整路径，只取最后一段）。

### 4.3 改造 `_apply_lora`

```python
def _apply_lora(self):
    if self.use_mole:
        # MoLE 自己接管 LoRA，不走 peft
        return
    # original peft path (existing code)
    ...
```

### 4.4 `forward` 改造

```python
def forward(
    self,
    z_t,
    timestep,
    prompt_embeds,
    pooled_prompt_embeds=None,
    text_ids=None,
    degradation_vector=None,      # 不再使用，但保留接口兼容
    z_lr=None,
    dino_tokens=None,
    lr_cond_mode=None,
    profile_text=None,            # 新增：[B] list of str
    return_routing: bool = False, # 训练时 True
):
    if not self.use_mole:
        return self._forward_baseline(...)   # 原 forward

    # ---- routing ----
    text_emb = self.text_router_encoder.encode(profile_text)
    text_emb = text_emb.to(device=z_t.device, dtype=torch.float32)
    g_d = self.deg_router(text_emb)                                 # [B, K_d]
    g_s, soft_label, scene_feat = self.scene_router(z_lr)           # [B, K_s], ...

    # ---- transformer forward with MoLE routing ----
    with MoLERouterRegistry.routing(g_d=g_d, g_scene=g_s):
        out = self._forward_baseline_body(
            z_t=z_t,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            degradation_vector=None,            # disabled per user
            z_lr=z_lr,
            dino_tokens=dino_tokens,
            lr_cond_mode=lr_cond_mode,
        )
    if return_routing:
        return out, {
            "g_d": g_d, "g_s": g_s,
            "soft_label": soft_label,
            "text_emb": text_emb,
        }
    return out
```

> 把现有 `forward` 主体抽出为 `_forward_baseline_body(...)`，无逻辑变化，仅是 refactor。

### 4.5 `save_trainable` / `load_trainable`

新增持久化字段：

- `mole_lora_state.pt`：所有 `MoLELoraLinear` 的 `deg_A/deg_B/scn_A/scn_B`。
- `routers_state.pt`：`deg_router.state_dict()` + `scene_router.state_dict()`（含 buffer 中心）。
- 在 `rg_flux_checkpoint_meta.json` 中加 `"mole": {"K_d": ..., "K_s": ..., "rank": ...}`。
- `load_trainable` 校验 K_d/K_s/rank 一致。

---

## 5. Dataset 改造

文件：`dataloaders/rg_flux_jsonl_dataset.py`

#### 5.1 `__getitem__` 新增字段

```python
profile = record["profile"]
raw_profile_text = build_profile_router_text(profile)   # 见 5.2
sample["raw_profile_text"] = raw_profile_text
```

#### 5.2 `build_profile_router_text` (新建于 `models/prompt_builder.py` 末尾)

```python
def build_profile_router_text(profile: dict) -> str:
    """Compact representation for ROUTING ONLY. Not the main FLUX prompt."""
    profile = profile if isinstance(profile, dict) else {}
    iqa = profile.get("iqa", {}) or {}
    iaa = profile.get("iaa", {}) or {}
    sugg = profile.get("suggestion") or ""

    def _flat(d):
        if not isinstance(d, dict):
            return _safe_text(d)
        return "; ".join(f"{k}: {_safe_text(v)}" for k, v in d.items() if _safe_text(v))

    parts = []
    iqa_text = _flat(iqa)
    if iqa_text:
        parts.append(f"[IQA] {iqa_text}")
    comp = _safe_text(iaa.get("comprehensive") if isinstance(iaa, dict) else None)
    if comp:
        parts.append(f"[IAA] {comp}")
    sugg_text = _safe_text(sugg)
    if sugg_text:
        parts.append(f"[SUGG] {sugg_text}")
    return " ".join(parts) or "no profile"
```

#### 5.3 `rg_flux_collate_fn`

加：`collated["raw_profile_text"] = [item["raw_profile_text"] for item in batch]`

---

## 6. 训练脚本改造（`train_rg_flux_sr.py`）

### 6.1 主循环改动（伪代码）

```python
# inside the train step
unwrapped_artist = accelerator.unwrap_model(artist)

with torch.no_grad():
    z_hr = unwrapped_artist.encode_images(hq)
    z_lr = unwrapped_artist.encode_images(lq_up)
    prompt_embeds, pooled_prompt_embeds, text_ids = unwrapped_artist.encode_prompts(prompts, ...)
    sigma = sample_sigma(...)
    eps = torch.randn_like(z_hr)
    z_t, v_target = build_flow_matching_inputs(z_hr, eps=eps, sigma=sigma)

with accelerator.accumulate(artist):
    with accelerator.autocast():
        v_pred, routing_pkg = artist(
            z_t=z_t,
            timestep=sigma,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            degradation_vector=None,
            z_lr=z_lr,
            dino_tokens=None,
            lr_cond_mode=lr_cond_mode,
            profile_text=batch["raw_profile_text"],
            return_routing=True,
        )
        loss_fm = F.mse_loss(v_pred.float(), v_target.float())

        # routing losses
        g_d = routing_pkg["g_d"]
        g_s = routing_pkg["g_s"]
        l_route_text = info_nce_text_routing(g_d, routing_pkg["text_emb"])
        l_route_scene = scene_soft_ce(g_s, routing_pkg["soft_label"])
        l_balance = load_balance_loss(g_d) + load_balance_loss(g_s)

        loss = (
            fm_weight * loss_fm
            + lam_text * l_route_text
            + lam_scene * l_route_scene
            + lam_bal * l_balance
        )
    accelerator.backward(loss)
    ...
```

### 6.2 Scene Router warmup / EM 更新（rank 0 only）

```python
# in train loop
if accelerator.is_main_process and global_step == int(cfg(config, "mole.scene_warmup_steps", 200)):
    unwrapped_artist.scene_router.init_centers()
    # broadcast to other ranks
if accelerator.is_main_process and global_step > 0 \
   and global_step % unwrapped_artist.scene_router.update_every == 0:
    unwrapped_artist.scene_router.update_centers()
# always broadcast centers buffer
torch.distributed.broadcast(unwrapped_artist.scene_router.centers, src=0)
torch.distributed.broadcast(unwrapped_artist.scene_router.centers_inited.float(), src=0)
```

注：warmup 之前 `soft_label` 用 uniform，自动绕过 scene loss 强约束（loss 仍可计算，但 gradient 不会推动 g_s 偏离 uniform）。可选：warmup 期间将 `lam_scene` 置 0。

### 6.3 Optimizer param groups

```python
mole_params = [p for n, p in artist.named_parameters()
               if p.requires_grad and ("deg_A" in n or "deg_B" in n or "scn_A" in n or "scn_B" in n)]
router_params = [p for n, p in artist.named_parameters()
                 if p.requires_grad and ("deg_router" in n or "scene_router" in n)]
adapter_params = [p for n, p in artist.named_parameters()
                  if p.requires_grad and not any(k in n for k in
                  ["deg_A","deg_B","scn_A","scn_B","deg_router","scene_router"])]
param_groups = [
    {"params": adapter_params, "lr": lr_adapter},
    {"params": mole_params, "lr": lr_lora},
    {"params": router_params, "lr": lr_router},   # 推荐 1e-4
]
```

---

## 7. Inference 改造（`inference_rg_flux_sr.py`）

唯一需要的修改：
- 在调用 `sample_multistep_fm` 之前，把 `profile_text` 通过 `artist.text_router_encoder.encode(...)` 计算一次 `text_emb`，并把 forward 接口加 `profile_text` 参数。
- `sample_multistep_fm` 在 `rg_flux_fm.py` 内的 forward 调用增加 `profile_text=profile_text`。
- 推理时**关闭** scene router 的 EM 更新（`unwrapped_artist.scene_router.eval()`），不要 push buffer（在 forward 中加 `if self.training:` 守卫）。

---

## 8. Checkpoint 兼容矩阵

| 模式 | 加载 baseline ckpt | 加载 MoLE ckpt |
|---|---|---|
| baseline | ✅ | ❌（不兼容） |
| MoLE | ⚠️ 仅加载 condition adapters；LoRA 部分跳过 | ✅ |

`load_trainable` 用 `mole_lora_state.pt` 是否存在来判断模式。

---

## 9. 推荐超参（写进 config 默认）

```yaml
mole:
  enabled: true
  num_deg_experts: 4
  num_scene_experts: 4
  lora_rank: 8
  lora_alpha: 8
  lora_dropout: 0.0
  top_k_deg: 2
  top_k_scene: 1
  router_hidden_dim: 256

  text_encoder_name: sentence-transformers/all-MiniLM-L6-v2
  text_encoder_device: cpu
  router_temperature_deg: 1.0
  router_noise_std_deg: 0.3

  router_temperature_scene: 1.0
  scene_soft_label_tau: 0.5
  scene_buffer_size: 8192
  scene_update_every: 1000
  scene_warmup_steps: 200

loss:
  fm_weight: 1.0
  lam_route_text: 0.05
  lam_route_scene: 0.05
  lam_route_balance: 0.01
  scene_warmup_steps_zero_lambda: 200   # 与 mole.scene_warmup_steps 对齐
```

**经验值**

- 三个 routing loss 都很小（0.01~0.1），不会盖过 FM loss。
- `router_noise_std_deg = 0.3` 用 Switch-Transformer 经验值，防止退化轴 expert collapse。
- 三个 routing lr 都低（1e-4），且需要 grad clip。

---

## 10. 实施顺序（给 Codex 的 task plan）

| Step | 文件 | 内容 |
|---|---|---|
| 1 | `models/mole_lora.py` | 实现 `MoLERouterRegistry`, `MoLELoraLinear`, `convert_linear_to_mole`。包含单元测试：`test_lora_zero_init_matches_base()`。 |
| 2 | `models/profile_text_router.py` | `FrozenSentenceEncoder`, `DegradationRouter`。包含单元测试：`test_router_softmax_sum_to_one()`, `test_topk_sparsity()`。 |
| 3 | `models/scene_router.py` | `SceneRouter` + EM。单元测试：`test_buffer_ring()`, `test_init_centers_smoke()`。 |
| 4 | `models/mole_losses.py` | 三个 loss。单元测试：`test_balance_uniform_returns_one()`. |
| 5 | `models/prompt_builder.py` | 新增 `build_profile_router_text()`. 单元测试在 `tests/`. |
| 6 | `dataloaders/rg_flux_jsonl_dataset.py` | `__getitem__` 返回 `raw_profile_text`；collate fn 加字段。 |
| 7 | `models/flux2_klein_sr_artist.py` | 抽 `_forward_baseline_body`；加 `use_mole` 分支；改 `_apply_lora` / `_build_mole_modules` / save / load。 |
| 8 | `rg_flux_fm.py` | `sample_multistep_fm` forward 调用加 `profile_text` 透传。 |
| 9 | `train_rg_flux_sr.py` | 主循环加 routing loss、EM 更新、param groups、broadcast。 |
| 10 | `inference_rg_flux_sr.py` | 推理路径加 `profile_text` 透传；关闭 buffer push。 |
| 11 | `configs/train_rg_flux2_klein_sr_smoke_256.yaml` | 加 `mole:` 节；smoke 用 `num_deg_experts: 2, num_scene_experts: 2` 验证。 |
| 12 | smoke run | `--dry_run` 跑通；检查 routing entropy 与 LoRA delta L2 在合理范围。 |

---

## 11. 验收 checklist（每步必须满足）

- [ ] 设置 `g_d / g_s` 为均匀分布、`deg_B / scn_B` 为 0 时，模型输出 == baseline transformer 输出（数值级 ≤ 1e-4）。
- [ ] forward + backward 通过 `--dry_run`。
- [ ] 第 200 step 后 `scene_router.centers_inited == True` 且 cluster size > K*8。
- [ ] 第 1000 step：`g_d` entropy / log(K_d) ∈ [0.5, 0.95]（不 collapse 也不 uniform）。
- [ ] `g_s` 的 top-1 distribution 在 K_s 上不 collapse（每个 expert 使用率 ≥ 5%）。
- [ ] LoRA delta L2 norm 随训练单调上升，但不爆炸。
- [ ] 与 baseline 在 LSDIR val 上的 CLIPIQA / MUSIQ / MANIQA 比较：MoLE ≥ baseline。
- [ ] Routing 可视化：每个 expert 激活样本 IQA 关键词分布有显著差异（χ² test p < 0.01）。
- [ ] Scene cluster 代表样本人工 inspect 后能看出语义/场景差异。

---

## 12. 论文价值速记（给作者，不在代码里）

- **Contribution 1**：Decomposed dual-axis MoE on FLUX.2 SR（首次）。
- **Contribution 2**：文本-profile-routed degradation experts，**without numerical degradation vector**。
- **Contribution 3**：Self-discovered scene experts via latent-space EM（无监督）。
- **Contribution 4**：Routing-consistency loss 让 routing 可解释、可可视化。
- **Ablation 表**：
  - (a) baseline (single LoRA)
  - (b) only degradation experts
  - (c) only scene experts
  - (d) both, no routing loss
  - (e) full DDA-MoLE
- **可视化图**：(a) routing heatmap × IQA keyword, (b) scene cluster t-SNE on z_lr, (c) per-expert delta L2 sparsity。
