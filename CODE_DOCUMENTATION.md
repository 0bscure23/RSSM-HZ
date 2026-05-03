# RSSM-HZ 最优方法详解文档

## 一、我们要解决什么问题？

### 1.1 全色锐化的任务定义

卫星拍一张图，受物理限制，只能二选一：

| 图像 | 空间分辨率 | 光谱分辨率 | 通道数 |
|------|-----------|-----------|--------|
| PAN（全色） | 高（200×200） | 无（灰度） | 1 |
| MS（多光谱） | 低（50×50） | 有（RGB+近红外等） | 4 |

**任务**：把 PAN 和 MS 融合，输出一张**既有 200×200 的清晰度，又有 4 个光谱波段**的图像。

```
输入:                                  输出:
  PAN [1, 200, 200]  ← 高分辨率灰度      HRMS [4, 200, 200]  ← 高分辨率多彩
  MS  [4,  50,  50]  ← 低分辨率多彩
  LMS [4, 200, 200]  ← MS双三次上采样（辅助）
```

### 1.2 已有方法：WFANet 怎么做

WFANet 的核心思路是**用小波变换把图像拆成不同频率，然后用注意力机制去融合**。

```
PAN (灰度, 200×200)                 MS (多彩, 50×50)
       │                                  │
       ▼                                  ▼
  小波分解 (Haar)                    上采样到 200×200
  拆成4个子带:                              │
  ├── LL: 低频近似 (100×100)               ▼
  ├── LH: 水平高频                     小波分解
  ├── HL: 垂直高频                     拆成4个子带
  └── HH: 对角高频
       │                                  │
       └──────────┬───────────────────────┘
                  ▼
    ┌──────────────────────────────┐
    │  Self-Attention 交叉注意力    │
    │  Q = PAN子带  "我要什么细节？" │
    │  K = PAN低频  "在哪里匹配？"   │
    │  V = MS特征   "注入什么内容？" │
    │                              │
    │  问题: 100×100特征图 →       │
    │  注意力矩阵 [10000, 10000]    │
    │  复杂度 O(N²)，大图会爆显存   │
    └──────────────┬───────────────┘
                   ▼
              逆小波重建
                   │
                   ▼
        输出 HRMS [4, 200, 200]
```

### 1.3 我们的方法：RSSM-HZ 的不同思路

WFANet 把每一层小波**独立**地用 attention 处理。我们问：**能不能把"从粗到细"的三层小波当作一个序列过程来建模？**

```
WFANet 的思路:                      我们的思路:
                                    
  粗层 ──→ attention ──→ 输出         粗层 ──→ GRU ──→ 状态_粗
  中层 ──→ attention ──→ 输出                      │ 上采样传递
  细层 ──→ attention ──→ 输出         中层 ──→ GRU ──→ 状态_中
  (各自独立，无信息传递)                           │ 上采样传递
                                                  细层 ──→ GRU ──→ 状态_细
                                                  (状态逐层积累上下文)
```

**核心直觉**：粗尺度的融合结果（"这片区域是城市"）应该指导细尺度的融合（"这个像素是屋顶边缘还是道路"）。GRU 的状态传递机制天然适合这种"粗→细"的信息流。

---

## 二、RSSM-HZ 的完整架构

### 2.1 从输入到输出：端到端数据流

```
┌─────────────────────────────────────────────────────────────────────┐
│                       RSSMHWViTHZ 前向传播                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PAN [B,1,200,200]            MS [B,4,50,50]     LMS [B,4,200,200] │
│       │                            │                    │           │
│       ▼                            ▼                    │           │
│  ┌──────────┐              ┌──────────────┐            │           │
│  │pan_raise │              │ ms_upsample  │            │           │
│  │Conv 1→32 │              │Conv→PixelShuffle(4×)     │           │
│  └────┬─────┘              └──────┬───────┘            │           │
│       │                           │                    │           │
│       │                   ┌───────▼────────────────────┘           │
│       │                   │ ms_up = PReLU(上采样MS + LMS)          │
│       │                   │ ★ LMS残差连接 —— 模型从LMS基线出发     │
│       │                   └───────┬────────────────────             │
│       │                           │                                │
│       ▼                           ▼                                │
│  [B,32,200,200]            [B,32,200,200]  ← ms_raise: 4→32通道    │
│       │                           │                                │
│       └─────────┬─────────────────┘                                │
│                 ▼                                                  │
│  ┌─────────────────────────────────────────┐                       │
│  │         WaveletPyramid (3级Haar)         │                       │
│  │                                         │                       │
│  │  PAN: [32,200,200] → [32,100,100] ×4子带│ (Level 0, 最细)      │
│  │       [32,100,100] → [32, 50, 50] ×4子带│ (Level 1, 中等)      │
│  │       [32, 50, 50] → [32, 25, 25] ×4子带│ (Level 2, 最粗)      │
│  │                                         │                       │
│  │  MS:  同样的三级分解                      │                       │
│  └──────────────────┬──────────────────────┘                       │
│                     │                                              │
│                     ▼                                              │
│  ┌─────────────────────────────────────────┐                       │
│  │       RSSMWaveletFusionHz               │  ← ★ 本文核心 ★      │
│  │                                         │                       │
│  │  从最粗尺度(25×25)到最细尺度(100×100):    │                       │
│  │    h0=0, z0=0                           │                       │
│  │    Level 2: GRU更新 → h₂,z₂             │                       │
│  │    Level 1: 上采样h₂,z₂ → GRU更新 → h₁,z₁│                       │
│  │    Level 0: 上采样h₁,z₁ → GRU更新 → h₀,z₀│                       │
│  │                                         │                       │
│  │  每层同时做 PAN高频注入:                  │                       │
│  │    fused = MS子带 + α · PAN子带          │                       │
│  │    (α 由门控网络学习)                     │                       │
│  └──────────────────┬──────────────────────┘                       │
│                     │                                              │
│                     ▼                                              │
│  ┌──────────────────────┐                                          │
│  │ reduce_channel 32→4  │                                          │
│  └──────────┬───────────┘                                          │
│             │                                                      │
│             ▼                                                      │
│  ┌────────────────────────────────┐                                │
│  │ out = PReLU(fused + ms_up)     │  ← 最终残差: LMS基线+RSSM增强   │
│  │                                │                                │
│  │ ★ fused_weight 控制融合强度    │    训练初期≈0→纯LMS输出         │
│  │   训练后可学到最优融合比例      │    训练后期≈1→完整RSSM贡献      │
│  └────────────────────────────────┘                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 代码入口对应关系

```python
# rssm_hz_wfanet.py 第 457-495 行
class RSSMHWViTHZ(nn.Module):
    def forward(self, pan, ms, lms):
        # ===== 步骤1: 输入预处理 =====
        ms_up = self.ms_upsample(ms)          # [B,4,50,50] → [B,4,200,200]
        ms_up = self.ms_act(ms_up + lms)      # + LMS 残差 → PReLU激活
        
        # ===== 步骤2: 通道提升 =====
        pan_feat = self.pan_raise(pan)         # [B,1,200,200] → [B,32,200,200]
        ms_feat  = self.ms_raise(ms_up)        # [B,4,200,200] → [B,32,200,200]
        
        # ===== 步骤3: 小波分解 =====
        pan_pyr = self.wavelet(pan_feat)       # 3级分解, 每级有LL/LH/HL/HH
        ms_pyr  = self.wavelet(ms_feat)
        
        # ===== 步骤4: RSSM跨尺度融合 =====  ★ 核心 ★
        fused, kl_loss = self.rssm_fusion(
            pan_pyr, ms_pyr, training=self.training
        )
        
        # ===== 步骤5: 输出层 =====
        fused = self.reduce(fused)             # [B,32,200,200] → [B,4,200,200]
        out = self.out_act(fused + ms_up)      # 残差连接 → 最终输出
        return out, kl_loss
```

---

## 三、GRU 状态传播 —— 整个方法的核心机制

### 3.1 为什么需要"状态"？

**问题**：三个小波尺度独立处理时，粗尺度知道"这片区域是森林"，但细尺度不知道这个上下文，可能把树冠边缘误判为建筑边缘。

**解决方案**：让粗尺度的处理结果（"森林"）作为一个**隐藏状态**，传递给细尺度。细尺度在自己的处理中**同时参考**当前观测和粗尺度传来的状态。

### 3.2 GRU 是什么？为什么用它？

GRU（门控循环单元）是一种带"门控"的循环神经网络。它的核心能力是**选择性记忆**——用两个门（重置门 r、更新门 u）自动决定：

- 记住多少旧信息
- 吸收多少新信息

```
┌──────────────────────────────────────────────────────────┐
│                    GRU 门控机制                           │
│                                                          │
│  输入: x = [观测obs, 随机状态z_prev] 拼接                 │
│  旧状态: h_prev                                          │
│                                                          │
│  ┌─────────────────────────────────────────────────┐     │
│  │  重置门: r = σ(W_r · [x, h_prev])              │     │
│  │          决定"忘记多少旧状态"                     │     │
│  │          r→0: 清空旧状态，当作第一次看到         │     │
│  │          r→1: 保留所有旧状态                     │     │
│  └─────────────────────────────────────────────────┘     │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐     │
│  │  候选状态: n = tanh(W_n · [x, r⊙h_prev])        │     │
│  │           融合新输入和被选择性保留的旧状态         │     │
│  └─────────────────────────────────────────────────┘     │
│                          ↓                               │
│  ┌─────────────────────────────────────────────────┐     │
│  │  更新门: u = σ(W_u · [x, h_prev])              │     │
│  │         决定"新旧各占多少比例"                    │     │
│  │         u→0: 完全保留旧状态 h_new≈h_prev        │     │
│  │         u→1: 完全采用新状态 h_new≈n             │     │
│  └─────────────────────────────────────────────────┘     │
│                          ↓                               │
│         h_new = (1-u)⊙h_prev + u⊙n                      │
│                ↑旧状态保留比例   ↑新状态采纳比例          │
└──────────────────────────────────────────────────────────┘
```

### 3.3 GRU 在我们的方法中如何使用

```python
# rssm_hz_wfanet.py 第 112-117 行
def _forward_2d(self, obs, h_prev, z_prev, training):
    # 拼接当前观测和上一时刻随机状态 → 输入GRU
    h_bar = self.gru(
        torch.cat([obs, z_prev], dim=1),   # 输入 = 当前观测 + 上时刻随机状态
        h_prev                               # 上一时刻的确定性状态
    )
    # Phase A (h-only): z=0, kl=0
    if self.deterministic_only:
        z = torch.zeros_like(z_prev)
        kl = torch.zeros(...)
        return h_bar, z, kl
    
    # Phase B (h+z): 继续计算随机状态 z...
```

**这里的 obs 是什么？** 在当前小波尺度上，PAN 特征和 MS 特征拼接后的向量。例如在 Level 2（最粗，25×25）：

```
obs = cat([
    pan_proj(PAN_Level2_特征),    # PAN的LL/LH/HL/HH拼接后投影 → 128通道
    ms_proj(MS_Level2_特征)       # MS的LL投影 → 128通道
])  # → [B, 256, 25, 25]
```

GRU 读了 obs，结合上一级传来的 h，决定：
- 哪些粗尺度上下文要保留（更新门 u）
- 哪些要遗忘、让当前观测主导（重置门 r）
- 输出新的隐藏状态 h₂，包含融合了"粗尺度上下文 + 当前PAN/MS观测"的信息

### 3.4 状态跨尺度传播的完整过程

```
                         特征图尺寸变化
                         ═══════════

  Level 2 (最粗)         h₂ [B,96,25,25]    ← 25×25 状态，含"全局结构"信息
       │                       │
       │           ┌───────────▼────────────────────┐
       │           │ ConvTranspose2d (转置卷积)      │
       │           │ 输入: [B,96,25,25]             │
       │           │ 输出: [B,96,50,50]             │
       │           │ ★ 可学习的上采样，不是简单插值  │
       │           └───────────┬────────────────────┘
       │                       │
       ▼                       ▼
  Level 1 (中等)         h₁ [B,96,50,50]    ← 50×50 状态，含"中等尺度上下文"
       │        obs₁ = [PAN_Level1, MS_Level1]
       │        h₁ = GRU(obs₁, h₂↑)      ← GRU融合新观测+旧上下文
       │                       │
       │           ┌───────────▼────────────────────┐
       │           │ ConvTranspose2d (转置卷积)      │
       │           │ 输入: [B,96,50,50]             │
       │           │ 输出: [B,96,100,100]           │
       │           └───────────┬────────────────────┘
       │                       │
       ▼                       ▼
  Level 0 (最细)         h₀ [B,96,100,100]   ← 100×100 状态，含"局部细节上下文"
       │        obs₀ = [PAN_Level0, MS_Level0]
       │        h₀ = GRU(obs₀, h₁↑)      ← GRU融合最终观测+全局上下文
       │
       ▼
  最终融合特征 → 逆小波重建 → 输出
```

**关键点**：状态 h 的维度始终是 [B, 96, H, W]，只是 H×W 随尺度变化。转置卷积负责把粗尺度的 25×25 状态"展开"到 50×50，保留通道维度的语义信息。

---

## 四、PAN 高频注入 —— 细节从哪来

GRU 状态传播解决的是**跨尺度上下文传递**，但 PAN 的高频细节（边缘、纹理）如何注入到输出中？

### 4.1 门控注入机制

每个小波尺度上，除了用 GRU 更新状态，还有一个**高频注入**步骤：

```
┌──────────────────────────────────────────────────────────────┐
│              高频注入（以 LH 子带为例）                        │
│                                                              │
│  PAN 小波分解后:                                              │
│    LL_PAN [B,1,100,100]    ← 低频，不太需要                   │
│    LH_PAN [B,1,100,100]    ← 水平边缘，要注入！               │
│    HL_PAN [B,1,100,100]    ← 垂直边缘，要注入！               │
│    HH_PAN [B,1,100,100]    ← 对角纹理，要注入！               │
│                                                              │
│  MS 小波分解后:                                               │
│    LH_MS  [B,4,100,100]    ← MS自己的水平高频（弱）           │
│                                                              │
│  ┌─────────────────────────────────────────────┐             │
│  │       门控网络 high_gate_lh                  │             │
│  │                                             │             │
│  │  输入拼接:                                   │             │
│  │    fused_ll   ← GRU融合后的LL特征            │             │
│  │    ll_ms      ← MS的低频                     │             │
│  │    pan_lh     ← PAN的水平高频                │             │
│  │    pan_hl     ← PAN的垂直高频                │             │
│  │    pan_hh     ← PAN的对角高频                │             │
│  │    z_gate     ← 随机隐变量z（如有）           │             │
│  │                                             │             │
│  │  输出: α = sigmoid(小型Conv网络(输入))       │             │
│  │         α ∈ [0,1], 逐像素不同                │             │
│  └────────────────────┬────────────────────────┘             │
│                       │                                      │
│                       ▼                                      │
│  ┌─────────────────────────────────────────────┐             │
│  │  fused_LH = LH_MS + α · LH_PAN              │             │
│  │             ↑基座     ↑可调增益               │             │
│  │                                             │             │
│  │  α→0: 这个像素不需要PAN细节(平坦区域)         │             │
│  │  α→1: 这个像素充分注入PAN细节(边缘/纹理)      │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  LH/HL/HH 三个子带各有独立的门控网络，分别学习:               │
│    high_gate_lh → α_lh (水平方向注入强度)                     │
│    high_gate_hl → α_hl (垂直方向注入强度)                     │
│    high_gate_hh → α_hh (对角方向注入强度)                     │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 代码实现

```python
# rssm_hz_wfanet.py 第 363-391 行 (在 RSSMWaveletFusionHz.forward 中)
# 对每个小波层级:

# 1. PAN高频从1通道转到MS的通道数
pan_lh = self.pan_high_to_ms[level]["lh"](lh_pan)   # [B,1,H,W]→[B,4,H,W]
pan_hl = self.pan_high_to_ms[level]["hl"](hl_pan)
pan_hh = self.pan_high_to_ms[level]["hh"](hh_pan)

# 2. z状态影响门控（Phase B时有效）
z_gate = self.z_to_gate[level](z_state)

# 3. 拼接所有信息 → 门控网络 → 产生 α
gate_in = torch.cat([fused_ll, ll_ms, pan_lh, pan_hl, pan_hh, z_gate], dim=1)

# 4. 各子带独立的门控
alpha_lh = torch.sigmoid(self.high_gate_lh[level](gate_in))
alpha_hl = torch.sigmoid(self.high_gate_hl[level](gate_in))
alpha_hh = torch.sigmoid(self.high_gate_hh[level](gate_in))

# 5. 门控注入
fused_lh = lh_ms + alpha_lh * pan_lh
fused_hl = hl_ms + alpha_hl * pan_hl
fused_hh = hh_ms + alpha_hh * pan_hh

# 6. 四个子带 (fused_ll, fused_lh, fused_hl, fused_hh) → 逆小波重建
```

### 4.3 门控的物理意义

```
  α ≈ 1 (门控开):                   α ≈ 0 (门控关):
  ┌─────────────────┐               ┌─────────────────┐
  │ 天空区域(平坦)   │               │ 建筑边缘(高频)   │
  │                 │               │ █████████████   │
  │ 无边缘→不需要   │               │ ████    ████   │
  │ PAN高频注入     │               │ ████    ████   │
  │                 │               │ PAN细节充分注入 │
  │ α→0, 纯MS输出   │               │ α→1, PAN主导   │
  └─────────────────┘               └─────────────────┘
```

门控是**逐像素学习**的——模型自己学会判断每个像素位置需要多少 PAN 细节。这种自适应性是方法效果好的关键原因之一。

---

## 五、Phase A vs Phase B —— 两种训练模式

### 5.1 Phase A（h-only，确定性训练）

```python
# 代码: rssm_hz_wfanet.py 第 112-117 行
if self.deterministic_only:
    z = torch.zeros_like(z_prev)      # z永远是0
    kl = torch.zeros(...)              # KL损失永远是0
    return h_bar, z, kl               # 只返回GRU更新后的h
```

**Phase A 做了什么**：
- GRU 更新隐藏状态 h（跨尺度传递上下文）
- z 固定为零（无随机隐变量）
- 无 KL 损失（无变分推断）

**训练配置**：
- `deterministic_only=True`
- 损失函数: `L = L1(pred, gt)`
- 800 epochs, lr = 9e-4 → 0 (cosine)

**Phase A 单独就够好了**：h-only 模式下，Q8 已经达到 0.9524，在所有指标上追平 WFANet。这说明**GRU 状态传播 + 门控高频注入**已经足够，随机隐变量不是必需的。

### 5.2 Phase B（h+z，随机状态空间）

```python
# 代码: rssm_hz_wfanet.py 第 119-132 行
# (deterministic_only=False 时执行)
prior_stats = self.prior(h_bar)           # 先验网络: h → [μ_p, σ_p]
mu_p, logvar_p = torch.chunk(prior_stats, 2, dim=1)

if training:
    post_stats = self.posterior(           # 后验网络: [h, obs] → [μ_q, σ_q]
        torch.cat([h_bar, obs], dim=1)
    )
    mu_q, logvar_q = torch.chunk(post_stats, 2, dim=1)
    z = mu_q + ε · exp(logvar_q/2)        # 重参数化采样
    kl = KL( q(z|h,obs) || p(z|h) )       # KL散度正则化
else:
    z = mu_p                               # 推理时直接用先验均值
    kl = 0
```

**Phase B 做了什么**：
- 在 GRU 状态 h 之外，额外学习一个**随机隐变量 z**
- z 的分布由先验网络（只看 h）和后验网络（看 h + 观测 obs）共同决定
- KL 散度约束"后验不要离先验太远"——信息瓶颈

**训练配置**：
- 从 Phase A 最佳 checkpoint 加载权重
- `deterministic_only=False`
- 冻结浅层特征提取器（`pan_raise, ms_upsample, ms_act, ms_raise`）
- 损失函数: `L = L1(pred, gt) + β_t · KL`
- KL 系数从 0 线性增长到 8e-5（100 epoch ramp-up）
- 200 epochs, lr = 2.7e-4 → 0

**Phase B 的收益**：在 jilin 数据集上约 +0.0005 Q8，几乎可以忽略。**但随机隐变量的价值在于**：
- 对含噪声数据的鲁棒性（z 建模不确定性）
- 可作为时序融合的扩展基础
- 为未来工作（不确定性估计、主动学习）留接口

### 5.3 训练阶段切换示意

```
Phase A (主力, 800 epochs)              Phase B (微调, 200 epochs)
════════════════════════════           ══════════════════════════════
                                      ┌─ 加载 Phase A 最佳权重
h_t = GRU(obs_t, h_{t-1})            │  h_t = GRU(obs_t, h_{t-1})     ← 相同
z_t = 0                              │  z_t ~ N(μ_q, σ_q)            ← 新增
kl  = 0                              │  kl  = KL(q‖p)                ← 新增
L   = L1(pred, gt)                   │  L   = L1 + β·KL (+蒸馏)      ← 扩展
                                      │
★ 训练稳定, 快速收敛                  │  ★ 微调RSSM核心, 冻结特征提取器
★ 已超越WFANet                       │  ★ 可选蒸馏(论文中应去掉)
```

---

## 六、数据流：裁切训练 + 拼贴推理

### 6.1 为什么需要裁切训练

jilin 数据是 200×200 的大图。RSSM-HZ 的 GRU 在 100×100 特征图上运行时，随机初始化的权重会产生超大激活值，导致梯度爆炸（Inf/NaN）。我们实测：200×200 全图训练时，**83% 的模型参数梯度是非有限的**。

**解决方案**：训练时只用 64×64 的小块，模型在正常尺度下学习。推理时把大图切成小块分别处理，再拼回去。

### 6.2 CropDataset：如何生成训练块

```
原始图像 200×200
┌──────────────────────┐
│                      │        每个 epoch, 每张图随机取 4 个位置:
│   ┌────┐             │        
│   │块1 │   ┌────┐    │        ┌──────────┐  ┌──────────┐
│   │64×64│  │块3 │    │   ──→  │ PAN 块   │  │ MS 块    │
│   └────┘   │64×64│   │        │ 64×64    │  │ 16×16    │
│   ┌────┐   └────┘    │        └──────────┘  └──────────┘
│   │块2 │   ┌────┐    │        (PAN的4个像素 = MS的1个像素)
│   │64×64│  │块4 │    │
│   └────┘   │64×64│   │        900张图 × 4块/图
│            └────┘    │        = 3600个训练块/epoch
└──────────────────────┘
```

```python
# train_rssmhz_crop.py CropDataset.__getitem__
y = random.randint(0, H - 64)          # 随机Y起始位置
x = random.randint(0, W - 64)          # 随机X起始位置

return (
    pan[i, :, y:y+64, x:x+64],         # PAN块:   [1, 64, 64]
    gt[i,  :, y:y+64, x:x+64],         # GT块:   [4, 64, 64]
    ms[i,  :, y//4:y//4+16, ...],      # MS块:   [4, 16, 16]  ← 注意 y//4!
    lms[i, :, y:y+64, x:x+64],         # LMS块:  [4, 64, 64]
)
```

**关键细节**：MS 的裁切位置用 `y//4`，因为 PAN 分辨率是 MS 的 4 倍。PAN 的 (y, x) 对应 MS 的 (y//4, x//4)，保证空间严格对齐。

### 6.3 tiled_forward：推理时如何拼回去

```
推理: 200×200 全图 → 切成 64×64 tiles → 逐个推理 → 拼贴 → 输出

  ┌─────┬─────┬─────┬─────┐
  │ T1  │ T2  │ T3  │ T4  │     tiles = ceil(200/64) = 4
  ├─────┼─────┼─────┼─────┤     每边 4 个 tile = 16 个 tile
  │ T5  │ T6  │ T7  │ T8  │
  ├─────┼─────┼─────┼─────┤     每个 tile 64×64, 含 8px overlap
  │ T9  │ T10 │ T11 │ T12 │     相邻 tile 重叠 16 像素(8+8)
  ├─────┼─────┼─────┼─────┤
  │ T13 │ T14 │ T15 │ T16 │     重叠区域: 两个tile的结果加权平均
  └─────┴─────┴─────┴─────┘     (重叠越多, 权重归一化)

  边缘 tile 可能不完整 → 按实际尺寸裁剪, 覆盖次数少, 权重自动调整
```

```python
# train_rssmhz_crop.py tiled_forward 核心逻辑
output = torch.zeros(1, C_out, H, W)       # 输出累积器
weight = torch.zeros(1, 1, H, W)           # 覆盖次数记录器

for ti in range(tiles_h):
    for tj in range(tiles_w):
        out_tile = model(pan_tile, ms_tile, lms_tile)  # 单tile推理
        out_crop = out_tile[:,:,pad:-pad,pad:-pad]     # 去掉overlap边界
        
        output[:,:,i:i+eh, j:j+ew] += out_crop         # 累加
        weight[:,:,i:i+eh, j:j+ew] += 1.0              # 计数

final = output / weight.clamp_min(1.0)                 # 归一化
```

---

## 七、代码文件结构与职责

```
WFANet/
│
├── rssm_hz_wfanet.py          ★ 模型定义（本文核心贡献）
│   ├── ConvGRUCell2d           2D卷积GRU (可选, jilin未用)
│   ├── RSSMHzCell             ★ GRU + 先验/后验网络 (状态更新单元)
│   ├── CrossScaleFusionHz     ★ 单层融合块 (GRU→门控→输出)
│   ├── RSSMWaveletFusionHz    ★ 三级粗→细融合管线 (状态传播+高频注入)
│   ├── WaveletPyramid          小波分解金字塔
│   ├── WaveletReconstructor    逆小波重建器
│   └── RSSMHWViTHZ            ★ 完整模型 (组装上述所有组件)
│
├── train_rssmhz_crop.py       ★ 裁切训练脚本（产生最优模型）
│   ├── CropDataset             随机64×64裁切数据加载器
│   ├── tiled_forward           拼贴推理 (大图→tile→拼回)
│   ├── Phase A 训练循环         h-only, L1 loss, 800 epochs
│   └── Phase B 训练循环         h+z, L1+KL+distill, 200 epochs
│
├── net_torch.py               WFANet 模型 + 小波变换（复用+对比）
│   ├── DWT_2D / IDWT_2D       ★ Haar小波正/逆变换 (被RSSM-HZ导入使用)
│   ├── HWViT                   WFANet 主模型 (教师/基线)
│   └── Attention / S_MWiT / F_MWiT / L_MWiT   WFANet子模块
│
├── train_wfanet_jilin_crop.py WFANet 基线训练 (产生对比基线)
├── train_rssm_hz.py           原始全图训练脚本 (梯度不稳定, 已弃用)
├── evaluate_wv3_metrics.py    评估指标 (PSNR/SAM/ERGAS/Q8)
├── super_para_panscale.yml    jilin 数据集配置 (ratio=255, L_up_channel=4)
└── super_para.yml             WV3 数据集配置 (original)
```

### 核心 import 关系

```
train_rssmhz_crop.py
  ├── from rssm_hz_wfanet import RSSMHWViTHZ     ← 创建自己的模型
  ├── from net_torch import HWViT                 ← 创建教师模型(蒸馏用)
  ├── from evaluate_wv3_metrics import calculate_metrics  ← 评估
  └── (net_torch 中的 DWT_2D 由 rssm_hz_wfanet 间接使用)

rssm_hz_wfanet.py
  └── from net_torch import (
        DWT_2D, IDWT_2D,         ← 小波变换 (复用WFANet的实现)
        raise_channel, reduce_channel,  ← 通道升降 (复用WFANet的实现)
        resblock, DWC, FFN,      ← 基础模块 (复用WFANet的实现)
      )
```

---

## 八、关键超参数与设计选择

### 8.1 模型结构参数

| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 128 | GRU 隐藏状态维度（标准96→优化128） |
| latent_dim | 48 | 随机隐变量 z 维度（标准32→优化48） |
| pan_target_channel | 32 | PAN 通道提升目标 |
| ms_target_channel | 32 | MS 通道提升目标 |
| L_up_channel | 4 | 输出通道数 = jilin 波段数 |
| 小波级数 | 3 | Haar小波3级分解（200→100→50→25） |
| 参数量 | 2.08M | h128 大模型; 标准模型 1.23M |

### 8.2 训练超参数

| 参数 | Phase A | Phase B | 选择理由 |
|------|---------|---------|---------|
| batch_size | 12 | 12 | h128 GRU状态占显存，3090上限 |
| epochs | 800 | 200 | 训练越久越好，800ep后趋于饱和 |
| lr_max | 9e-4 | 2.7e-4 | Phase B用0.3×保护Phase A权重 |
| LR schedule | cosine | cosine | 平滑衰减，无突变 |
| crop_size | 64 | 64 | 64×64下梯度正常，128×128无明显增益 |
| grad_clip | 1.0 | 1.0 | 梯度范数裁剪，防止偶发爆炸 |
| w_kl | 0 | 8e-5 | Phase A无KL；Phase B KL权重微小 |
| kl_ramp | — | 100 epoch | KL从0线性增长，避免早期KL主导 |
| freeze | — | shallow | 冻结特征提取器，只训练RSSM+输出 |

### 8.3 为什么这些设计选择有效

1. **hidden_dim=128（非96）**：更大的GRU状态容量 → 更丰富的跨尺度上下文。代价是参数量从 1.23M → 2.08M，训练速度下降约 15%。

2. **cosine LR（非指数衰减）**：cosine 在训练末期 LR 极低（→0），允许模型精细收敛。指数衰减在末期 LR 可能仍然偏高。

3. **shallow freeze（Phase B）**：冻结 `pan_raise, ms_upsample, ms_act, ms_raise` —— 这些是特征提取器，Phase A 已经学好。Phase B 只微调 RSSM 核心和输出层，防止 KL loss 破坏已学好的特征。

4. **裁切尺寸=64**：在这个尺寸下，小波分解后的特征图最大为 32×32（Level 0 的子带），GRU 在此尺度下梯度完全正常。128×128 裁切虽然没有梯度问题，但并未带来性能增益——说明 64×64 的感受野已经足够。

---

## 九、评估指标说明

| 指标 | 全称 | 方向 | 物理含义 |
|------|------|------|---------|
| **Q8** | Q-index (8×8 window) | ↑ 越高越好 | 8×8滑动窗口内的空间-光谱综合质量，最接近人眼感知 |
| **PSNR** | Peak Signal-to-Noise Ratio | ↑ 越高越好 | 逐像素重建精度，越高的dB值表示误差越小 |
| **SAM** | Spectral Angle Mapper | ↓ 越低越好 | 光谱向量角度误差（度），衡量"颜色"保真度 |
| **ERGAS** | Relative Global Synthesis Error | ↓ 越低越好 | 综合相对全局误差，越低表示整体融合质量越好 |

---

## 十、论文建议：简化版方案

基于实验结论，论文中最干净、最强的方案是：

```
RSSM-HZ (h-only, 无蒸馏, 无随机隐变量)
├── 模型: hidden_dim=128, deterministic_only=True
├── 训练: 64×64 随机裁切, L1 loss, 800 epochs
├── 推理: tiled_forward (64×64 tiles, pad=8 overlap)
└── 结果: Q8=0.9524, PSNR=39.41, SAM=1.14, ERGAS=1.18
          全面追平 WFANet, PSNR/SAM/ERGAS 三项超越
```

优势表述：
- **独立方法**：不依赖任何预训练教师模型
- **线性复杂度**：GRU 的 O(HW) vs Attention 的 O((HW)²)
- **简单干净**：无随机采样，可完美复现
- **可扩展**：Phase B 作为可选扩展（噪声/时序场景）
