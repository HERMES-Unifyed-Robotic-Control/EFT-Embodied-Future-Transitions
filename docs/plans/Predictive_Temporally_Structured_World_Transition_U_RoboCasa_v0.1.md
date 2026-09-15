# Predictive Temporally Structured World Transition U  
## RoboCasa 初版实验方案 v0.1

---

# 1. 实验目标

本阶段目标是在 RoboCasa 数据上训练一个：

\[
\boxed{
U_t=[u_t^1,u_t^2,u_t^3,u_t^4]
}
\]

其中每个 \(u_t^k\) 表示未来一个局部时间区间内：

- 场景视觉变化；
- 机器人自身状态变化；
- robot-object interaction 变化；
- 与真实机器人动作相关的可执行动态。

与传统 latent action 不同，本阶段不要求：

\[
U_t\approx A_{t:t+H}
\]

而希望：

\[
\boxed{
U_t
=
\text{Predictive Temporally Structured World Transition}
}
\]

即 \(U_t\) 主要描述：

> **在当前任务条件下，未来一个 action horizon 内，世界和机器人将如何逐段演化。**

真实 action 仅用于给 U 提供 executable grounding，而不是定义 U 本身。

---

# 2. 与上一版方法的核心区别

上一版 U 主要通过未来视觉和未来 robot state 构造：

\[
U
=
E(
F_t,F_{t+H},
S_{t:t+H}
)
\]

该设计适合分析真实轨迹，但存在两个问题：

1. inference 时没有 future observation；
2. U 可能直接编码未来结果，而不是预测未来 transition。

因此本版本改为：

\[
\boxed{
U_t
=
P_\theta(
I_{t-M:t},
S_{t-M:t},
L
)
}
\]

即：

> **U 只能由过去和当前信息预测。**

未来：

\[
I_{t+1:t+H},
S_{t+1:t+H},
A_{t:t+H}
\]

全部只在训练阶段作为监督目标。

因此整体方向由：

\[
Future\rightarrow U
\]

变为：

\[
\boxed{
Current/History\rightarrow U\rightarrow Future
}
\]

这是本版本最关键的修改。

---

# 3. 核心假设

本阶段主要验证以下假设：

\[
\boxed{
\begin{aligned}
&\text{A temporally structured latent predicted only from}\\
&\text{past/current task context can preserve richer}\\
&\text{world-transition and action-relevant dynamics}\\
&\text{than endpoint or action-only representations.}
\end{aligned}
}
\]

中文：

> **相比仅表示最终状态变化或直接压缩机器人动作，仅由当前及历史任务上下文预测、并通过时间结构化未来视觉和机器人状态变化监督得到的 U，可以保存更完整、更具可执行性的世界转移信息。**

---

# 4. 本阶段研究范围

第一版仅解决：

\[
\boxed{
RGB+Robot\ State+Language
\rightarrow
U_{1:4}
}
\]

并通过以下三个目标训练：

\[
\boxed{
U\rightarrow Future\ Visual\ Transition
}
\]

\[
\boxed{
U\rightarrow Future\ Robot\ Transition
}
\]

\[
\boxed{
U\rightarrow Robot\ Action
}
\]

本阶段暂不引入：

- 完整 future video generation；
- dense 3D scene flow；
- VQ / VAE；
- LEAP-style MSRQ；
- global + local hierarchical U；
- 多尺度 future horizon；
- contact / tactile branch；
- 复杂 latent world model rollout；
- 大规模跨 embodiment 训练。

优先保证研究问题足够干净。

---

# 5. 数据集

第一阶段使用：

\[
\boxed{\text{RoboCasa}}
\]

机器人 manipulation trajectory。

每条 episode 至少需要：

- multi-view RGB；
- robot proprio/state；
- robot action；
- language instruction；
- episode / timestep index。

如数据中存在：

- subtask label；
- stage label；
- atomic skill label；

第一版不作为 U 输入，只用于后续 representation probe。

---

# 6. 单个训练样本定义

对于 episode 中时刻：

\[
t
\]

构造：

\[
\boxed{
X_t=
\{
I_{t-M:t},
S_{t-M:t},
L
\}
}
\]

作为模型输入。

Future：

\[
Y_t=
\{
I_{t+\tau_1:t+\tau_K},
S_{t:t+H},
A_{t:t+H}
\}
\]

仅用于监督。

---

# 7. 时间窗口设置

第一版建议：

\[
\boxed{
T_{history}=0.5s
}
\]

历史 RGB 采样：

\[
N_h=4
\]

即：

\[
I_{t-0.5s},
I_{t-0.33s},
I_{t-0.17s},
I_t
\]

具体 frame index 根据数据 fps 自动换算。

未来 horizon：

\[
\boxed{
T_{future}=1.5s
}
\]

划分为：

\[
\boxed{
K=4
}
\]

个 interval：

\[
0=\tau_0<\tau_1<\tau_2<\tau_3<\tau_4=H
\]

第一版均匀划分：

\[
\tau=
[
0.375,
0.75,
1.125,
1.5
]s
\]

分别对应：

\[
u_1,u_2,u_3,u_4
\]

---

# 8. 数据边界处理

若：

\[
t+H>T_{episode}
\]

则不构造该样本。

同时建议避免大量采样完全静止的 window。

可根据：

\[
\|A_{t:t+H}\|
\]

或：

\[
\|\Delta S_{t:t+H}\|
\]

过滤长期无动作样本。

但不应完全删除静止状态。

建议训练样本比例：

\[
Moving:Static\approx4:1
\]

用于保留“无需动作”的状态理解能力。

---

# 9. 模型整体结构

整体模型：

```text
                Language L
                    │
                    ▼
               Text Encoder
                    │
                    │
RGB history ─► Visual Encoder ─┐
                               │
Robot history ─► State Encoder ├──► Context Fusion
                               │
Language tokens ───────────────┘
                                      │
                              Temporal Queries
                              q1 q2 q3 q4
                                      │
                                      ▼
                         Predictive U Predictor
                                      │
                             u1 u2 u3 u4
                                      │
                 ┌────────────────────┼─────────────────┐
                 │                    │                 │
                 ▼                    ▼                 ▼
           Visual Decoder        State Decoder      Action Head
                 │                    │                 │
          Future Visual         Future Robot        GT Action
          Transition            Transition
```

---

# 10. Visual Encoder

第一版使用：

\[
\boxed{\text{Frozen DINOv3}}
\]

对于：

\[
I_{t,v}
\]

得到：

\[
F_{t,v}
=
\phi_{DINO}(I_{t,v})
\]

其中：

\[
v=1,\dots,V
\]

表示不同 camera view。

DINOv3 全程：

\[
\boxed{Frozen}
\]

不参与更新。

---

# 11. Visual Feature 缓存

为了减少训练计算成本，建议首先对完整 RoboCasa 数据离线运行 DINOv3。

保存：

\[
F_{t,v}
\]

至：

```text
episode_xxx/
    view_0_features.npz
    view_1_features.npz
    view_2_features.npz
```

每帧不保存完整高分辨率 dense token。

第一版建议：

\[
AdaptivePool
:
H_f\times W_f
\rightarrow4\times4
\]

因此每个 camera：

\[
16
\]

个 spatial token。

如使用 3 个 view：

\[
N_{vision}=48
\]

tokens / timestep。

统一投影到：

\[
d=256
\]

维。

---

# 12. History Visual Encoding

对于历史：

\[
I_{t-M:t}
\]

得到：

\[
F^{hist}
=
[
F_{t_1},
F_{t_2},
F_{t_3},
F_t
]
\]

加入：

- temporal positional embedding；
- camera-view embedding。

随后输入轻量 Temporal Transformer：

\[
H_V
=
E_{hist}^{V}(F^{hist})
\]

第一版建议：

- Layers：4；
- Hidden dim：512；
- Heads：8。

---

# 13. Robot State 输入

Robot state 表示：

\[
S_t=
[
S_{base},
S_{arm},
S_{eef},
S_{gripper}
]
\]

根据 RoboCasa 实际字段选择可用项。

至少建议包含：

- joint position；
- EEF position；
- EEF rotation；
- gripper state。

如果存在移动底盘：

额外加入：

- base position；
- base orientation；
- base velocity。

---

# 14. State History Encoder

输入：

\[
S_{t-M:t}
\]

建议 state 保持较高时间分辨率，不需要像 RGB 一样只取 4 个 timestep。

例如：

\[
S^{hist}
=
[
S_{t-n},...,S_t
]
\]

先按 modality 进行 Grouped MLP：

\[
S_{arm}\rightarrow h_{arm}
\]

\[
S_{eef}\rightarrow h_{eef}
\]

\[
S_{gripper}\rightarrow h_{gripper}
\]

再 concat：

\[
h_t^S
\]

最后通过 2-layer Temporal Transformer：

\[
H_S
=
E_{hist}^{S}(S^{hist})
\]

---

# 15. Language Encoder

Language instruction：

\[
L
\]

通过 frozen text encoder：

\[
H_L
=
E_L(L)
\]

第一版 language 只作为 condition：

\[
L\rightarrow U
\]

不增加：

\[
U\rightarrow L
\]

的语言 reconstruction loss。

原因是本阶段需要解决的是：

\[
P(Future|I,S,L)
\]

而不是让 U 本身承担 language generation。

---

# 16. Context Fusion

将：

\[
H_V,
H_S,
H_L
\]

组成：

\[
C_t
=
[
H_V;
H_S;
H_L
]
\]

第一版不进行复杂 cross-modal expert routing。

使用：

\[
4
\]

个 learnable temporal queries：

\[
Q_U=
[q_1,q_2,q_3,q_4]
\]

通过 Cross Attention：

\[
U_t
=
CrossAttn(
Q_U,C_t
)
\]

输出：

\[
\boxed{
U_t=
[u_1,u_2,u_3,u_4]
}
\]

每个：

\[
u_k\in\mathbb R^{512}
\]

---

# 17. Temporal Query 的含义

强制：

\[
q_k
\leftrightarrow
[\tau_{k-1},\tau_k]
\]

即：

\[
u_1
\]

描述最近未来：

\[
0\rightarrow0.375s
\]

而：

\[
u_4
\]

描述：

\[
1.125s\rightarrow1.5s
\]

。

因此 U 天然形成：

\[
\boxed{
u_1\rightarrow u_2\rightarrow u_3\rightarrow u_4
}
\]

的时间有序结构。

第一版暂不使用 autoregressive：

\[
u_{k-1}\rightarrow u_k
\]

所有 \(u_k\) 并行生成。

---

# 18. Future Visual Target

Future RGB 不参与 U Predictor。

只离线编码：

\[
F_k
=
\phi_{DINO}
(
I_{t+\tau_k}
)
\]

获得：

\[
F_0,F_1,F_2,F_3,F_4
\]

其中：

\[
F_0=F_t
\]

---

# 19. Visual Transition Prediction

对于第 \(k\) 段：

\[
\boxed{
\hat F_k
=
D_V(
F_{k-1},
u_k
)
}
\]

而不是：

\[
u_k\rightarrow F_k
\]

从而迫使：

\[
u_k
\]

主要保存：

\[
F_{k-1}\rightarrow F_k
\]

的 transition information。

Visual Decoder 第一版：

- 4-layer Transformer；
- hidden dim 512；
- cross-attention condition = \(u_k\)。

---

# 20. Visual Loss

定义：

\[
\mathcal L_V^k
=
\mathcal L_{cos}^k
+
\lambda_{mse}
\mathcal L_{mse}^k
\]

建议：

\[
\lambda_{mse}=0.1
\]

最终：

\[
\boxed{
L_V
=
\frac1K
\sum_{k=1}^{K}
L_V^k
}
\]

---

# 21. Visual Change Weight

RoboCasa 场景存在大量静态背景。

因此增加 transition-aware token weight：

\[
c_{k,p}
=
1-
cos(
F_{k-1,p},
F_{k,p}
)
\]

然后：

\[
w_{k,p}
=
1+\alpha c_{k,p}
\]

第一版：

\[
\boxed{\alpha=2}
\]

视觉 loss：

\[
L_V^k
=
\frac{
\sum_p
w_{k,p}
d(
\hat F_{k,p},F_{k,p}
)
}{
\sum_p w_{k,p}
}
\]

从而提高：

- robot arm；
- gripper；
- moving object；
- interaction region；

对应视觉 token 的监督强度。

---

# 22. Future Robot Transition Target

未来 state trajectory：

\[
S_{t:t+H}
\]

按照 4 个 interval 划分：

\[
S^1,S^2,S^3,S^4
\]

每个 interval 内统一重采样：

\[
R=4
\]

个时间点。

得到：

\[
Y_k^S
\]

---

# 23. 使用 Relative Robot Transition

不建议直接预测绝对状态。

定义：

\[
Y_k^S=
[
\Delta P_{eef},
\Delta R_{eef},
\Delta Q_{joint},
\Delta G,
\Delta P_{base}
]
\]

其中根据实际机器人配置删除不存在的项。

所有连续变量首先：

\[
x'
=
\frac{x-\mu}{\sigma}
\]

标准化。

---

# 24. State Transition Decoder

第一版直接：

\[
\boxed{
\hat Y_k^S
=
D_S(u_k)
}
\]

即 decoder 不再额外读取：

\[
S_{\tau_{k-1}}
\]

从而要求：

\[
u_k
\]

本身保留 interval 内 robot transition information。

Decoder：

- 3-layer MLP；
- hidden dim 1024；
- 输出 \(R\times D_S\)。

---

# 25. State Loss

定义：

\[
L_S^k
=
\lambda_pL_{eef-pos}
+
\lambda_rL_{eef-rot}
+
\lambda_qL_{joint}
+
\lambda_gL_{gripper}
+
\lambda_bL_{base}
\]

第一版：

\[
\lambda_p=1.0
\]

\[
\lambda_r=1.0
\]

\[
\lambda_q=0.5
\]

\[
\lambda_g=0.5
\]

存在底盘时：

\[
\lambda_b=1.0
\]

最终：

\[
L_S
=
\frac1K
\sum_k
L_S^k
\]

Continuous state 使用：

\[
SmoothL1
\]

rotation 建议使用：

- 6D rotation；
或
- relative rotation representation。

避免直接 Euler MSE。

---

# 26. Action Grounding

真实 action：

\[
A_{t:t+H}
\]

只作为 executable grounding。

不再采用：

\[
D_A(
S_{\tau_{k-1}},
u_k
)
\]

因为 \(k>1\) 时包含未来 state。

部署兼容的 Action Head 定义为：

\[
\boxed{
\hat A_{t:t+H}
=
D_A(
S_t,
u_1,u_2,u_3,u_4
)
}
\]

输入始终只使用当前：

\[
S_t
\]

和 predictive U。

---

# 27. 第一版 Action Head

暂不使用：

- diffusion；
- flow matching；
- MSRQ；
- action tokenizer。

第一版使用轻量 Temporal Transformer：

输入：

\[
[
Proj(S_t),
u_1,u_2,u_3,u_4
]
\]

加入未来 action query：

\[
q^A_1,\dots,q^A_{N_A}
\]

输出：

\[
\hat A_{t:t+H}
\]

配置：

- Layers：4；
- Hidden dim：512；
- Heads：8。

---

# 28. Action Loss

连续动作：

\[
L_{continuous}
=
SmoothL1(
\hat A,A
)
\]

gripper 根据数据类型：

连续：

\[
SmoothL1
\]

离散：

\[
BCE
\]

最终：

\[
\boxed{
L_A
=
L_{continuous}
+
\lambda_g^A L_{gripper}
}
\]

---

# 29. 两阶段训练

本版本不建议一开始同时使用全部监督。

---

## Stage A：World Transition Pretraining

训练：

- U Predictor；
- Context Fusion；
- Visual Decoder；
- State Decoder。

冻结：

- DINOv3；
- Text Encoder。

Loss：

\[
\boxed{
L_{StageA}
=
L_V
+
\lambda_S L_S
}
\]

第一版：

\[
\lambda_S=1.0
\]

此阶段暂不使用 action loss。

目的：

\[
\boxed{
\text{先让 U 成为 World Transition Representation}
}
\]

而不是 action latent。

建议：

\[
50K\sim100K
\]

steps。

---

# 30. Stage B1：Action Head Warm-up

加载 Stage A checkpoint。

冻结：

\[
U\ Predictor
\]

只训练：

\[
D_A
\]

目标：

\[
U,S_t\rightarrow A
\]

Loss：

\[
L=L_A
\]

建议：

\[
5K\sim10K
\]

steps。

目的：

> 检查 Stage-A U 本身是否已经包含足够 action-related information。

---

# 31. Stage B2：Joint Executable Grounding

随后解冻 U Predictor。

联合优化：

\[
\boxed{
L_{StageB}
=
L_V
+
L_S
+
\lambda_A L_A
}
\]

第一版：

\[
\boxed{
\lambda_A=0.25
}
\]

即 action supervision 权重显著低于 world-transition supervision。

目的：

\[
\boxed{
\text{Action grounds U but does not redefine U}
}
\]

---

# 32. Optimizer

推荐：

\[
AdamW
\]

配置：

\[
weight\ decay=0.05
\]

Stage A：

\[
LR_U=1\times10^{-4}
\]

\[
LR_{decoder}=1\times10^{-4}
\]

Stage B：

Action Head：

\[
1\times10^{-4}
\]

U Predictor：

\[
\boxed{
2\times10^{-5}
}
\]

即联合 action grounding 时显著降低 U 的更新速度。

---

# 33. Training Schedule

推荐：

- bf16；
- gradient clipping = 1.0；
- warmup = 2k steps；
- cosine learning-rate decay；
- effective batch size：

\[
128\sim256
\]

windows。

第一轮无需追求完整 scaling。

优先：

\[
\boxed{
\text{小规模验证 representation 是否成立}
}
\]

---

# 34. 第一阶段必须监控的指标

训练期间每个 epoch / evaluation interval 记录：

### Visual

\[
L_V
\]

以及：

\[
CosSim(
\hat F_k,F_k
)
\]

### Robot State

\[
MSE_{\Delta EEF}
\]

\[
MSE_{\Delta joint}
\]

\[
MSE_{\Delta base}
\]

### Action

Stage B：

\[
Action\ MSE
\]

\[
Action\ R^2
\]

---

# 35. U Intervention Test

这是第一版必须加入的验证。

正常：

\[
D(F,u)
\]

对比：

\[
D(F,0)
\]

以及：

\[
D(F,shuffle(u))
\]

其中：

\[
shuffle(u)
\]

来自 batch 内其他样本。

分别测：

\[
Visual\ Recon
\]

和：

\[
State\ Recon.
\]

要求：

\[
\boxed{
Loss_{correct-U}
<
Loss_{shuffle-U}
}
\]

且：

\[
Loss_{correct-U}
<
Loss_{zero-U}
\]

否则说明 decoder 可能没有真正使用 U。

---

# 36. Frozen-U Probe

完成 Stage A 后：

\[
\boxed{\text{Freeze U}}
\]

训练轻量 probe。

---

## 36.1 Action Probe

\[
U\rightarrow A_{t:t+H}
\]

指标：

\[
MSE\downarrow
\]

\[
R^2\uparrow
\]

---

## 36.2 State Probe

\[
U\rightarrow
\Delta EEF
\]

\[
U\rightarrow
\Delta joint
\]

指标：

\[
MSE\downarrow
\]

---

## 36.3 Phase Probe

如果 RoboCasa 存在 subtask / stage 标签：

\[
u_k\rightarrow phase
\]

使用：

- Linear Probe；
或
- 2-layer MLP。

指标：

\[
Accuracy\uparrow
\]

---

## 36.4 Progress Probe

\[
u_k
\rightarrow
\frac{k}{K}
\]

用于判断 temporal progression 是否保留。

---

# 37. Temporal Representation Analysis

计算：

\[
d(u_k,u_{k+1})
\]

分析：

\[
u_1\rightarrow u_2\rightarrow u_3\rightarrow u_4
\]

是否具有：

- temporal smoothness；
- meaningful phase transition；
- grasp/contact 等边界变化。

如存在阶段标签，进一步统计：

\[
d_{within-phase}
\]

和：

\[
d_{cross-phase}
\]

理想情况下：

\[
\boxed{
d_{cross-phase}
>
d_{within-phase}
}
\]

---

# 38. Counterfactual Layout Test

为了排除：

\[
\text{fixed trajectory shortcut}
\]

选取同一 task：

\[
L
\]

构建不同 object layout：

\[
I_t^A,
I_t^B
\]

其中语言保持一致，但目标物体位置发生变化。

得到：

\[
U^A
=
P(I_t^A,S_t,L)
\]

\[
U^B
=
P(I_t^B,S_t,L)
\]

分析：

\[
d(U^A,U^B)
\]

并检测：

\[
U\rightarrow\Delta EEF
\]

预测是否随目标物体位置正确改变。

这是判断 U 是否真正 visual-grounded 的核心实验之一。

---

# 39. Baseline 1：Endpoint Future U

保留旧版 endpoint baseline。

输入 GT：

\[
F_t,F_{t+H}
\]

以及：

\[
S_t,S_{t+H}
\]

得到：

\[
U_{endpoint}
\]

用于分析：

> predictive U 与传统 post-hoc endpoint transition latent 的信息差异。

注意该模型不作为 deployable policy，只作为 representation baseline。

---

# 40. Baseline 2：Action Latent

训练一个简单 action autoencoder：

\[
A_{t:t+H}
\rightarrow
z_A
\rightarrow
A_{t:t+H}
\]

控制 latent capacity 与 U 接近。

用于比较：

\[
U_{world}
\]

和：

\[
z_{action}
\]

分别拥有多少：

- visual future；
- robot future；
- action；
- phase；

信息。

---

# 41. Baseline 3：Predictive Endpoint U

为了判断收益究竟来自：

- Predictive training；
还是
- Temporal \(K=4\)；

增加：

\[
K=1
\]

版本。

即：

\[
(I_{\leq t},S_{\leq t},L)
\rightarrow
u_{global}
\]

然后预测：

\[
F_{t+H},
S_{t+H}
\]

。

因此核心比较为：

| Model | Predictive | Temporal |
|---|---|---|
| Endpoint GT U | × | × |
| Predictive Endpoint | ✓ | × |
| **Predictive Temporal U** | ✓ | ✓ |
| Action Latent | × | action only |

这样能够真正分离：

\[
\boxed{
Predictive\ Benefit
}
\]

与：

\[
\boxed{
Temporal\ Structure\ Benefit
}
\]

---

# 42. 第一轮主结果表

## Table 1. Representation Quality

| Method | Visual Future ↑ | State Future ↑ | Action \(R^2↑\) | Action MSE ↓ | Phase ↑ |
|---|---:|---:|---:|---:|---:|
| Action Latent | | | | | |
| Endpoint GT U | | | | | |
| Predictive Endpoint | | | | | |
| **Predictive Temporal U** | | | | | |

---

# 43. Intervention Table

## Table 2. Is U Actually Used?

| Input | Visual Error ↓ | State Error ↓ |
|---|---:|---:|
| Correct U | | |
| Zero U | | |
| Shuffled U | | |

希望：

\[
\boxed{
Correct
\ll
Zero/Shuffle
}
\]

---

# 44. Temporal Ablation

第二轮再做：

\[
K\in
\{1,2,4,8\}
\]

比较：

| K | Visual ↑ | State ↑ | Action ↑ | Phase ↑ | Training Cost |
|---|---:|---:|---:|---:|---:|
| 1 | | | | | |
| 2 | | | | | |
| **4** | | | | | |
| 8 | | | | | |

暂不假设：

\[
K越大越好
\]

因为过细时间粒度可能使 U 退化成：

\[
\text{raw trajectory encoding}
\]

而失去 abstract transition representation。

---

# 45. 第一阶段成功标准

第一阶段不要求立即得到最高 rollout success。

首先要求满足以下 representation-level 条件：

### Criterion 1

\[
Predictive\ Temporal\ U
>
Predictive\ Endpoint
\]

在：

\[
Future\ State
\]

或：

\[
Action\ Probe
\]

上存在明显优势。

---

### Criterion 2

\[
Correct\ U
>
Shuffle/Zero\ U
\]

证明 decoder 确实依赖 U。

---

### Criterion 3

U 可以预测：

\[
\Delta EEF
\]

以及：

\[
Action
\]

但仍明显包含：

\[
Visual\ Future
\]

信息。

即 U 不能退化为：

\[
Action\ Latent.
\]

---

### Criterion 4

Counterfactual object layout 下：

\[
U
\]

以及：

\[
\Delta EEF
\]

预测会随场景变化正确改变。

从而排除：

\[
fixed\ trajectory\ shortcut.
\]

---

# 46. 第二阶段增强方向

只有完成上述验证以后，才依次增加：

### Enhancement A：Interaction Geometry

增加：

\[
L_G
\]

预测：

\[
\Delta p_{EEF}
\]

\[
\Delta p_{object}
\]

\[
\Delta p_{object-EEF}
\]

形成：

\[
\text{task-relevant physical transition}
\]

监督。

---

### Enhancement B：V-JEPA2 Target

比较：

\[
DINOv3
\]

与：

\[
V\text{-}JEPA2
\]

作为 future visual representation。

该实验作为：

\[
encoder\ ablation
\]

而非修改主方法。

---

### Enhancement C：Structured Action Manifold

只有当：

\[
U
\]

representation 很好，但：

\[
U\rightarrow Action
\]

仍成为性能瓶颈时，再考虑：

\[
LEAP/MSRQ\text{-style}
\]

structured action space。

---

### Enhancement D：Phase-aware Temporal Anchor

将均匀：

\[
K=4
\]

替换为：

\[
approach
\rightarrow
contact
\rightarrow
grasp
\rightarrow
transport
\]

等 phase boundary。

该方向留给后续，不进入初版。

---

# 47. 第一轮实现顺序

建议严格按照以下顺序实现：

1. RoboCasa window dataset；
2. 离线 DINOv3 feature extraction；
3. robot state/action normalization；
4. history context encoder；
5. temporal query U predictor；
6. state transition decoder；
7. visual transition decoder；
8. Stage-A training；
9. frozen-U probes；
10. zero/shuffle-U intervention；
11. Action Head；
12. Stage-B executable grounding；
13. predictive endpoint baseline；
14. action latent baseline；
15. counterfactual layout analysis；
16. 最后再进行 rollout。

不要一开始就同时实现完整 policy evaluation。

---

# 48. 第一轮默认配置

| 项目 | Default |
|---|---|
| Dataset | RoboCasa |
| Views | 3 views if available |
| History | 0.5 s |
| History RGB frames | 4 |
| Future horizon | 1.5 s |
| \(K\) | **4** |
| Visual encoder | Frozen DINOv3 |
| Visual pooling | 4×4 / view |
| Context dim | 512 |
| \(d_U\) | **512** |
| U tokens | **4** |
| State samples / interval | 4 |
| Stage A Loss | \(L_V+L_S\) |
| Stage B Loss | \(L_V+L_S+0.25L_A\) |
| Optimizer | AdamW |
| Stage A LR | \(1e-4\) |
| Stage B U LR | \(2e-5\) |
| Action Head LR | \(1e-4\) |
| DINO | Frozen |
| Text Encoder | Frozen |
| Precision | bf16 |

---

# 49. 最终模型定义

最终：

\[
\boxed{
U_t
=
P_\theta
(
I_{t-M:t},
S_{t-M:t},
L
)
}
\]

其中：

\[
\boxed{
U_t=
[u_t^1,u_t^2,u_t^3,u_t^4]
}
\]

每个：

\[
u_t^k
\]

表示未来第 \(k\) 个时间区间内：

\[
\boxed{
\text{task-conditioned world transition}
}
\]

并通过：

\[
U_t\rightarrow
\Delta V
\]

\[
U_t\rightarrow
\Delta S
\]

以及：

\[
U_t\rightarrow
A
\]

进行训练。

本阶段核心原则是：

\[
\boxed{
\textbf{Future defines U through supervision,
but future never enters the U predictor.}
}
\]

---

# 50. 本阶段最终研究问题

本实验最终希望回答：

> **机器人是否可以仅根据当前任务上下文，提前形成一个时间结构化的未来世界转移表示，并且该表示相比 endpoint transition 和 action-only latent，更完整地同时保存未来场景变化、机器人运动以及可执行动作信息？**

如果答案成立，则下一阶段才有必要进一步研究：

\[
\boxed{
\text{World Transition U}
\rightarrow
\text{Structured Action Manifold}
\rightarrow
\text{VLA Policy}
}
\]

而当前阶段的首要目标是证明：

\[
\boxed{
U
\text{ 本身是一个有效的 predictive world-transition representation。}
}
\]