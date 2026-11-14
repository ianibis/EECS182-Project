# 5.4.1b 正则化 sweep（2026-11-14）

lr 1e-4，seed 0，15k 步，两张卡各跑 2 个，每个约 40–44 min。对照组是 `sweep_lr1e-4`（10k 步，配置相同，只是没有下面这些改动）。
- **B** = 选择指标：B 在 k=5 和 k=20 上的平均 R²。
- **real** = val stim 在 k=5 和 k=20 上的平均 R²。

| run | 改动 | B 最优（step） | B 最后 | real 最优 | real 最后 | A k=9 最后 |
|---|---|---|---|---|---|---|
| sweep_lr1e-4 | — | 0.248 (6k) | 0.243 | 0.191 | 0.185 | 0.190 |
| reg_drop | dropout 0.1，enc_dropout 0.5 | 0.230 (6k) | 0.205 | 0.196 | 0.156 | 0.192 |
| reg_small | 同上，再加 d 64、2 层 | 0.221 (12k) | 0.220 | 0.190 | 0.190 | 0.190 |
| reg_drop_stim | dropout + train_stim | 0.307 (14k) | 0.305 | 0.276 | 0.264 | 0.229 |
| **reg_stim** | **train_stim** | **0.339 (13k)** | **0.331** | **0.295** | **0.286** | **0.273** |

`reg_stim` 最优 checkpoint（第 13k 步）在不同 k 下的 R²：

| | k=0 | k=2 | k=5 | k=10 | k=20 |
|---|---|---|---|---|---|
| B | 0.24 | 0.35 | 0.34 | 0.35 | 0.34 |
| real（第 14k 步） | 0.21 | 0.29 | 0.29 | 0.29 | 0.28 |

只用 context 标签的基线 `ctx_mean_profile` 在 k=20 时，B 为 0.07，real 为 0.09。

**结论**
- **train_stim 赢了，而且优势很大。** B 提高 0.09，real 提高 0.10，远超 0.02 的噪声门槛。15k 步时还没有过拟合，B 在 13k 步左右进入平台。
- **提升主要来自数据量，而不只是来自"见过真实 artifact"。** 只有 OFF 数据的两个验证集也一样提升了：A（训练 session 的留出 block）从 0.19 升到 0.27，B（val 日 OFF + 模拟 shift）从 0.25 升到 0.34。训练数据从 13 个 session、520 个 trial 增加到 39 个 session、1560 个 trial，原来的瓶颈是 session 和 trial 太少。
- **dropout 有害。** 不管加不加 train_stim，加了 dropout 的 B 都低 0.02～0.03。小模型稳定，但上限更低。
- **选定配置**（已写回 base.yaml）：train_stim = true，dropout = 0，enc_dropout = 0.25，d 128 × 4 层，lr 1e-4，15k 步。

**对报告的影响**：模型的训练数据现在包括 train 日真实 stim 的 trial；val/test 日的 stim 数据仍然没有进入训练。这需要两个对照：
1. 用同样配置但只用模拟 shift 训练的模型（`simonly_seed0`，和 5 个 seed 一起跑）。
2. 5.6 里比较两种模型在 test stim 数据上的表现。
