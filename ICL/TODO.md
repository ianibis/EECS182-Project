# Section 5 · In-context model TODO

所有 section 5 的东西都在 `ICL/` 里，命令都在这个目录下运行 · 分支 `ICL`
环境 `ICL/.venv/` · 数据 `~/data/EECS182` · 缓存 `~/data/EECS182/icl_cache/export`
跑测试：`.venv/bin/python -m pytest tests -q`

```
ICL/
  TODO.md          本文件
  icl/             代码（cache, shifts, features_torch, encoder, model, ...）
  scripts/         一次性脚本（冒烟测试、sanity check、sim vs real、画图）
  tests/           pytest
  results/         csv、json、figs/、logs/
  requirements.txt
```

## 进度

| WBS | 任务 | 截止 | 状态 |
|---|---|---|---|
| — | 环境、数据、loader 冒烟测试、分支 | — | ✅ |
| 5.2a | trial 缓存 `icl/cache.py` | 11/12 | ✅ |
| 5.1.1 | 谐波 artifact generator `icl/shifts.py` | 11/12 | ✅ |
| 5.1.2 | 真实模板 artifact（`TemplateBank`，`kind='mix'`） | 11/13 | ✅ |
| 5.1.3 | sim vs real 检查 + 校准 amp | 11/14 | ✅ |
| 5.3.1 | 移植 Garg 代码 + 模型 `icl/model.py` | 11/14 | ✅ |
| 5.2b | episode sampler `icl/episodes.py` | 11/15 | ✅ |
| 5.3.2 | EEGNet token encoder `icl/encoder.py` | 11/15 | ✅ |
| 5.3.3 | overfit 测试 `scripts/overfit_tests.py` | 11/16 | ✅ |
| 5.4.1 | lr sweep → lr = 1e-4（`results/lr_sweep.md`） | 11/18 | ✅ |
| 5.4.1b | 正则化 sweep → train_stim，不加 dropout（`results/reg_sweep.md`） | 11/18 | ✅ |
| 5.4.2 | 5 个 seed | 11/20 | ⬜ |
| 5.5 | 真实数据评估 | 11/21 | ⬜ |
| 5.6 | context 对照 | 11/21 | ⬜ |
| 5.7 | diversity sweep（可选） | 12/4 | ⬜ |

## 已定的做法

- 训练数据：13 个 train split 的 no-stim session（去掉 250711-101424），只用 `part=='train'` 的 trial。**另外加入 26 个 train 日的 stim session**（`train_stim: true`，由 5.4.1b 决定，B 从 0.25 升到 0.34）。这些 session 只用 `part=='train'`；它们本身带真实 artifact，所以不再叠加模拟 artifact，只加 neural gain drift 和 channel dropout。val/test 日的 stim 数据始终不进训练。报告里要写明这一点，并用 5.6 的 `icl_simonly` 对照
- 模板：训练只用 train split stim session 的模板（SNR ≥ 1.5）；val/test 的模板只用于半合成评估
- 归一化：每个 session、每个通道做 robust z，统计量只用 20 个 pool trial；之后 clip 到 ±10
- `amp=1`（5.1.3 已校准）= 训练模板 RMS：3 Hz 为 2.87 µV，20 Hz 为 1.75 µV。真实 AM 超额功率对应 κ ≈ 1（中位数 0.97 / 1.03，各 session 在 0–2.7 之间）。训练用的 amp 范围是 [0.25, 10]
- **训练时，每个 trial 的窗口相对 go cue 随机平移 ±0.5 s**。原因：速度在时间上非常刻板（只用"go 后时间"预测 speed 就有 R² 0.55），如果 bin 位置固定等于 go 后时间，模型可以不看 LFP、只按位置平均 context 标签来"解码"。评估时用标准窗口
- 加一个只用标签的基线 `ctx_mean_profile`：把 k 个 context trial 的速度按 bin 平均后作为预测。ICL 必须赢过它才算用到了脑信号
- artifact 波形：`kind='mix'`，一半用谐波、一半用真实模板（其中一半是两个模板的凸组合）
- `trial_jitter` 每个 episode 抽 U(0, 0.6)；每个 trial 的 AM 起始相位随机
- 不模拟宽带功率升高，也不模拟通道 11/23 的谐波 pickup
- splits 不改；artifact 异常大的 test session（250616-102853、250609-110427、250609-111203）照常评估，另做一个去掉它们的敏感性分析
- encoder：流式 EEGNet（见 5.3.2），用 BatchNorm，不用 GroupNorm：GroupNorm 会把每个窗口的整体功率除掉，而 beta 功率下降正是运动信号
- reference：先用 `none`，3.1 出结论后再跟着改
- R²：每个 (session, k, draw) 在 query 的有效 bin 上算，vx、vy 各算一次再平均

## 任务

### 5.1.2 真实模板 artifact ✅
- `TemplateBank.load(('train',))`：训练用，12 个 3 Hz + 15 个 20 Hz 模板；`TemplateBank.load(('val', 'test'))`：评估用，7 + 11 个。都去掉了 SNR < 1.5 的
- 重采样用三角插值（`periodic_resample`），相位 0（包络峰）对齐到网格第 0 点，RMS 保持不变
- `sample_shift(rng, bank=..., kind='harmonic' | 'template' | 'mix')`；`trial_jitter` 每个 episode 抽 U(0, 0.6)
- 没做"随机相位平移"：每个 trial 的起始相位本来就是随机的，再平移一次是多余的
- 测试：`tests/test_templates.py` 有 9 个；全套 36 个测试全部通过。示例图 `results/figs/shift_examples.png` 已加上模板这一行

### 5.1.3 sim vs real 检查 ✅
- `icl/features_torch.py`：在 GPU 上复现 LMP 和 7 个频段，与 `features/export_none` 逐 bin 一致（误差 < 1e-3）
- `scripts/check_shift_vs_real.py` → `results/sim_vs_real.csv`、`amp_calibration.json`、`figs/sim_vs_real_psd.png`。覆盖 17 个有同天 OFF 的 train stim session（6 个 3 Hz + 11 个 20 Hz）
- 结论（PSD 用整个 4.5 s 缓存窗口）：
  1. 两个频率下，session 自己的模板都在 κ ≈ 1 时复现真实的 AM 超额功率（3 Hz 中位数 0.97，20 Hz 中位数 1.03），所以 `amp=1` = 模板 RMS
  2. 只有 250602-102701 和 250605-102648 的 κ = 0，这两个就是 README 标出的高频功率异常的 session。之前用 3.5 s 窗口时，估计噪声更大（20 Hz κ 为 0.82，3 Hz 有 4/6 测不出）
  3. 真实水平的 artifact 对解码几乎没有影响：OFF ridge 在 sim-ON 上的 R² 等于 OFF（约 0.05）。real ON 降到 0.03，但这不是 artifact 造成的，是 session 差异
  4. **OFF 解码本身就很弱**：13 个 OFF session、组内 ridge（9 lag，8 种特征），在留出 block 上 R² 中位数为 0.055（−0.03 到 0.16）；只用"go 后时间"的平均曲线是 0.09。改解码 **speed** 的话，LFP ridge 为 0.40，只用时间为 0.55
- 踩过的坑（Zhen 3.1 也会遇到）：`RidgeCV` 的 GCV 按 bin 留一，选出的 α 太小，R² 变成很大的负数，要按 trial 做交叉验证；单目标 `predict` 返回 (n,) 而不是 (n,1)；特征要用 float64

### 5.3.1 移植 Garg 代码 + 模型 ✅
- `scripts/sanity_linreg_icl.py`（4 层 × 128，20k 步）：k ≤ 4 时误差贴着 least squares（例如 0.76 vs 0.75），k ≥ 7 时为 0.03（zero 估计器为 1.0），PASS。结果在 `results/sanity_linreg.csv`
- `icl/model.py`：`ICLDecoder`，序列 `X_1 Y_1 … X_n`，用 GPT2 causal mask，一次前向得到所有 k 的预测；参数约 1.0M（不含 encoder）；`masked_mse`
- `icl/encoder.py`：`token_windows`
- `tests/test_model_causality.py`：13 个测试（标签只影响后面的 trial；trial 内 LFP 因果；无效标签被忽略；参数量）。全套 53 个测试通过

### 5.2b episode sampler ✅
- 缓存改成 go 前 2.5 s 到 go 后 2.0 s（90 个 bin，2.0 GB），kin/valid 也存全部 90 个 bin；标准解码窗口是 bin 40–79
- `icl/episodes.py`：
  - `TrialStore`：把 session 的 trial 放到 GPU 上（训练集 0.33 GB）
  - `EpisodeSampler.sample(B)`：一个 session、一个 shift、21 个 trial（20 个 pool + 1 个 pool 外的 query）。每个 trial 的窗口随机平移 ±10 bin，用 pool 做 robust z，clip ±10，速度用 `VelocityScaler` 标准化
  - `eval_batch(store, session, k, draw, splits, shift=None)`：context 来自 splits 的 subsets，30 个 query，用 pool 归一化，窗口不平移
  - 模型输入窗口是 51 个 bin（2550 个样本）：40 个目标 bin + 11 个历史 bin（500 ms token 窗口 + 卷积的感受野）
- 速度：B=32 每批 30 ms；`tests/test_episodes.py` 有 8 个测试（和 numpy 参考实现逐元素一致，平移后 LFP 与速度仍然对齐，每个 episode 只调一次 `sample_shift`，eval 与 splits 一致，归一化只用 pool）
- 验证集 A/B 和真实验证的组织放到 5.4 的 `train.py` 里做

### 5.3.2 EEGNet token encoder ✅
- `EEGNetEncoder`（流式）：第一块整段 trial 只算一次，先做空间滤波（35→16），再做因果时间卷积（k=64），然后 BN、ELU；每个 token 取自己 500 ms 窗口内的输出，过第二块（AvgPool 4 → separable conv → BN → ELU → AvgPool 8）和 Linear → 128
- 和原版 EEGNet 的区别：时间卷积和空间卷积交换了顺序（都是线性的），因此去掉了两者之间的 BN，每个空间信号有自己的时间滤波器。严格因果
- 速度：逐窗口的原版 B=32 显存不够，B=16 要 816 ms/步；流式 + 先空间后时间，B=32 只要 **54 ms/步，4.2 GB**
- 参数：encoder 33k，整个模型 1.05M
- 备用 `FeatureTokenEncoder`（LMP + 7 个频段 → MLP），44 ms/步
- 测试：EEGNet token 因果；全套 64 个测试通过

### 5.3.3 overfit 测试 ✅
- L1：1 个固定 episode，R² 0.999，PASS
- L2：64 个固定 episode，R² 0.987，PASS
- L3：真实 LFP + 合成标签（每个 episode 一个随机 2×2 矩阵 × 两组通道 LMP），held-out R²：k=0 为 −0.04，k=2 为 0.974，k=20 为 0.984，PASS，说明模型确实在读 context。没有用真实速度做 L3，因为它几乎解不出来（见 5.1.3），会掩盖 ICL 是否工作
- 结果：`results/overfit/overfit_results.json`、`results/logs/overfit.log`

### 5.4 训练脚本（已写好）
- `icl/train.py` + `icl/configs/base.yaml`：`.venv/bin/python -m icl.train --name <run> [--lr ... --steps ... --seed ...]`，输出到 `runs/<run>/`（config、git hash、metrics.csv、ckpt_last.pt、ckpt_best.pt）
- 验证：A = 训练 session 的留出 block（k ≤ 9）；B = 3 个 val 日 OFF session + val 模板的 shift（选模型用：B 在 k=5 和 20 上的平均 R²）；real = 6 个 val stim session。B 和 real 同时记录 `ctx_mean_profile` 基线
- `icl/metrics.py`：`r2_score`（vx、vy 分别算再平均）、`context_mean_profile`
- 20 步冒烟测试通过；每次验证约 5 s。`ctx_mean_profile` 在 k=20 时：B 为 0.07，real 为 0.09
- 新配置项：`dropout`（transformer）、`enc_dropout`（EEGNet，默认 0.25）、`train_stim`（true/false）
- ⚠️ GPU 1 有时会被另一个项目（supermimic）占用；跑之前先 `nvidia-smi` 看一下

### 5.4.1 lr sweep ✅
结果见 `results/lr_sweep.md`。三个 lr 的峰值都在 0.25 左右；3e-4 和 1e-3 在 2k 步以内就把 520 个训练 trial 背下来了。选定 lr = 1e-4，因为它在 6k 步后进入平台，不依赖选到一个好的 checkpoint。原始计划如下：
**做什么**：选学习率。
**命令**（在 `ICL/` 下运行）：
```bash
for lr in 1e-4 3e-4 1e-3; do
  CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python -m icl.train --name sweep_lr$lr --lr $lr --steps 10000 \
    > runs/sweep_lr$lr.log 2>&1 &
done
.venv/bin/python scripts/summarize_runs.py        # 看结果（随时可以跑）
```
**怎么做**：
1. lr ∈ {1e-4, 3e-4, 1e-3}，seed 0，每个 10k 步（GPU 0 上可以同时跑 3 个，每个约 4 GB）
2. 配置：AdamW (0.9, 0.95)，weight decay 0.05，1k 步 warmup + cosine，batch 32，bf16，grad clip 1.0
3. 每 1000 步记录验证 A/B 在 k ∈ {0, 2, 5, 10, 20} 上的 R²；每 2000 步记录真实 val stim 在 k = 5、20 上的 R²
4. 按验证 B 在 k = 5 和 20 上的平均 R² 选；最优值在边界上就往外再加一个点

**完成标准**：`results/lr_sweep.md`，选定值写回 `icl/configs/base.yaml`。

### 5.4.1b 正则化 sweep ✅
**结果**（`results/reg_sweep.md`）：train_stim 胜出，B 从 0.248 升到 0.339，real 从 0.191 升到 0.295。只有 OFF 数据的验证集 A 也从 0.19 升到 0.27，说明提升主要来自 trial 数量（520 → 1560）。dropout 让 B 低 0.02～0.03；小模型上限更低。选定 train_stim = true、dropout = 0，已写回 base.yaml。

原始计划：
**做什么**：在 lr = 1e-4 下解决数据量太小（520 个 trial）的问题。基线是 `sweep_lr1e-4`：B 0.248，real 0.191。
**命令**（2 张卡各跑 2 个，15k 步，约 45–50 min）：
```bash
run() { CUDA_VISIBLE_DEVICES=$1 nohup .venv/bin/python -m icl.train --name $2 --steps 15000 "${@:3}" > runs/$2.log 2>&1 & }
run 0 reg_drop      --dropout 0.1 --enc_dropout 0.5
run 0 reg_stim      --train_stim true
run 1 reg_drop_stim --dropout 0.1 --enc_dropout 0.5 --train_stim true
run 1 reg_small     --dropout 0.1 --enc_dropout 0.5 --d_model 64 --n_layer 2
.venv/bin/python scripts/summarize_runs.py runs/sweep_lr1e-4 runs/reg_*
```
**怎么选**：B 只有 3 个 session × 3 次 draw，差距小于 0.02 当作噪声。一个配置要算赢，需要 B 和 real 都不比基线差（下降 < 0.01），并且至少一个提高 ≥ 0.02。没有配置赢就用基线（最简单的那个）。选定的配置写回 base.yaml。

### 5.4.2 5 个 seed（11/18–20）
**做什么**：用选定的配置训练正式模型。
**命令**（配置全部在 base.yaml 里：lr 1e-4、train_stim、15k 步）。每张卡跑 3 个，约 70–75 min：
```bash
for s in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$((s % 2)) nohup .venv/bin/python -m icl.train --name seed$s --seed $s > runs/seed$s.log 2>&1 &
done
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python -m icl.train --name simonly_seed0 --seed 0 --train_stim false \
  > runs/simonly_seed0.log 2>&1 &
watch -n 30 .venv/bin/python scripts/summarize_runs.py --brief runs/seed* runs/simonly_seed0
```
**怎么做**：
- seed 0–4，每个 15k 步；输出在 `runs/seed{s}/`；导出每个 seed 的曲线图。
- `simonly_seed0` 是只用模拟 shift 训练的对照（同样配置、同样步数），用于 5.6。
**完成标准**：5 个 checkpoint + 曲线；把路径和 git hash 记进 `results/checkpoints.md`，之后不再改模型。

### 5.5 真实数据评估（11/20–21）
**做什么**：在 test 的 stim trial 上算 R² 随 k 的变化。
**怎么做**：
1. 先在 val 上把流程跑通
2. test：14 个 stim session × k ∈ {0, 2, 5, 10, 20} × 10 个 draw（k=0 只有 1 个）× 5 个 seed，每次预测 30 个 query trial；5 个 OFF reference session 也同样跑（用 OFF context）
3. 半合成：test 日的 5 个 OFF session + `eval_bank` 的模板，amp ∈ {0.5, 1, 2, 5, 10}；对照是在 k 个 context trial 上拟合的 ridge
4. 每个 (session, k, draw) 同时算 `ctx_mean_profile` 基线（k 个 context trial 的速度按 bin 平均）
5. 输出 `results/test_r2.csv`，列为 `method,seed,session,split,condition,am_freq_hz,off_reference,k,draw,amp,r2_vx,r2_vy,r2,n_valid_bins`；预测存到 `results/raw/icl_preds_seed{s}.npz`

**完成标准**：CSV 交给 6.2。

### 5.6 context 对照（11/20–21）
**做什么**：证明模型确实用到了 context 里的标签和 shift 信息。
**怎么做**：用 5.5 的同一批 test session，k ∈ {2, 5, 10, 20}，5 个 seed：
- `icl_shuffled`：context 的标签在 trial 之间打乱
- `icl_offctx`：context 换成同一天 OFF session 的 pool trial
- `icl_query_swap`：query 的 LFP 换成另一个 query trial 的 LFP（标签不变）。如果 R² 不降，说明模型没在用脑信号
- （可选）`icl_wrongday`：context 换成另一天的 ON session
- `icl_simonly`：`runs/simonly_seed0`，只用模拟 shift 训练（不含 train 日的真实 stim），其余设置完全相同。用来回答"只靠模拟 shift 能迁移多少"

**完成标准**：`results/controls_r2.csv` + 一张 R² vs k 的对比图。

### 5.7 diversity sweep（可选，11/24–12/4）
**做什么**：看 R² 如何随训练时见过的 shift 数量 M 变化。
**怎么做**：M ∈ {1, 4, 16, 64, 256, 1024, ∞}（预先采 M 个 shift，每个 episode 从中均匀选一个），3 个 seed；在 val/test 日 OFF session 上用 `eval_bank` 的新 shift 测试；画 R² vs M（k = 5、20 两条线），加上 least squares 的水平线。
**完成标准**：一张图。

## 要沟通的

- **全组（3.4，11/14）**：OFF 解码 velocity 的 R² 只有约 0.05，和只用时间的预测差不多；speed 可以解到 0.40。建议 3.4 讨论：是否把目标改成 speed（或同时报告），以及所有方法都报告"只用时间"和 `ctx_mean_profile` 这两个基线
- Ian：问一句有没有 shift generator spec（不阻塞）
- Kevin：归一化 + clip ±10；EEGNet 用 `icl/encoder.py` 里的 `EEGNetEncoder`（流式、BatchNorm），4.2 的 fine-tuning/LoRA 用同一个结构才能公平比较
- Zhen：同样的归一化；ridge 的三个坑（见 5.1.3）；要他 3.1 的 CAR 结论；分工（5.1.3 已完成，建议他做 5.6）
- 6.1/6.2：R² 定义、CSV 格式、去掉异常 session 的敏感性分析

## 数据速查

- 70 个 session，每个 50 个 trial；stim session 的 trial 全部是 ON。缓存了 67 个（7 月 11 日的 3 个没有 cm 速度）
- split：train 42（14 OFF / 13 个 3 Hz / 15 个 20 Hz），val 9（3 / 2 / 4），test 19（5 / 7 / 7）
- val/test：block 1–2 是 20 个 pool，block 3–5 是 30 个 query；`subsets[k]` 有 10 个 draw
- 缓存的每个 trial：lfp (35, 4500)，单位 µV，覆盖 go 前 2.5 s 到 go 后 2.0 s；kin/valid (90 个 bin)；标准解码窗口是 bin 40–79（go 前 0.5 s 到 go 后 1.5 s，`target_bins()`）；`part` ∈ train / val / pool / query
- 读取：`from icl.cache import load_cached, load_index`；`config.load_stats()` 给出 vel_std = 53.5 / 37.1 cm/s
- 有效标签 bin 约占 84%；约 6% 的 trial 有超过 15 sd 的瞬态
- 模板（`L.load_templates`）：`mean` (64, P)，P = 一个周期的样本数；`gain` (64, n_windows)；`channel_ids` 是 float，要转 int；属性有 `template_rms_uv`、`template_snr`、`am_freq_hz`
- AM 频率 2.9971 / 19.8911 Hz；训练模板在 35 通道上的 RMS 中位数为 2.87 / 1.75 µV，即 `amp=1`；LFP robust sd 约 25 µV
- OFF 组内 ridge 的 R²：velocity 约 0.05，speed 约 0.40；只用时间的预测：velocity 0.09，speed 0.55
- 模板的相位网格：3 Hz 有 334 个点，20 Hz 有 50 个点，`phase_bins` 是 −π 到 π 的 bin 中心；模板在周期内已经去均值
