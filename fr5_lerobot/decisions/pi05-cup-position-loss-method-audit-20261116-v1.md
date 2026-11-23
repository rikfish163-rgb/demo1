# FR5/pi0.5 位置损失与训练方法核对

更新：2026-11-17。terminal 训练运行 v11 使用源码 v14，已完成 256 步、exit0；step128／256 完整包均已保存。GPU 工作按用户要求暂停，terminal 推理结果待评。clean v9/step256 完成训练 exit0，主套件离线及原 fit 首抽专项已评分、未放行。terminal v9／v10 第二步 OOM、exit1 的失败记录保留，完整实机抓放成功仍为 0。

## 结论

v8 coverage 的 6×XYZ 对称 Huber 确实参与训练并传递位置梯度。pickup wait mask 只移除夹爪监督；clean_error 的符号、sample_time 因子、归一化半跨度与逐点动作重建一致；fixed condition 在训练和服务中走相同路径。公式、mask、量纲换算和梯度传递一致。

有证据的剩余问题是训练目标与最终采样输出的区别：位置正则监督随机噪声时刻的一步 clean 估计，漂移指标来自多步积分结束后的动作。coverage 已覆盖原先遗漏的静止输入，仍有 31/120 个静止输入抽样的第三点预测移动超过 2mm，31 个前三点夹爪均为 100%。已有优化器、梯度与保存参数证据确认模型发生了更新。

同权重 10/20 步正式配对对照中，静止漂移为 31/120→32/120，endpoint error 中位数为 0.998→1.045mm，延迟为 0.314→0.527s。增加推理步数没有改善这批输入。

**terminal XYZ 使用 10 步完整采样结束后的前三点位置正则，保留系数 6、容差 0.25 mm 和其余训练设置。** 监督对象是最终动作，收益和计算成本由同起点配对评估确定。

新 phase 输入上的 clean v9/step256 仍有 31/120 次静止漂移，两提示相同；原 fit100 首抽 prefix3 检出 4 次正常闭爪迟延和 8 次支撑后迟开。CPU 生产选择器首抽回放中，562/600 次原样选中第一抽、38 次需要真实后续抽样；4 次迟闭仍被选中，8 次迟开均需额外抽样，最终部署结果未知。主套件的已知时序违规 0 未覆盖该专项。[clean 评分](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json) · [首抽评分](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json) · [首抽筛选回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-first-proposal-replay-20261117-v1.json)

迟闭的两个原观测中，TCP 距示范夹持参考仅约 0.054／0.203 mm，但原 `tracking.rim_minus_jaw_mm=null`、`visual=missing`，使 `close_at_pickup=false`，要求及时闭爪的候选筛选分支未触发。v16 已恢复 source5-frame183 的 rim 和该帧的闭爪分支 readiness；source3-frame170 仍 missing、readiness false。共同训练配置、归一化和四类代表参数更新已核验，现有具体问题包含监督对象、观察字段缺失与评分覆盖，不能将失败全部归因于八条数据量不足。v16 尚未放行，terminal 方法收益及新观测下的模型时机仍未评估。[迟闭明细](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-close-first-selection-detail-20261117-v1.json) · [readiness](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json)

## 当前终点监督验证

服务器根目录 `R=/home/data4/hts/demo1/fr5-learning/`。当前训练运行为 `R/checkpoints/pi05-fr5-cup-terminal-xyz-terminal-n3-20261116-v11/`，实际源码为 `R/src/fr5-pi05-cup-terminal-xyz-adam-offload-20261116-v14/`。冻结的 input-coverage 源码 v11 对应历史 v8 采样实验。

### 原生输出与梯度

v12 原来源 0／2／5 的完整 50×7 输出均与原生采样逐元素一致。来源 0 正损失为 0.13105054199695587，视觉、语言、完整动作专家和投影均有有限非零梯度；来源 2／5 最大 XYZ 误差为 0.239223／0.164646 mm，在原 0.25 mm 容差内，损失和梯度为零。原 probe 的 exit1 与按原容差通过的 assessment 分别保留。[原 probe](../results/pi05-cup-terminal-xyz-gradient-probe-20261116-v1.json) · [原回执](../results/pi05-cup-terminal-xyz-gradient-execution-20261116-v1.json) · [assessment](../results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json)

v13 来源 0 实测 probe exit0，完整 50×7 的 350 个元素与原生采样一致，四类组件梯度有限非零，峰值 allocated 为 **19,507,052,544 bytes**，optimizer_steps=0。[v13 probe](../results/pi05-cup-terminal-xyz-gradient-probe-20261116-v2.json) · [回执](../results/pi05-cup-terminal-xyz-gradient-execution-20261116-v2.json)

### 训练与内存

| 运行 | 源码／GPU | 已记录优化步 | 结果 |
| --- | --- | ---: | --- |
| clean v9 | v12／GPU8 | 256/256 | exit0；主套件离线已评分，未放行 |
| terminal v9 | v12／GPU9 | 1/256 | 第二步 OOM、exit1，训练失败 |
| terminal v10 | v13／GPU9 | 1/256 | 第二步 OOM、exit1，训练失败 |
| terminal v11 | v14／GPU9 | 256/256 | exit0，step128／256 完整包保存；GPU 评估暂停 |

四次运行共用 v7 fixed128 起点、fresh optimizer、seed `20261123`、LR `1e-6`、batch 1 × accumulation 4；训练只用原确认来源 0–6，来源 7 固定留出，失败只评估。v9 回执中的 completed 是进程结束，terminal exit1 不计训练完成。[v9 回执](pi05-cup-terminal-xyz-paired-training-execution-20261116-v9.json) · [v10 回执](pi05-cup-terminal-xyz-checkpointed-training-execution-20261116-v10.json)

v10 首步 384 个张量梯度有限非零：vision 108、language 72、expert 200、projection 4；首个 optimizer 成功，峰值 20,279.75 MiB。逐 denoise checkpoint 加 flow 先反传仍在第二步 OOM。

AdamW CPU 暂存 helper 已完成，GPU9 四项 OptimizerMomentTests 全通过，含六步 CUDA 参数及状态逐元素一致；8 MiB 测试 moment 释放／恢复 allocated 各 **8,388,608 bytes**。[校验结果](../results/pi05-cup-terminal-adam-offload-validation-20261116-v1.json)

v11 已完成 256 步、exit0。256 步指标中的 moment_bytes 均为 **3,485,335,808 bytes（约 3.25 GiB）**，记录峰值 **20,279.75 MiB（约 19.8 GiB）**；前 8 步 poststep allocated 为 **11,235.74 MiB**。首步 loss **0.15538615779951215** 和四类组件梯度 norm 与 v10 首步完全相同。实际日志为 `R/logs/pi05-fr5-cup-terminal-xyz-terminal-n3-20261116-v11.log`。[v11 计划](pi05-cup-terminal-xyz-adam-offload-training-plan-20261116-v11.json) · [v11 回执](pi05-cup-terminal-xyz-adam-offload-training-execution-20261116-v11.json)

CPU 配对核验为 `paired_training_consistent`，确认 50 项共同配置、全部 256 步实际采样及归一化张量一致。每版抽样 1,024 次，来源 7 为 0；256 步梯度均为有限正值，视觉 LoRA、语言 LoRA、动作专家和输出投影各一项代表参数均有有限非零更新。共同配置错配与完全没有更新的解释不符合这些核验结果。[配对核验](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-paired-training-audit-20261117-v1.json)

### 输入与评估状态

正式新输入 219/219 原 source／frame／time 匹配，共 215 个独立原帧；仅 heldout 2 案与 fit 5 案改变 phase，共 12 个字段值。原 JPEG、时戳、反馈、history、task、targets、tracking 和其他时钟不变，54 条失败保持 evaluation_only。[输入摘要](../results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json)

clean 十二种子主套件所有命令 exit0，状态为 `offline_assessed_not_released`；`controller_verified=false`、`robot_commands=0`、`task_success_verified=false`。原训练静止 120 组合中 31 次超过 2 mm 漂移，31 次的前三点夹爪均为 100% 全开，两提示相同。selected prefix3 已知时序违规 0；六种子真实五秒 fit 提前下降 0/126，最大下探 0.4136131 mm。来源 7 的 48 个无效反馈样本仍未评。[离线评分](../results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json) · [执行回执](pi05-cup-terminal-xyz-clean256-evaluation-execution-20261116-v9.json)

原 fit100 的闭／开爪专项已评的是 raw-first 原始首抽，100 案例 × 六种子，共 600 次。prefix3 已评 498、未评 102、失败 12，其中正常闭爪迟延 4、支撑后迟开 8，迟开可评 48；prefix10 失败 49。CPU 回放只使用第一抽，未产生第二／第三抽，实际最多三候选的最终 fit 评分仍待评。该证据用于定位模型首抽时序缺陷。[首抽评分](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json) · [分类与评分限制](/home/dianrobot/demo1/fr5_lerobot/PI05_CUP_RUNBOOK.md#正式输入与评估进度)

实际部署模式 fit evaluator 已实现并通过 26 项 CPU 测试，使用同 fit100／六种子和每 case／seed 重置，复用生产 `choose_proposal` 最多三候选原样选择。第一抽与 selected 均保存完整 50×7 和完整 request，原 raw true5s fit 评分保留。工具目录为 `R/diagnostics/pi05-cup-terminal-xyz-fit-selection-20261117-v10/`；GPU 生成及与原 raw600 完整首抽精确配对暂停。[CPU 核验](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json)

terminal step128／256 完整包均已保存，正式 GPU 评估按用户要求暂停；尚无 terminal 正式推理结果。[评估计划](pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json) · [GPU 暂停记录](pi05-cup-gpu-suspension-20261117-v1.json)

Gemini 连续 12 张不同新 RGB-D 的静止杯口定位均为 found，本轮机械臂与夹爪命令均为 0。定位证据位于 `R/audits/pi05-cup-localization-diagnostic-20261116-v3.json` 及同名目录；完整实机抓放成功仍为 0。

### 当前杯沿观察适配

v16 源为 `R/src/fr5-pi05-cup-rim-refit-fallback-20261117-v16/`。它复用 shared ellipse helper，仅在原候选不能形成有效观测时补充拟合候选，优先保留原有效候选，原几何与深度门槛保留。旧权重和正式输入不变，重放不进入训练。

| 原八条 4,209 帧 CPU 对照 | observed 旧→新 | 恢复／丢失 | 共同检出最大坐标差 | 状态 |
| --- | --- | --- | --- | --- |
| v15 | 2,961→3,844 | 884／1，source1-frame253 丢失 | 来源 3，约 53.58 mm | 保留未放行对照 |
| v16 | 2,961→3,712 | 751／0 | 来源 6，16.222308 mm | exit0，`cpu_observer_assessed_not_released` |

v16 各来源观察处理时间中位数约 5.93–6.17 ms。初版视觉 14 项、候选优先调整后 tracking 7 项、完整导出 6 子案例、selected-fit 26 项 CPU 核验通过。[v15 全帧回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-original-eight-replay-20261117-v1.json) · [v16 全帧回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2.json) · [CPU 验证记录](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json)

readiness 报告保留原 pose／history，只重算新 RGB-D 观测条件：source3-frame170 仍 missing，`close_branch_ready_before=false`、`close_branch_ready_after=false`；source5-frame183 恢复 rim，48 个 rim edge samples／30 个 raised depth samples，`false→true`。v15 两帧细查中 source5 有 6 个 depth sectors，source3 仅 3 个、低于原 4 扇区门槛。该报告的模型推理调用与设备命令均为 0，完整实机成功仍为 0。[readiness 报告](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json)

## 源码与数据依据

服务器数据根目录 R=/home/data4/hts/demo1/fr5-learning。

冻结源码：R/src/fr5-pi05-cup-input-coverage-20261116-v11。

v8 训练方法的源码依据为冻结 v11；terminal 方法发布源为 v12，当前 Adam offload 训练源为 v14，当前观察适配源为 v16，v15 保留未放行对照。本机实现：

- [train_stream_pi05.py](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)
- [serve_stream_pi05.py](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py)

运行时计算依据来自服务器 /home/hts/lerobot/src/lerobot/policies/pi05/modeling_pi05.py、policies/common/flow_matching.py、processor/normalize_processor.py。

数据为原确认 8 条示范，训练来源 0–6，来源 7 仅验证。v8 两组均从 v7 fixed step-000128 初始化，完成 256 个新优化器步，LR 1e-6，clip 1，batch 1、累积 4，vision/language LoRA rank 32，加完整 action expert。训练报告中 optimizer_restored=false，优化器为新 AdamW，betas=(0.9,0.95)、weight_decay=0.01。

[配对训练约束](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-input-coverage-paired-training-invariants-20261116-v8.json)记录两组共同配置及来源划分；[实际采样记录](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-input-coverage-sampling-20261116-v8.json)记录 29/29 个静止锚点、64 次专项抽样、来源 7 抽样为 0。

## 6×XYZ 的实际计算与梯度

[训练分支](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)由 coupled-support sampling profile 启用 coupled_support、support_focus、boundary_loss，并用实际 action quantiles 计算 cup_action_scale。[训练调用](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)把这些参数传给 masked_flow_loss。coverage 和 control 报告均记录：

- cup_boundary_loss=stable_setdown_active_prefix_pickup_wait_mask_and_xyz_prefix_v6；
- prefix_training=uncached；
- first_action_weight=12、executed_prefix_weight=6、vertical_weight=4、gripper_weight=4。

对应[报告位置](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-input-coverage-coverage256-training-20261116-v8.json)与[缓存字段](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-input-coverage-coverage256-training-20261116-v8.json)。

[XYZ 项](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)计算：

~~~text
e_mm[k,xyz] = 1000 × k × action_scale[xyz] × clean_error[k,xyz]
r[k] = sqrt(sum_xyz(e_mm[k,xyz]^2) + 1e-8)
L_xyz_numerator = 6 × sum_k weights[k,X] × H(max(r[k]−0.25mm, 0) / 1mm)
~~~

H 是 beta=1 的 smooth L1。误差半径不超过 0.25mm 时梯度为 0；0.25–1.25mm 为二次区；超过 1.25mm 为线性区。这里是一个 XYZ 欧氏半径正则，对方向对称。6 是该半径项的系数，并非每个轴独立各加 6 个 Huber。

coupled_support 分支里的 XYZ 项没有 actual_open、预测夹爪开度或 pickup-ready 条件。预测保持全开不会关闭这个位置项。

[forward hook](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)取得 action_out_proj 的速度输出；[clean_error](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)直接使用这个输出，没有 detach。运行时 modeling_pi05.py 第 462 行的 checkpoint 使用 use_reentrant=False；第 609 行经同一机制计算 action_out_proj，输出保留计算图。第 801 行 height_excess.detach() 只隔离额外夹爪惩罚中的高度通路，不作用于 XYZ 正则。

CPU 按冻结损失函数及上述 checkpoint 路径核对前三点各为 2mm clean 位置误差、全部 pickup mask 为 True 的情况。使用实际保存的量纲，sample_time=0.5：

| 误差轴 | 正负方向增加的 XYZ 损失 | 对三个速度输出的梯度绝对值 |
|---|---:|---|
| X | 0.195016 | 0.060110，0.060110，0.090164 |
| Y | 0.195016 | 0.052913，0.052913，0.079369 |
| Z | 0.195016 | 0.065577，0.065577，0.098365 |

正负误差损失相同、梯度方向相反；三个位置点的梯度均非零；被 mask 的夹爪梯度为 0。这确认损失数值与自动微分均有效。

## Temporal weighting 与 wait mask

[temporal 权重](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)先给前三点设 6，再把第一点覆盖为 12。实际前三点是 12、6、6；第 4–50 点仍是 1。维度权重为 X=1、Y=1、Z=4、RPY=1、夹爪=4。

[mask 实现](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)复制 weights 为 supervision，并只对 supervision[...,6] 执行 masked_fill(0)。因此：

- X/Y/Z 的 flow MSE 与 6×XYZ 项都保持监督；
- 使用 weights[...,0] 给欧氏 XYZ 项加权是公共的 valid×temporal 系数，没有从夹爪 mask 取权重；
- 传入 boundary penalty 的 supervision 使相关夹爪额外惩罚也被遮罩；
- 分母仍为 weights.sum()，没有随夹爪 mask 重新缩放。

50 个目标全部有效时，时间权重总和为 12+6+6+47=71，维度权重总和为 13，分母为 923。[累积损失 normalizer](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)与这个分母一致，[微批次累积](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)按实际有效目标权重合并。

物理位置换算还乘 k，因此在线性 Huber 区、同一轴和相同噪声时刻下，对速度残差的时间系数为 12×1、6×2、6×3。CPU 结果中第三点梯度较前两点大 1.5 倍正是这个换算。

冻结 cup_relative_stream_data.py 第 308 行的 pickup_wait_mask 使用观测判断 pickup-ready，并遮罩首次真实关闭前的 leading open targets。它是夹爪的操作等待处理。[batch_for](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)把该 mask 送入损失，没有改变原动作标签。

## clean_error、归一化与推理积分

设 a 为归一化后的真实动作，ε 为噪声，v 为模型速度。运行时 modeling_pi05.py 第 564 行的训练 forward 使用：

~~~text
x_t = tε + (1−t)a
u = ε−a
x_clean_hat = x_t − tv
x_clean_hat − a = t(u−v)
~~~

因此 clean_error=t(ε−a−v) 的符号和时间因子正确。它表示从时刻 t 用当前速度一步退到 0 的动作误差，不能用单个推理步长替代 t。

服务器 flow_matching.py 第 96 行的推理循环为 dt=−1/N，第 99 行取 t=1+step×dt，第 117 行执行 x←x+dt×v。v8 保存 config 的 N=10，RTC=None；训练时间分布为 Beta(1.5,1.0)×0.999+0.001。这些使用同一种 flow 时间方向。

运行时 normalize_processor.py 第 394 行用 q99−q01 作为分母，第 400 行反归一化：

~~~text
a_phys = q01 + (a_norm+1) × (q99−q01)/2
s = (q99−q01)/2
delta_a_phys = s × delta_a_norm
~~~

实际保存的 XYZ 半跨度为 1.541142、1.356627、1.681316 mm/归一化单位，与 cup_action_scale 一致。quantiles 非退化，eps 分支不介入这些维度。

冻结 relative_stream_actions.py 第 58 行对第 k 点使用：

~~~text
训练：a_tick[k] = (p_target[k]−p_anchor)/k
服务：p_output[k] = p_anchor + k×a_tick_output[k]
位置误差：p_output[k]−p_target[k] = 1000×k×s×delta_a_norm[k]  （mm）
~~~

这是同一锚点下每个未来点的平均每 tick 偏移。重建逐点乘 k；不做 cumsum，也不额外乘 6Hz。

[训练动作转换](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)在归一化前进行；[服务端](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py)先反归一化，再用原请求的 cup-relative measured state 重建。[actions_for_wire](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py)只做一次手工转换。保存 config 为 use_relative_actions=false，保存的前后 processor 没有启用官方 relative/absolute pair。

## fixed condition

[固定条件实现](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)保持前七个 noisy action 输入，将通道 7–31 替换为因果 action-clock 条件；非 fixed 分支才把条件加到对应噪声通道上。v8 的 clock 长度为 25，正好占满七个动作维度以外的通道。

[训练 batch](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)与[服务请求 batch](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py)都对 CUP_SETDOWN_FIXED_PROFILE 设置 fixed_cup_action_clock=True。[服务预测](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py)调用训练模块的同一 predict_action_chunk/action_clock_condition，在每次 denoise_step 中应用覆盖。CPU 核对中改变填充噪声和 diffusion time 后，固定通道保持相同，前七个动作通道保持各自输入。

fixed 与 clean-setdown 是两个设置关系：fixed 决定填充通道覆盖方式；clean_setdown 决定是否将 settled wait mask 扩展到执行前三点。[clean_setdown 选择](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)仅匹配 CLEAN_PROFILE，v8 FIXED_PROFILE 保留 active prefix 的 setdown 监督。所以训练报告里的 active_prefix 名称与 fixed condition 同时存在是正确行为。


已有固定条件配对报告在同一 v6 权重、相同其余条件下记录静止漂移 19/40→11/40、endpoint error 中位数 1.644→0.993mm。该机制已有独立证据，v8 训练与服务已使用它。

## 梯度与更新的实际证据

两组 metrics.jsonl 都完整保存 256 步，全部 gradient_norm 有限且非零，LR 全程为 1e-6；各有 243/256 步的裁剪前 norm 超过 1。[记录点](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)取得 clip_grad_norm_ 返回的裁剪前 norm，[首步组件汇总](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)在裁剪后执行。

两组首步日志相同：

| 组件 | 有梯度的张量数 | 裁剪后范数 |
|---|---:|---:|
| vision LoRA | 108 | 0.168125 |
| language LoRA | 72 | 0.399534 |
| full action expert | 200 | 0.637725 |
| action projections | 4 | 0.636721 |

梯度 dtype 与首步 AdamW 状态 dtype 都为 FP32。证据位置分别是 R/logs/pi05-fr5-cup-input-coverage-control-n3-20261116-v8.log 与 pi05-fr5-cup-input-coverage-n3-20261116-v8.log 的第 3、4 行。实际 optimizer.step 在[训练第 2453 行](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)执行。

直接比较共同父权重和两组 step-000256 保存张量，得到：

| 参数范围 | control 更新 RMS | coverage 更新 RMS | coverage 相对父参数 RMS |
|---|---:|---:|---:|
| 输出投影 XYZ 三行，3072 个参数 | 1.38351e-5 | 1.54429e-5 | 0.076823% |
| 输入投影 fixed 条件 25 列 | 1.37734e-5 | 1.28423e-5 | 0.006171% |
| expert 第 0 层 q_proj base | 1.74300e-5 | 1.58270e-5 | 0.046146% |
| expert 第 17 层 q_proj base | 1.74365e-5 | 1.92263e-5 | 0.053010% |
| language LoRA | 1.67894e-5 | 1.93493e-5 | 0.186660% |
| vision LoRA | 1.99979e-5 | 1.91629e-5 | 0.158673% |

两组 XYZ 输出投影的 3072 个 FP32 值全部改变。language LoRA 改变 70/72 个张量，vision LoRA 改变 108/108 个；继承的 expert LoRA 72 个张量全部保持原值，与[冻结选择](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py)一致。

将保存值分别舍入到 BF16 后，coverage 的 XYZ 输出投影有 43.522% 元素改变，fixed 条件输入列有 3.816% 改变，所抽取的第 0/17 层 expert q_proj 有 18.089%/20.908% 改变。这提供了更新幅度较小的数值依据，同时确认可训练路径发生了实际更新。输出投影总体只有 7/32 行改变，是只监督七个真实动作维度的预期结果；上述 XYZ 更新统计已排除 25 个填充输出行。

metrics 中 coverage 的 loss 中位数从首 32 步的 0.067051，到末 128 步的 0.080434；control 为 0.078001→0.065185。metrics 保存的是混合 flow/boundary 总损失，没有 XYZ 分项和积分终点损失。

clip 的统一缩放进入 AdamW 的一、二阶矩；更新不是简单的 LR×裁剪后梯度。所以 243 次 clip 本身不是更新不足的证明。保存参数说明更新较小，来源 7 的离线诊断与训练支持分层结果提供输出行为证据。

## 训练方法证据与 terminal 监督

服务器两个 evaluations/pi05-cup-input-coverage-{control,coverage}256-suite-20261116-v8/training-support-score.json 的 reports.original.unique_source_row_seed 使用相同的去重来源、行号和 seed 口径：

| 指标 | control256 | coverage256 |
|---|---:|---:|
| 抽样总数 | 312 | 312 |
| 原轨迹移动≤0.2mm 的抽样 | 120 | 120 |
| 这些抽样预测移动>2mm | 16 | 31 |
| 漂移抽样前三点夹爪全开 | 13/16 | 31/31 |
| 全部支持抽样 XYZ next3 MAE 中位数 | 0.301329mm | 0.338982mm |
| 全部支持抽样 endpoint error 中位数 | 0.864708mm | 0.989358mm |

全开分层计数采用已保存结果；其余指标来自保存的 score。正式 phase 对照提供早闭 1/48→0/48，96 个接触控制配对的完整动作相同。该结果支持终点位置监督。

已完成同一 v8 coverage 权重的 10/20 步单因素对照，运行 exit 0；324 对输入、初始 noise 与 RNG 一致，去重后为 312 对，其中静止抽样 120 对：

| 指标 | 10 步 | 20 步 |
|---|---:|---:|
| 静止抽样预测移动>2mm | 31/120 | 32/120 |
| 漂移抽样前三点夹爪全开 | 31/31 | 31/32 |
| endpoint error 中位数 | 0.998mm | 1.045mm |
| endpoint error 最大值 | 6.744mm | 7.142mm |
| 延迟 | 0.314s | 0.527s |

该对照没有观察到步数增加带来的位置收益，延迟增加约 68%；当前服务与终点正则均选择 N=10。上述对照指标采用已保存汇总口径，与前一表原正式 assessment 分开列示。

29/29 静止锚点已采到，专项抽样总计 64 次，占 1024 个训练样本的 6.25%。可达性与实际选取均已有证据。固定条件、位置梯度以及参数更新也均有证据。剩余的目标差异明确可见：

~~~text
现有 XYZ 正则的输入：t × (ε−a−v(x_t,t,c))
实际漂移评测的输入：a_hat_N − a
a_hat_N：从 ε 开始，按服务端的时间网格积分后的最终输出
~~~

前三点的 endpoint error 与执行时间 k/6 一致。现有物理正则在给定训练时刻对一步 clean 误差正确，但该误差没有逐步计算模型自己生成的中间状态和最终样本。

当前 terminal 方法将 XYZ 项替换为完整采样终点监督：

~~~text
L_xyz_end =
  6 / weights.sum()
  × sum_{k=1..3} supervision[k,X]
  × H(max(||1000×k×s_xyz×(a_hat_N[k,xyz]−a[k,xyz])||₂−0.25mm, 0) / 1mm)
~~~

其余 flow MSE、normal/gripper boundary 项、wait mask、标签、归一化、fixed condition、LoRA/full-expert 可训练范围与 v8 coverage 配置一致。共同初始化仍为 v7 fixed128，使用原确认训练来源 0–6，来源 7 用于验证。这样得到一个明确的位置损失单因素。


N=10，与现有服务配置和已完成的 10/20 配对结论一致。训练保留 256 个新优化器步、LR 1e-6 与 clip 1；实际更新已确认，现有混合 loss 与裁剪记录不足以单独支持增加训练步数。

取舍：位置正则直接惩罚最终漂移，包括预测全开分支；多步反向传播增加后缀计算和显存需求，增量约为 N 次 denoise 后缀的计算图。前缀可在同一有梯度图中预填一次，后缀可使用已有 checkpoint 机制。6、0.25mm 和时间/维度权重保留，使变化集中在监督对象。

验证目标使用相同来源、原始输入和 seed 的静止>2mm漂移率与各轴/逐点误差，并按夹爪全开分层；正式 phase、接触与返回位置指标评价其取舍。单个混合训练 loss 不作为判定依据。

## 已知问题

CPU 损失核对确认公式、mask 和梯度连通；没有测量 v8 坏抽样在真实大模型中的逐时刻 velocity 或 XYZ 分项梯度。保存参数比较覆盖所有投影、所有 LoRA 与选定 expert 张量，没有将抽样结果外推为完整 expert 的更新统计。

小幅参数更新、BF16 元素舍入比例和混合 loss 走势不能单独证明优化步数或 LR 不足；全开分层的相关性也不能单独证明夹爪导致位置漂移。现有证据支持直接监督终点位置这一监督选择，尚未证明漂移的唯一原因。

10/20 同权重推理对照及正式 phase 结果采用已保存汇总。终点正则的真实输出与梯度已验证；v11 已完成 256 步、exit0，step128／256 完整包保存，CPU 配对核验一致；四类参数更新核验各取一项代表张量，不是全部参数扫描。终点权重正式 GPU 推理评估暂停。clean 已有主套件和原 fit 首抽评分，静止漂移、迟闭及迟开仍有失败证据；实际选中候选 fit 专项与来源 7 的 48 个无效反馈样本未评。8 次迟开需后续真实抽样，不能据此判定部署成功或失败。终点位置收益及完整闭环时序效果尚未验证。

v16 CPU 全帧检出改善且未丢失原有效观测，但共同检出坐标仍相差最多 16.222308 mm，精度未证实；source3-frame170 仍 missing，source5 readiness true 尚未验证新观测下的模型动作。GPU 保持暂停，控制器与实机未验证，闭爪、五秒悬停及放稳后开爪时机不能记为已修好。v15 未放行对照保留。

唯一留出杯位距最近训练约 0.22 mm，现场杯位约 112.04 mm；标签 6 Hz 与 trial02 ACK 端点吞吐 2.128 Hz 的动作影响尚未因果分离。Gemini 12 张新帧仅验证静止定位，未验证毫米精度、运动遮挡跟踪或完整抓放。
