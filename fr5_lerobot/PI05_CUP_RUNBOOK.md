# 抓杯训练与评估

主数据为八条用户确认成功的完整教师示范，来源 0–6 训练、来源 7 留出。π0.5 完整实机抓放成功数为 **0**。历史对照为 v8 control；当前终点方法的 clean 配对对照为 v9/step256，主套件离线已评分、未放行。terminal v11 已完成 256 步、exit0，step128／256 完整包已保存，结果待评；GPU 工作按用户要求暂停。

服务器根目录 `R=/home/data4/hts/demo1/fr5-learning/`，SSH 主机为 `4090`。R 下的源码、权重、日志和原数据均属于服务器；本机文档与小证据位于 `/home/dianrobot/demo1/fr5_lerobot/`。

原示范拟合、观测缺失、执行时序与新杯位覆盖的具体原因及公开案例对照，见[当前原因分析](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-current-cause-analysis-20261117-v1.md)。用户确认八条示范后杯子和夹爪指尖配置未变。

## 数据与执行决定

数据验收使用真实观测、下发命令、完成反馈、用户评价和来源映射。以完整物理示范划分训练与留出，保留真实悬停时长、原始数据和验收绑定。RGB 是模型图像输入；深度用于杯沿和几何，8 维状态包含夹爪测量有效性。

当前使用杯中心图像、杯相对 TCP tick delta 与 fixed setdown 上下文。模型每次返回完整 `50×7` 原生采样，候选最多三条，只执行选中候选的原样前三点。ServoJ 节拍 8 ms，夹爪使用原生整数百分比和 1 个百分点保持规则。

## 已完成对照

v8 control 和 coverage 使用相同数据、初始化、优化器配置和固定 action-clock；二者均完成 256 步。比较的是“真实位移不超过 0.2 mm 的全开静止点，模型预测超过 2 mm”的次数。

| 指标 | control | coverage |
| --- | --- | --- |
| 全开静止漂移 | 13/120 | 31/120 |
| 可评分 selected prefix3 的时序违规 | 0 | 0 |
| 真实 5 秒训练拟合中的提前下降 | 0/126 | 0/126 |
| 放行 | 否 | 否 |

control 在该静止漂移指标较好。48/408 无效反馈样本未评估，整体结论仍为 inconclusive。对应证据：[配对评估](results/pi05-cup-input-coverage256-paired-assessment-20261116-v8.json)、[采样决定](decisions/pi05-cup-input-coverage-sampling-20261116-v8.json)、[训练不变量](decisions/pi05-cup-input-coverage-paired-training-invariants-20261116-v8.json)、[control 放行决定](decisions/pi05-cup-input-coverage-control256-release-20261116-v8.json)、[coverage 放行决定](decisions/pi05-cup-input-coverage-coverage256-release-20261116-v8.json)。

## terminal XYZ 与采样步数

[位置损失方法](decisions/pi05-cup-position-loss-method-audit-20261116-v1.md)对完整原生 10 步采样后的前三点 XYZ 施加监督，系数 6、容差 0.25 mm，保持完整 50 点输出。[正式梯度评估](results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json)确认一例正损失具有有限非零梯度，两例在容差内梯度为零；该证据验证可微监督，不代表任务成功。

配对训练从 v7 fixed-clock step 128 初始化，数据划分和采样分布相同，比较 clean 与 terminal XYZ。v9 clean 已完成 256 步和主套件离线评分、未放行；terminal v9、v10 的训练未成功完成；v14 源码的 v11 Adam offload 已完成 256 步、exit0，详见 [当前产物](CURRENT_PI05.md)。

[10/20 步对照](results/pi05-cup-coverage-v8-denoising-10-20-20261116-v2-summary.json)的静止漂移为 31/120 与 32/120，endpoint 中位数为 0.997962 与 1.044962 mm，推理耗时中位数为 0.314226 与 0.527414 秒。当前选择 10 步。

### 完整训练状态

四次运行共用 v7 fixed128 起点、fresh optimizer、seed `20261123`、LR `1e-6`、batch 1 × accumulation 4，来源 0–6 训练、7 固定留出；失败只评估。当前训练运行 v11 使用 `R/src/fr5-pi05-cup-terminal-xyz-adam-offload-20261116-v14/`；冻结的 `R/src/fr5-pi05-cup-input-coverage-20261116-v11/` 属于历史 v8 输入覆盖实验。

| 运行 | 源码／GPU | 已记录优化步 | 结果 |
| --- | --- | ---: | --- |
| clean v9 | v12／GPU8 | 256/256 | exit0，128／256 两包保存；主套件离线已评分，未放行 |
| terminal v9 | v12／GPU9 | 1/256 | 第二步 OOM、exit1，训练失败 |
| terminal v10 | v13／GPU9 | 1/256 | 第二步 OOM、exit1，训练失败 |
| terminal v11 | v14／GPU9 | 256/256 | exit0，128／256 完整包保存；GPU 评估暂停 |

v10 首步 `first_gradient` 的 384 个张量全部有限非零：vision 108、language 72、expert 200、projection 4；首个 optimizer 成功，峰值 20,279.75 MiB。逐 denoise checkpoint 加 flow 先反传仍在第二步 OOM。原 v9 回执的 `status=completed` 表示进程结束，terminal 的 exit1 不计训练完成。[v9 回执](decisions/pi05-cup-terminal-xyz-paired-training-execution-20261116-v9.json) · [v10 回执](decisions/pi05-cup-terminal-xyz-checkpointed-training-execution-20261116-v10.json)

AdamW CPU 暂存 helper 已完成。GPU9 四项 OptimizerMomentTests 全通过，含六步 CUDA 参数与 optimizer 状态逐元素一致；8 MiB 测试 moment 释放／恢复 allocated 各 8,388,608 bytes。[校验结果](results/pi05-cup-terminal-adam-offload-validation-20261116-v1.json)

v11 已完成 256 步、exit0，step128／256 完整包均已保存。256 步指标中的 moment_bytes 均为 **3,485,335,808 bytes（约 3.25 GiB）**，记录峰值为 **20,279.75 MiB（约 19.8 GiB）**。前 8 步 poststep allocated 为 **11,235.74 MiB**。首步 loss 为 **0.15538615779951215**，与 v10 首步完全相同，四类组件的梯度 norm 也完全相同。实际训练日志为 `R/logs/pi05-fr5-cup-terminal-xyz-terminal-n3-20261116-v11.log`。[v11 计划](decisions/pi05-cup-terminal-xyz-adam-offload-training-plan-20261116-v11.json) · [v11 回执](decisions/pi05-cup-terminal-xyz-adam-offload-training-execution-20261116-v11.json)

CPU 配对核验为 `paired_training_consistent`：50 项共同配置、全部 256 步实际抽样和两版归一化逐元素一致。每版共抽样 1,024 次，来源 0–6 为 149／145／138／173／136／148／135，来源 7 为 0。两版 256 步梯度均为有限正值；视觉 LoRA、语言 LoRA、动作专家和输出投影各一项代表张量均与父权重有有限非零差异。[配对训练核验](results/pi05-cup-terminal-xyz-paired-training-audit-20261117-v1.json)

2026-11-17 停止检查时训练已自然退出，本项目无 GPU 作业，未发送停止信号。GPU 评估按用户要求暂停，权重及日志保留。[GPU 暂停记录](decisions/pi05-cup-gpu-suspension-20261117-v1.json)

### 原生采样与真实梯度

v12 原来源 0／2／5 的完整 50×7 输出均与原生采样逐元素一致。来源 0 四类组件梯度有限非零；来源 2／5 最大 XYZ 误差为 0.239223／0.164646 mm，在原 0.25 mm 容差内，损失和梯度为零。原 probe 的 exit1 和按原容差通过的 assessment 分别保留。[原 probe](results/pi05-cup-terminal-xyz-gradient-probe-20261116-v1.json) · [原回执](results/pi05-cup-terminal-xyz-gradient-execution-20261116-v1.json) · [assessment](results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json)

v13 来源 0 实测 probe exit0，完整 50×7 的 350 个元素与原生采样一致，四类组件梯度有限非零；峰值 allocated 为 **19,507,052,544 bytes**，optimizer_steps=0。[v13 probe](results/pi05-cup-terminal-xyz-gradient-probe-20261116-v2.json) · [回执](results/pi05-cup-terminal-xyz-gradient-execution-20261116-v2.json)

### 正式输入与评估进度

219/219 原 source／frame／time 匹配，共 215 个独立原帧。仅 heldout 2 案和 fit 5 案改变 phase，共 12 个字段值；原 JPEG、时戳、反馈、history、task、targets、tracking 和其他时钟保留。54 条失败案例保持 `evaluation_only`。[输入摘要](results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json)

clean step256 的十二种子主套件所有命令 exit0，六种子真实五秒 fit 已评分；状态为 `offline_assessed_not_released`，`controller_verified=false`、`robot_commands=0`、`task_success_verified=false`。[执行回执](decisions/pi05-cup-terminal-xyz-clean256-evaluation-execution-20261116-v9.json) · [离线评分](results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json)

| 指标 | clean v9/step256 |
| --- | --- |
| 原训练静止组合预测漂移 >2 mm | 31/120，两提示相同；31 次漂移抽样的前三点夹爪均为 100% 全开 |
| selected prefix3 已知时序违规 | 0；总体 gate 为 inconclusive |
| 真实五秒 fit 提前下降 | 0/126，最大下探 0.4136131 mm |
| 来源 7 无效反馈样本 | 48 个仍未评 |

原训练 fit100 的闭／开爪专项使用原始首抽（raw-first），100 案例 × 六种子，共 600 次完整 50×7 预测，按名义 6 Hz 评分：

| 前缀 | 已评／未评 | 失败次数与分类 | 迟开可评次数 |
| --- | --- | --- | ---: |
| prefix3 | 498／102 | 12：正常闭爪迟延 `late_normal_close` 4，支撑后迟开 `late_supported_open` 8 | 48 |
| prefix10 | 528／72 | 49：迟闭 11、迟开 8、支撑高度以上开爪 20、稳定支撑前开爪 4、支撑高度以下开爪 4、目标运动时开爪 2 | 48 |

主套件的已知时序违规 0 未覆盖原 fit 的正常闭爪／迟开专项；首抽专项确认模型首抽存在这两类时序缺陷。该评分保持原标签、原始时间戳、反馈和 targets，没有设备执行。[首抽专项评分](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json)

同一 raw600 经生产 `choose_proposal` 做 CPU 首抽回放，真实模型推理调用和设备命令均为 0：

| 结果 | 次数 | 已知含义 |
| --- | ---: | --- |
| 直接选择首抽 | 562 | 完整 50×7 原样保留 |
| 需要后续真实模型抽样 | 38 | 最终候选尚未知 |
| 原迟闭首抽仍被选中 | 4 | `late_normal_close` 原样入选 |
| 原迟开首抽需要后续抽样 | 8 | `late_supported_open` 的最终部署结果尚未知 |

[首抽筛选回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-first-proposal-replay-20261117-v1.json)只回放已保存第一抽，未生成第二／第三抽；8 次迟开不能据此认定部署成功或失败。

4 次迟闭来自两个原观测，原 `tracking.rim_minus_jaw_mm=null`、`visual=missing`，生产 `close_at_pickup=false`，要求及时闭爪的候选筛选分支未触发：

| 原观测／case | 首抽 seed | TCP 距示范夹持参考 |
| --- | --- | ---: |
| source3-frame170，`source3-cycle1-close_at-1468` | 20261114 | 0.0542496 mm |
| source5-frame183，`source5-cycle1-close_at-2524` | 20261113／14／15 | 0.2033292 mm |

[迟闭明细](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-close-first-selection-detail-20261117-v1.json)及原 640×360 图像：[来源 3 帧 170](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-close-rgbd-20261117-v1/source3-frame000170.jpg)、[来源 5 帧 183](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-close-rgbd-20261117-v1/source5-frame000183.jpg)。两帧杯口可见与深度观测通过门槛是不同证据。

实际部署模式 fit evaluator 已实现并通过 26 项 CPU 测试。同一 fit100／六种子、每 case／seed 重置模型并设置随机种子，使用原 `model_request(case, 'setdown', profile)` 调生产 `choose_proposal`，带原桌面法向和 `check_boundaries=True`，最多抽取三条完整 50×7、选中回复原样保存。两份 FIT_SCHEMA 均保存完整 request：`fit-predictions.json` 保留实际第一抽，`fit-selected-predictions.json` 保留实际选中候选；原 raw true5s fit 评分保留。工具目录为 `R/diagnostics/pi05-cup-terminal-xyz-fit-selection-20261117-v10/`。真实权重下与原 raw600 的逐 case／seed 完整首抽精确配对、selected 专项评分仍随 GPU 工作暂停。[CPU 核验](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json)

terminal step128／256 完整包均已保存；两套正式评估按用户要求暂停，尚无 terminal 正式推理结果。[评估计划](decisions/pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json)

### Gemini 当前定位

原 Gemini 335 已恢复，连续 12 张不同新 RGB-D 的静止杯口定位全部 found，跨度 35.0637 秒，最大画面年龄 0.442676 秒。报告及完整 NPZ 位于 `R/audits/pi05-cup-localization-diagnostic-20261116-v3.json` 和同名目录。J1 在线静止，PGI‑140 未激活且反馈无效；本轮机械臂与夹爪命令均为 0。[只读现场](results/pi05-cup-current-readonly-scene-20261116-v1.json)

### 杯沿 CPU 对照与 v16 观察适配

当前观察源为 `R/src/fr5-pi05-cup-rim-refit-fallback-20261117-v16/`：复用 `cup_rim_geometry._ellipse` 在 Hough 与已有几何种子上拟合补充候选，仅在原候选不能形成有效观测时使用，优先保留原有效候选，保留原几何与深度门槛。shared helper 随完整 bundle 导出；旧权重、正式输入、标签和训练划分保持原样。

原八条全部 4,209 帧分别重放旧观察器与新观察器，未调用模型或设备：

| CPU 对照 | observed 旧→新 | 恢复／丢失 | 共同检出最大坐标差 | 状态 |
| --- | --- | --- | --- | --- |
| v15 拟合候选 | 2,961→3,844 | 884／1，丢 source1-frame253 | 来源 3，约 53.58 mm | 未放行对照 |
| v16 原有效候选优先、缺失时补充 | 2,961→3,712 | 751／0 | 来源 6，16.222308 mm | exit0，`cpu_observer_assessed_not_released` |

v16 各来源观察处理时间中位数约 5.93–6.17 ms。原有效观测不再丢失，但共同检出坐标仍有差异，检出数量不能证明毫米精度或任务时机已修好。[v15 回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-original-eight-replay-20261117-v1.json) · [v16 回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2.json)

两帧 readiness 按报告实际布尔值记录：

| 原帧 | v16 观测 | `close_branch_ready_before→after` |
| --- | --- | --- |
| source3-frame170 | 仍 missing，`rim_minus_jaw_mm=null` | false→false |
| source5-frame183 | rim 恢复，48 个 rim edge samples、30 个 raised depth samples | false→true |

v15 细查中，source5-frame183 的 fresh rim 有 30 个 raised samples／6 个 depth sectors；source3-frame170 仅 3 个 depth sectors，未满足原 4 扇区门槛。v16 readiness 用原 pose／history 与新 RGB-D 观测重算，source3 的缺失仍未恢复。[readiness 报告](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json)

CPU 验证记录含 v15 初版视觉测试 14 项（tracking 7、localization 7）、v16 候选优先调整后 tracking 测试 7 项、完整导出测试 1 项／6 子案例及 selected-fit 测试 26 项，均 exit0。[CPU 验证记录](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json)

新 tracker 下模型动作尚未推理；GPU 保持暂停，控制器和实机均未验证。完整实机抓放成功仍为 0。

### 模型输入与候选筛选依赖

当前 fixed 适配使用 RGB、任务、八维杯相对 TCP／夹爪状态，以及 25 个阶段／视觉／接触／时间条件。杯沿 XYZ 和量测有效性同时进入文本与每步去噪的动作专家输入。有效性为 0 表示没有可靠量测；物理场景仍可有杯子。每帧漏检不改变以开场杯位固定的 RGB 裁剪或杯坐标原点。

CPU 对照重新播放原 capture 与 v16 已保存的前后 tracking，保持同一图像、状态、历史和原模型回复：

| 原帧 | 模型图像／状态 | 条件变化 | 同一原首抽的筛选变化 |
| --- | --- | --- | --- |
| source3-frame170 | 相同 | 0；仍 missing | 六种子均原样选中首抽，包含一条已知迟闭。 |
| source5-frame183 | RGB 裁剪逐元素相同；状态、历史与原点相同 | 六项，50 点共 300 个条件元素 | 原六条首抽均直接入选；恢复观测后两条直接入选、四条要求后抽，含三条已知迟闭。 |

source5 同一原漏检输入的前三点既有 `[100,100,100]%` 也有 `[76,76,76]%`。模型能够在漏检时闭爪；漏检不能充分解释失败。恢复观测后要求后抽的第四条原回复在第三点才闭爪，其时序类别保持原评分标准。

输入诊断与三项 CPU 回归测试完成，未载入模型权重、真实推理调用和设备命令均为 0。网络对更新观测的响应及后续候选仍未评，GPU 暂停。[输入与筛选对照](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-input-dependence-20261117-v1.json) · [诊断入口](/home/dianrobot/demo1/fr5_lerobot/diagnostics/audit_cup_input_dependence.py)

### trial02 预测、提交与反馈时间

CPU 审计以 request id 配对源观测和完整模型回复，以 sequence／delivery token 配对提交与 ACK，以真实关节目标和发送／ACK 时间定位 ServoJ 端点。36 个预测块中 35 个完整执行、一块执行前拒绝；105 次提交／ACK 和 105 个 ServoJ 端点配对。未发生 MoveGripper，该轮用户复核杯子全程在桌上、未被推移。

| 原日志量 | 结果 |
| --- | --- |
| chunk15 首次负向终点，相对相机状态 | −0.284052 mm |
| 同终点，相对块首发送前状态 | −0.180192 mm；反馈年龄 0.039621 秒 |
| 同终点，随后反馈相对相机状态的进展 | −0.316820 mm |
| chunk30 较大负向终点，相对相机状态的预测／随后反馈 | −3.757291／−3.517618 mm；端点执行误差 0.259368 mm |
| 块终点在块首次发送前正向变负向 | 0/35 |
| 逐点按自己的发送前状态正向变负向 | 3/105，分别 −0.011122／−0.171456／−0.101060 mm |
| 反馈端点误差中位／最大 | 0.166161／0.603244 mm |
| 服务端推理中位 | 0.358660 秒 |
| 客户端整个预测调用中位 | 0.564150 秒 |
| 回复至首提交中位 | 0.089947 秒 |
| 源相机至首提交中位 | 0.759749 秒 |
| 三点首提交至末 ACK 中位 | 0.534938 秒 |
| 块间末 ACK 至下一首提交中位 | 0.825853 秒 |
| 含等待的 ACK 端点吞吐 | 2.128066 Hz |

首次负向目标在模型回复中已存在，发送前状态重算仍为负向；它没有由等待期间的状态变化产生。亚毫米负向只表示相对固定示范参考的数值方向，不能当成已经证实的大幅实机走偏。逐点的三次符号变化均小于 1 mm，后两项含块内已执行前序动作的影响，不能全部归为推理延迟。

该审计只使用严格早于发送时间的已存反馈，不插入未来反馈。105 次提交只有 35 个点有末 ACK 后、下一提交前的相机配对反馈；其余点的末端反馈误差保留未评。ACK 表示端点 RPC 返回，真实稳定时间仍需物理反馈。各项中位数分别统计，不相加重建单块时间。名义 6 Hz 的原前缀评分仍未包含上述等待。[原日志时序审计](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-trial02-action-time-audit-20261117-v1.json) · [审计入口](/home/dianrobot/demo1/fr5_lerobot/diagnostics/audit_cup_trial_action_time.py)

时序审计 13 项 CPU 回归通过；输入依赖三项通过，本轮合计 16 项。模型推理和设备命令均为 0。

## 评估入口与证据

`prepare_cup_precontact_evaluation.py` 准备来源 7、训练拟合和失败实机试验的输入；`prepare_cup_terminal_evaluation.py` 统一 fixed phase 的共享输入与原始反馈证据。当前 [输入准备结果](results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json)对应服务器 `audits/pi05-cup-terminal-evaluation-inputs-20261116-v9/`。

`probe_cup_terminal_gradients.py`、`probe_cup_release_phase.py`、`probe_cup_denoising_steps.py` 分别验证 terminal 梯度、release phase 配对输入和采样步数。`evaluate_cup_training_timing_fit.py` 提供实际第一抽／selected fit 生成 helper，`evaluate_cup_precontact_checkpoint.py` 的 fitloop 复用该 helper；`score_cup_training_timing_fit.py` 分别评分两份 FIT_SCHEMA。`score_cup_training_support.py`、`score_cup_release_phase_probe.py`、`score_cup_terminal_gradients.py`、`score_cup_checkpoint_suite.py` 与其直接依赖执行最终评分。输入准备、推理、评分均读取已封存证据；失败试验用于评估，成功教师来源仍为原来的八条。精确工具和证据清单见 [PI05_FILES.md](/home/dianrobot/demo1/fr5_lerobot/PI05_FILES.md)。

原始 RGB-D 和动作对齐结论见 [RGB-D 记录](results/raw-rgbd-summary.md)、[动作对齐](results/action-alignment-summary.md)。实际试验证据见 [trial01](results/pi05-cup-boundary-trial01-assessment-20261115.json)、[trial02 执行](results/pi05-cup-hover-speed-roll64-trial02-execution-audit-20261115.json)和 [v8 trial02 视觉进展](results/pi05-cup-input-coverage256-trial02-visual-progress-summary-20261116-v8.json)。

## 已知问题

- 来源 7 的杯位接近训练杯位，且多次参与模型选择；尚无独立新场景泛化结果。
- RGB 与原始深度的严格同像素对应未验证；杯沿可见、夹爪接触反馈和杯底稳定支撑是不同证据。
- 动作标签由实测 MoveL 反馈插值和夹爪命令派生，执行使用 ServoJ。
- `released` 表示先前夹持后由有效反馈确认开爪；整轮抓起并放稳仍需用户确认。
- v8 和 clean v9 的静止漂移、来源 7 无效反馈覆盖问题尚未解决；terminal 方法尚无完成训练后的对照评估。
- 当前 v11 已完成 256 步、exit0，step128／256 已保存；终点权重尚未推理评估，GPU 工作暂停。原 fit 首抽专项已评分，实际选中候选 fit 专项仍待评。
- v16 全帧 CPU 复核完成但未放行，旧／新共同检出坐标最大差 16.222308 mm，毫米精度未证实；source3-frame170 仍 missing。source5 的 readiness true 只证明闭爪筛选分支条件可达，新 tracker 下模型闭爪、悬停及放稳后开爪时机尚未推理或实机验证。v15 未放行对照保留。
- 原 fit 支撑证据由有效 PGI 接触、桌面法向 TCP 高度、连续稳定反馈和 Gemini 高度／遮挡支撑龄期构成，不能直接证明杯底放稳。首抽专项按名义 6 Hz 评分，尚未结合控制器实测时序；原标签与名义命令在前十点有 11 处差异。
- 留出杯位距最近训练约 0.22 mm，现场杯位约 112.04 mm；唯一留出原指令为 4 秒，真实五秒仅训练来源五次循环。
- 6 Hz 标签与真实执行等待仍不同；trial02 的首次负向终点已在发送前状态上重算，实际等待对闭／开爪与悬停的影响尚未验证。连续 12 新帧只验证静止杯口定位，未验证毫米精度、运动遮挡跟踪或抓放成功。
