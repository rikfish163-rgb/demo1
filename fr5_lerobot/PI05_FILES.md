# FR5 π0.5 文件索引

本机工程：`/home/dianrobot/demo1/`。训练服务器：SSH `4090`，`R=/home/data4/hts/demo1/fr5-learning/`。下文 R 下的路径均属于服务器；本机代码与小证据使用本机绝对路径。

当前 terminal 训练运行 v11 使用 AdamW 暂存源码 v14，已完成 256/256 步、exit0；step128／256 完整包已保存、待评估。GPU 工作按用户要求暂停，本项目无 GPU 作业。clean v9/step256 训练 exit0，主套件离线与原 fit 首抽专项已评分，未放行；实际选中候选 fit 专项待评。当前观察适配为 v16，原八条全帧 CPU 复核已完成、未放行；v15 保留未放行对照。终点权重及新 tracker 下模型动作尚未推理评估，完整实机成功为 0。当前状态见[CURRENT_PI05](/home/dianrobot/demo1/fr5_lerobot/CURRENT_PI05.md)，历史数字与失败记录见[RUNBOOK](/home/dianrobot/demo1/fr5_lerobot/PI05_CUP_RUNBOOK.md)。

## 本机代码

| 文件 | 用途 |
| --- | --- |
| [train_stream_pi05.py](/home/dianrobot/demo1/fr5_lerobot/training/train_stream_pi05.py) | 训练入口；服务器运行使用发布后的 R/src 源码。 |
| [stream_terminal_loss.py](/home/dianrobot/demo1/fr5_lerobot/training/stream_terminal_loss.py) | 原生完整采样的终点 XYZ 监督及内存计算。 |
| [serve_stream_pi05.py](/home/dianrobot/demo1/fr5_lerobot/training/serve_stream_pi05.py) | 完整 50×7 连续动作推理入口。 |
| [cup_localization.py](/home/dianrobot/demo1/fr5_jev/cup_localization.py) | Gemini 杯口定位，整数扇区覆盖门槛。 |
| [cup_tracking.py](/home/dianrobot/demo1/fr5_jev/cup_tracking.py)、[cup_rim_geometry.py](/home/dianrobot/demo1/fr5_jev/cup_rim_geometry.py) | v16 原有效候选优先、缺失时补充杯沿拟合，共用 ellipse helper。 |
| [stream_checkpoint.py](/home/dianrobot/demo1/fr5_lerobot/training/stream_checkpoint.py) | 完整 bundle 导出 shared 杯沿 helper。 |
| [evaluate_cup_training_timing_fit.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/evaluate_cup_training_timing_fit.py)、[evaluate_cup_precontact_checkpoint.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/evaluate_cup_precontact_checkpoint.py) | 实际部署三候选 fit helper 与主套件 fitloop；保存完整第一抽／selected 及 request。 |
| [fit evaluator CPU 测试](/home/dianrobot/demo1/fr5_lerobot/tests/test_evaluate_cup_training_timing_fit.py) | selected-fit 核验入口；相关 26 项 CPU 测试通过，GPU 生成和首抽配对暂停。 |
| [audit_cup_training_pair.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/audit_cup_training_pair.py) | 两版实际训练配置、抽样、归一化和代表参数的 CPU 比对。 |
| [replay_cup_first_proposal.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/replay_cup_first_proposal.py) | 已保存首抽通过生产候选选择器的 CPU 回放，无新增模型推理。 |
| [audit_cup_input_dependence.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/audit_cup_input_dependence.py)、[CPU 测试](/home/dianrobot/demo1/fr5_lerobot/tests/test_audit_cup_input_dependence.py) | 固定原图像、状态和动作，核对杯沿更新对输入及候选筛选的实际作用；三项测试通过。 |
| [audit_cup_trial_action_time.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/audit_cup_trial_action_time.py)、[CPU 测试](/home/dianrobot/demo1/fr5_lerobot/tests/test_audit_cup_trial_action_time.py) | 原日志源观测、完整回复、提交、ACK 与实际 TCP 的时序重建，发送前状态只取已收到反馈；13 项测试通过。 |
| [diagnose_cup_close_tracking.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/diagnose_cup_close_tracking.py) | 用原 RGB-D、TCP 与反馈核对迟闭帧的观测和筛选条件。 |
| [replay_cup_tracking_pair.py](/home/dianrobot/demo1/fr5_lerobot/diagnostics/replay_cup_tracking_pair.py) | 原八条逐帧对照观察器，保存检出变化、坐标差和处理时间。 |
| [run-monitor.sh](/home/dianrobot/demo1/run-monitor.sh) | 只读相机／机器人预览入口，[实时页面](http://127.0.0.1:8765/)。 |

本机界面配置在 `/home/dianrobot/.local/state/fr5-system/config.json`；训练和评估的实际版本以各计划、回执中的绝对路径为准。

## 本机小证据

计划、执行回执和位置方法审计位于 `fr5_lerobot/decisions/`；AdamW 校验、真实梯度、输入核验和评估结果位于 `fr5_lerobot/results/`。

| 证据 | 记录内容 |
| --- | --- |
| [AdamW 暂存校验](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-adam-offload-validation-20261116-v1.json) | GPU9 四项测试全通过，六步 CUDA 参数／状态逐元素一致；moment 释放／恢复各 8,388,608 bytes。 |
| [terminal v11 计划](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-adam-offload-training-plan-20261116-v11.json)、[执行回执](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-adam-offload-training-execution-20261116-v11.json) | 当前 v14 源码、GPU9、fresh optimizer 的实际启动记录。 |
| [GPU 暂停记录](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-gpu-suspension-20261117-v1.json) | 训练 256/256 自然退出、无本项目 GPU 作业；权重和日志保留，后续 GPU 评估暂停。 |
| [配对训练核验](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-paired-training-audit-20261117-v1.json) | CPU 直接比较 50 项共同配置、256 步实际采样、归一化张量和四类代表参数更新。 |
| [v9 回执](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-paired-training-execution-20261116-v9.json)、[v10 回执](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-checkpointed-training-execution-20261116-v10.json) | clean exit0；两次 terminal exit1，详细失败见 RUNBOOK。 |
| [v13 梯度探针](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-gradient-probe-20261116-v2.json)、[回执](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-gradient-execution-20261116-v2.json) | 原生完整动作一致，四类梯度非零、exit0，optimizer_steps=0。 |
| [v12 容差 assessment](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json)、[原回执](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-gradient-execution-20261116-v1.json) | 原容差内零梯度按公式评分，原 probe exit1 保留。 |
| [正式输入摘要](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json) | 219/219 原 source／frame／time 核验，phase 仅 heldout 2 案／fit 5 案改变，54 条失败 evaluation_only。 |
| [共享评估计划](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json) | 新输入的十二种子套件及六种子真实五秒 fit 设置；terminal128／256 已保存，GPU 评估暂停。 |
| [clean256 离线评分](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json)、[执行回执](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-terminal-xyz-clean256-evaluation-execution-20261116-v9.json) | 主套件命令均 exit0，已评分、未放行；控制器和实机未验证。 |
| [clean256 原 fit 首抽专项](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json) | 原 fit100 的 raw-first 闭／开爪时序评分；实际最多三条完整候选筛选后的 fit 待评。 |
| [clean256 首抽筛选回放](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-first-proposal-replay-20261117-v1.json) | raw600 中 562 次直接选第一抽、38 次需要后续抽样；4 次迟闭仍选中、8 次迟开需后续抽样，最终结果未知。 |
| [原帧输入依赖对照](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-input-dependence-20261117-v1.json) | source5 RGB／状态不变、六项条件改变；同原首抽的筛选四条要求后抽。source3 仍缺观测。详细证据和本次诊断工具位于服务器 `R/audits/pi05-cup-input-dependence-20261117-v1/`。 |
| [trial02 实测动作时序](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-trial02-action-time-audit-20261117-v1.json) | 35 个完整块、105 次提交／ACK／ServoJ 端点配对；首次负向小步在发送前仍为负向。详细证据位于服务器 `R/audits/pi05-cup-trial02-action-time-audit-20261117-v1/`。 |
| [迟闭首抽明细](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-close-first-selection-detail-20261117-v1.json)、[原 RGB 图像目录](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-close-rgbd-20261117-v1) | source3-frame170／source5-frame183，原杯沿字段缺失导致正常闭爪筛选未触发；原 640×360 JPG 保留。 |
| [v15 原八条全帧对照](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-original-eight-replay-20261117-v1.json) | 4,209 帧，observed 2,961→3,844，恢复 884／丢失 1，共同检出最大坐标差约 53.58 mm；未放行对照。 |
| [v16 原八条全帧对照](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2.json) | exit0，4,209 帧，observed 2,961→3,712，恢复 751／丢失 0；最大坐标差 16.222308 mm，各来源处理时间中位数约 5.93–6.17 ms；未放行。 |
| [杯沿 CPU 验证记录](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json) | `cpu_observer_assessed_not_released`；初版视觉 14 项、优先调整后 tracking 7 项、导出 6 子案例、selected-fit 26 项均通过。 |
| [v16 闭爪 readiness](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json) | source3-frame170 missing、false→false；source5-frame183 rim 恢复、false→true。新观测下模型动作未推理。 |
| [Gemini／J1 只读现场](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-current-readonly-scene-20261116-v1.json) | 原相机已恢复，机械臂／夹爪命令均为 0；连续 12 新帧定位证据在下方 R/audits。 |
| [实时预览状态](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-live-preview-status-20261117-v1.json) | 2026-11-17 原 Gemini 的 1920×1080 RGB-D 新鲜画面与只读服务状态。 |
| [位置方法审计](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-position-loss-method-audit-20261116-v1.md) | 损失、原生采样、真实梯度及当前内存与训练状态。 |
| [当前原因分析](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-current-cause-analysis-20261117-v1.md) | 已证实缺陷、适配差异、现场配置确认与区分原因的验证。 |
| [公开适配来源](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-public-adaptation-references-20261117-v1.json) | 官方 DROID／LIBERO 条件与社区类似现象的原文摘录和链接。 |

## R/src 实际运行版本

| 服务器源码路径（相对 R） | 对应运行 |
| --- | --- |
| `src/fr5-pi05-cup-rim-refit-fallback-20261117-v16/` | 当前观察适配，原有效候选优先、缺失时补充；CPU 全帧复核 exit0、未放行，旧权重和输入不变。 |
| `src/fr5-pi05-cup-rim-refit-20261117-v15/` | 保留为未放行对照；全帧检出增多伴随 1 帧丢失及共同检出坐标差。 |
| `src/fr5-pi05-cup-terminal-xyz-adam-offload-20261116-v14/` | terminal 训练运行 v11，已完成 256 步；AdamW moment 在反传期间暂存 CPU。 |
| `src/fr5-pi05-cup-terminal-xyz-20261116-v12/` | clean v9 完成训练；terminal v9 的失败源码。 |
| `src/fr5-pi05-cup-terminal-xyz-checkpointed-20261116-v13/` | clean step256 已完成主套件评估；真实梯度探针与 terminal v10 失败源码。 |
| `src/fr5-pi05-cup-input-coverage-20261116-v11/` | 历史 v8 输入覆盖的冻结源码，记录见[RUNBOOK](/home/dianrobot/demo1/fr5_lerobot/PI05_CUP_RUNBOOK.md#已完成对照)。 |

## R/checkpoints 候选与对照

| 路径（相对 R） | 用途与状态 |
| --- | --- |
| `checkpoints/pi05-fr5-cup-terminal-xyz-terminal-n3-20261116-v11/` | 当前候选运行，256 步、exit0；step128／256 完整包已保存，GPU 评估暂停。 |
| `checkpoints/pi05-fr5-cup-terminal-xyz-clean-n3-20261116-v9/step-000256/` | 配对 clean 对照，训练 exit0；主套件离线与原 fit 首抽专项已评分，未放行。 |
| `checkpoints/pi05-fr5-cup-fixed-clock-n3-20261116-v7/step-000128/` | 两版共同初始化父权重。 |

当前 terminal 日志为 `R/logs/pi05-fr5-cup-terminal-xyz-terminal-n3-20261116-v11.log`。terminal v9／v10 的 OOM 运行和旧未放行权重均保留，完整路径及结果集中在[训练与失败记录](/home/dianrobot/demo1/fr5_lerobot/PI05_CUP_RUNBOOK.md#terminal-xyz-与采样步数)。

## 原八条数据与失败回放

| 路径（相对 R） | 归属 |
| --- | --- |
| `datasets/fr5-joint-stream-20261102-v1/` | 原确认 8 条完整导出；来源 0–6 的 3,443 个训练锚点、来源 7 的 750 个固定留出锚点。 |
| `recording-spool/` | 原始 RGB-D、时间戳、FR5／PGI 反馈和命令；八条来源由数据集元数据列出。 |
| `recording-spool/pi05-cup-tcp-tick240-trial-20261113-01/materialized/` | 旧失败回放，仅评估。 |
| `recording-spool/pi05-cup-boundary-trial01-20261115/materialized/` | 失败实机完整归档、用户复核，training_ready=false，仅评估。 |
| `recording-spool/pi05-cup-hover-speed-roll64-trial02-20261115/materialized/` | 失败实机完整归档、用户复核，training_ready=false，仅评估。 |

训练不增加示范，失败回放不进入训练。原 fit100 的 raw-first 闭／开爪专项已评分；实际最多三条完整候选筛选后的 fit 专项仍 pending。

## R/audits 与正式评估输出

| 路径（相对 R） | 内容 |
| --- | --- |
| `audits/pi05-cup-terminal-evaluation-inputs-20261116-v9/` | 六套正式输入、phase-evidence.jsonl、summary.json、validation.json。 |
| `audits/pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json` | 服务器评估计划，完整输入与命令配置。 |
| `evaluations/pi05-cup-terminal-xyz-clean256-suite-20261116-v9/` | 主套件与原 fit 首抽专项已评分；实际选中候选 fit 专项待评。 |
| `evaluations/pi05-cup-terminal-xyz-terminal128-suite-20261116-v11/` | 已保存 step128 的中间评估计划，GPU 评估暂停。 |
| `evaluations/pi05-cup-terminal-xyz-terminal256-suite-20261116-v11/` | 已保存 step256 的最终评估计划，GPU 评估暂停。 |
| `diagnostics/pi05-cup-cpu-audit-20261117-v1/audit_cup_training_pair.py` | 两版实际训练、归一化和代表参数的 CPU 核验入口。 |
| `diagnostics/pi05-cup-cpu-audit-20261117-v1/replay_cup_first_proposal.py` | 已保存首抽的生产选择器 CPU 回放入口。 |
| `diagnostics/pi05-cup-cpu-audit-20261117-v1/diagnose_cup_close_tracking.py` | 原 RGB-D 迟闭观测与筛选条件诊断入口。 |
| `diagnostics/pi05-cup-cpu-audit-20261117-v1/replay_cup_tracking_pair.py` | 原八条逐帧观察器配对回放入口。 |
| `audits/pi05-cup-terminal-xyz-paired-training-audit-20261117-v1.json` | CPU 配对核验完整结果。 |
| `diagnostics/pi05-cup-terminal-xyz-fit-selection-20261117-v10/` | 实际部署 fit evaluator 工具目录，26 项 CPU 测试通过；GPU 候选生成与 raw600 完整首抽精确配对暂停。 |
| `audits/pi05-cup-rim-refit-original-eight-replay-20261117-v1/` | v15 原八条全帧 CPU 对照；保留未放行记录。 |
| `audits/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2/` | v16 原八条全帧 CPU 对照，exit0；毫米精度、控制器和实机未验证。 |
| `audits/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json` | v16 两个原观测的闭爪分支 readiness，实际 false／true 见本机小报告。 |
| `audits/pi05-cup-localization-diagnostic-20261116-v3.json`、同名目录 | 连续 12 张不同新 RGB-D 静止杯口定位报告和完整 NPZ。 |

历史实验数字、控制器与实机失败、约 112 mm 杯位和节奏差异等限制见[抓杯实验记录](/home/dianrobot/demo1/fr5_lerobot/PI05_CUP_RUNBOOK.md)。

## 包内精确清单

以下路径相对于 `fr5_lerobot/`。`whole_task_source`、`curate_teacher` 是 stream 的教师验收依赖；`whole_task_policy`、`task_contract` 提供 stream 共用协议。基础 helper 的动态依赖随完整 checkpoint 封存。

### 相机

```text
src/lerobot_camera_gemini/__init__.py
src/lerobot_camera_gemini/camera_gemini.py
src/lerobot_camera_gemini/configuration_gemini.py
src/lerobot_camera_gemini/timing.py
```

### 数据、推理与执行

```text
src/lerobot_robot_fairino/__init__.py
src/lerobot_robot_fairino/action_control.py
src/lerobot_robot_fairino/configuration_control.py
src/lerobot_robot_fairino/configuration_fairino.py
src/lerobot_robot_fairino/continuous_capture.py
src/lerobot_robot_fairino/control_transport.py
src/lerobot_robot_fairino/cup_action_selection.py
src/lerobot_robot_fairino/cup_relative_runtime.py
src/lerobot_robot_fairino/curate_teacher.py
src/lerobot_robot_fairino/demo_writer.py
src/lerobot_robot_fairino/export_stream_policy.py
src/lerobot_robot_fairino/feedback.py
src/lerobot_robot_fairino/frame_codec.py
src/lerobot_robot_fairino/human_review.py
src/lerobot_robot_fairino/pi05_bridge.py
src/lerobot_robot_fairino/pi05_rollout.py
src/lerobot_robot_fairino/record.py
src/lerobot_robot_fairino/robot_control.py
src/lerobot_robot_fairino/robot_fairino.py
src/lerobot_robot_fairino/servo_control.py
src/lerobot_robot_fairino/state_stream.py
src/lerobot_robot_fairino/state_structs.py
src/lerobot_robot_fairino/stream_contract.py
src/lerobot_robot_fairino/stream_policy.py
src/lerobot_robot_fairino/stream_recording.py
src/lerobot_robot_fairino/stream_runtime.py
src/lerobot_robot_fairino/task_contract.py
src/lerobot_robot_fairino/verify.py
src/lerobot_robot_fairino/whole_task_policy.py
src/lerobot_robot_fairino/whole_task_source.py
src/lerobot_robot_fairino/web/collection.html
src/lerobot_robot_fairino/web/policy.html
```

### 训练与模型服务

```text
training/audit_stream_servo_tracking.py
training/cup_relative_stream_data.py
training/download_pi05_fr5.py
training/evaluate_stream_approach.py
training/evaluate_stream_conditioning.py
training/evaluate_stream_hold_release.py
training/relative_stream_actions.py
training/serve_stream_pi05.py
training/stream_checkpoint.py
training/stream_conditioned_sampling.py
training/stream_image_profile.py
training/stream_observation_history.py
training/stream_prompt_integrity.py
training/stream_terminal_loss.py
training/train_pi05_fr5.py
training/train_stream_pi05.py
```

### 准备、probe 与评分

```text
diagnostics/evaluate_cup_precontact_checkpoint.py
diagnostics/evaluate_cup_training_timing_fit.py
diagnostics/evaluate_cup_setdown_gate.py
diagnostics/evaluate_cup_timing_policy.py
diagnostics/prepare_cup_precontact_evaluation.py
diagnostics/prepare_cup_terminal_evaluation.py
diagnostics/probe_cup_denoising_steps.py
diagnostics/probe_cup_release_phase.py
diagnostics/probe_cup_terminal_gradients.py
diagnostics/replay_true_five_second_pi05.py
diagnostics/score_cup_boundary_replay.py
diagnostics/score_cup_checkpoint_suite.py
diagnostics/score_cup_feedback_recovery.py
diagnostics/score_cup_proposal_trace.py
diagnostics/score_cup_release_phase_probe.py
diagnostics/score_cup_terminal_gradients.py
diagnostics/score_cup_training_support.py
diagnostics/score_cup_training_timing_fit.py
diagnostics/score_cup_visual_progress.py
```

配置：`config/99-gemini335.rules`。测试共享源夹具：`tests/stream_fixtures.py`。杯几何公共依赖：`fr5_jev/cup_tracking.py`、`fr5_jev/cup_rim_geometry.py`、`fr5_jev/cup_localization.py`、`fr5_jev/cup_timing.py`、`fr5_jev/rgbd_geometry.py`。视觉测试入口为 `fr5_lerobot/tests/test_cup_tracking.py` 与 `fr5_jev/test_cup_localization.py`。

### 结果

- [results/action-alignment-summary.md](results/action-alignment-summary.md)
- [results/cup-timing-released-stage-20261116-v1/changed-frames.jsonl](results/cup-timing-released-stage-20261116-v1/changed-frames.jsonl)
- [results/cup-timing-released-stage-20261116-v1/pgi-corrected-phase-evidence.json](results/cup-timing-released-stage-20261116-v1/pgi-corrected-phase-evidence.json)
- [results/cup-timing-released-stage-20261116-v1/summary.json](results/cup-timing-released-stage-20261116-v1/summary.json)
- [results/cup-timing-released-stage-20261116-v1/summary.md](results/cup-timing-released-stage-20261116-v1/summary.md)
- [results/cup-timing-released-stage-20261116-v1/validation.json](results/cup-timing-released-stage-20261116-v1/validation.json)
- [results/pi05-cup-action-clock-five-second-controller-20261113.json](results/pi05-cup-action-clock-five-second-controller-20261113.json)
- [results/pi05-cup-boundary-trial01-assessment-20261115.json](results/pi05-cup-boundary-trial01-assessment-20261115.json)
- [results/pi05-cup-coverage-v8-denoising-10-20-20261116-v2-summary.json](results/pi05-cup-coverage-v8-denoising-10-20-20261116-v2-summary.json)
- [results/pi05-cup-coverage-v8-denoising-10-20-execution-20261116-v2.json](results/pi05-cup-coverage-v8-denoising-10-20-execution-20261116-v2.json)
- [results/pi05-cup-current-readonly-scene-20261116-v1.jpg](results/pi05-cup-current-readonly-scene-20261116-v1.jpg)
- [results/pi05-cup-current-readonly-scene-20261116-v1.json](results/pi05-cup-current-readonly-scene-20261116-v1.json)
- [results/pi05-cup-fixed-clock128-paired-assessment-20261116-v7.json](results/pi05-cup-fixed-clock128-paired-assessment-20261116-v7.json)
- [results/pi05-cup-hover-speed-roll64-trial02-cadence-20261115.json](results/pi05-cup-hover-speed-roll64-trial02-cadence-20261115.json)
- [results/pi05-cup-hover-speed-roll64-trial02-execution-audit-20261115.json](results/pi05-cup-hover-speed-roll64-trial02-execution-audit-20261115.json)
- [results/pi05-cup-hover-speed-roll64-trial02-feedback-path-20261115.json](results/pi05-cup-hover-speed-roll64-trial02-feedback-path-20261115.json)
- [results/pi05-cup-hover-speed-roll64-trial02-tracking-audit-20261115.json](results/pi05-cup-hover-speed-roll64-trial02-tracking-audit-20261115.json)
- [results/pi05-cup-input-coverage-control256-training-20261116-v8.json](results/pi05-cup-input-coverage-control256-training-20261116-v8.json)
- [results/pi05-cup-input-coverage-coverage256-training-20261116-v8.json](results/pi05-cup-input-coverage-coverage256-training-20261116-v8.json)
- [results/pi05-cup-input-coverage256-paired-assessment-20261116-v8.json](results/pi05-cup-input-coverage256-paired-assessment-20261116-v8.json)
- [results/pi05-cup-input-coverage256-trial02-visual-progress-summary-20261116-v8.json](results/pi05-cup-input-coverage256-trial02-visual-progress-summary-20261116-v8.json)
- [results/pi05-cup-release-phase-paired-score-20261116-v1.json](results/pi05-cup-release-phase-paired-score-20261116-v1.json)
- [results/pi05-cup-release-phase-paired-score-summary-20261116-v1.json](results/pi05-cup-release-phase-paired-score-summary-20261116-v1.json)
- [results/pi05-cup-terminal-adam-offload-validation-20261116-v1.json](results/pi05-cup-terminal-adam-offload-validation-20261116-v1.json)
- [results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json](results/pi05-cup-terminal-evaluation-inputs-20261116-v9-summary.json)
- [results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json](results/pi05-cup-terminal-xyz-clean256-assessment-20261116-v9.json)
- [results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-terminal-xyz-clean256-training-timing-fit-20261116-v9.json)
- [results/pi05-cup-clean256-first-proposal-replay-20261117-v1.json](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-first-proposal-replay-20261117-v1.json)
- [results/pi05-cup-clean256-close-first-selection-detail-20261117-v1.json](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-clean256-close-first-selection-detail-20261117-v1.json)
- [results/pi05-cup-rim-refit-original-eight-replay-20261117-v1.json](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-original-eight-replay-20261117-v1.json)
- [results/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2.json](/home/dianrobot/demo1/fr5_lerobot/results/pi05-cup-rim-refit-fallback-original-eight-replay-20261117-v2.json)
- [results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json](results/pi05-cup-terminal-xyz-gradient-assessment-20261116-v1.json)
- [results/pi05-cup-terminal-xyz-gradient-execution-20261116-v1.json](results/pi05-cup-terminal-xyz-gradient-execution-20261116-v1.json)
- [results/pi05-cup-terminal-xyz-gradient-execution-20261116-v2.json](results/pi05-cup-terminal-xyz-gradient-execution-20261116-v2.json)
- [results/pi05-cup-terminal-xyz-gradient-probe-20261116-v1.json](results/pi05-cup-terminal-xyz-gradient-probe-20261116-v1.json)
- [results/pi05-cup-terminal-xyz-gradient-probe-20261116-v2.json](results/pi05-cup-terminal-xyz-gradient-probe-20261116-v2.json)
- [results/raw-rgbd-summary.md](results/raw-rgbd-summary.md)
- [results/success8-data-review-20261102-v1/action-alignment-report.json](results/success8-data-review-20261102-v1/action-alignment-report.json)
- [results/success8-data-review-20261102-v1/raw-rgbd-report.json](results/success8-data-review-20261102-v1/raw-rgbd-report.json)

### 决定

- [decisions/pi05-cup-fixed-clock-control128-release-20261116-v7.json](decisions/pi05-cup-fixed-clock-control128-release-20261116-v7.json)
- [decisions/pi05-cup-fixed-clock-fixed128-release-20261116-v7.json](decisions/pi05-cup-fixed-clock-fixed128-release-20261116-v7.json)
- [decisions/pi05-cup-fixed-clock-paired-training-invariants-20261116-v7.json](decisions/pi05-cup-fixed-clock-paired-training-invariants-20261116-v7.json)
- [decisions/pi05-cup-input-coverage-control256-release-20261116-v8.json](decisions/pi05-cup-input-coverage-control256-release-20261116-v8.json)
- [decisions/pi05-cup-input-coverage-coverage256-release-20261116-v8.json](decisions/pi05-cup-input-coverage-coverage256-release-20261116-v8.json)
- [decisions/pi05-cup-input-coverage-paired-training-invariants-20261116-v8.json](decisions/pi05-cup-input-coverage-paired-training-invariants-20261116-v8.json)
- [decisions/pi05-cup-input-coverage-paired-training-plan-20261116-v8.json](decisions/pi05-cup-input-coverage-paired-training-plan-20261116-v8.json)
- [decisions/pi05-cup-input-coverage-reachability-20261116-v8.json](decisions/pi05-cup-input-coverage-reachability-20261116-v8.json)
- [decisions/pi05-cup-input-coverage-sampling-20261116-v8.json](decisions/pi05-cup-input-coverage-sampling-20261116-v8.json)
- [decisions/pi05-cup-position-loss-method-audit-20261116-v1.md](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-position-loss-method-audit-20261116-v1.md)
- [decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-cpu-validation-20261117-v1.json)
- [decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json](/home/dianrobot/demo1/fr5_lerobot/decisions/pi05-cup-rim-refit-fallback-close-readiness-20261117-v1.json)
- [decisions/pi05-cup-terminal-xyz-adam-offload-training-execution-20261116-v11.json](decisions/pi05-cup-terminal-xyz-adam-offload-training-execution-20261116-v11.json)
- [decisions/pi05-cup-terminal-xyz-adam-offload-training-plan-20261116-v11.json](decisions/pi05-cup-terminal-xyz-adam-offload-training-plan-20261116-v11.json)
- [decisions/pi05-cup-terminal-xyz-checkpointed-training-execution-20261116-v10.json](decisions/pi05-cup-terminal-xyz-checkpointed-training-execution-20261116-v10.json)
- [decisions/pi05-cup-terminal-xyz-checkpointed-training-plan-20261116-v10.json](decisions/pi05-cup-terminal-xyz-checkpointed-training-plan-20261116-v10.json)
- [decisions/pi05-cup-terminal-xyz-clean256-evaluation-execution-20261116-v9.json](decisions/pi05-cup-terminal-xyz-clean256-evaluation-execution-20261116-v9.json)
- [decisions/pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json](decisions/pi05-cup-terminal-xyz-evaluation-plan-20261116-v9.json)
- [decisions/pi05-cup-terminal-xyz-paired-training-execution-20261116-v9.json](decisions/pi05-cup-terminal-xyz-paired-training-execution-20261116-v9.json)
- [decisions/pi05-cup-terminal-xyz-paired-training-plan-20261116-v9.json](decisions/pi05-cup-terminal-xyz-paired-training-plan-20261116-v9.json)
- [decisions/pi05-cup-terminal-xyz-source-publication-20261116-v12.json](decisions/pi05-cup-terminal-xyz-source-publication-20261116-v12.json)

