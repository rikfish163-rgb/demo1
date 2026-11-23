# FR5 机器人实验工程

| 工程 | 入口 | 运行位置 |
| --- | --- | --- |
| 戴屹视觉打磨 | [主工程](packages/daiyi_grinding/README.md)、[双机说明](packages/daiyi_grinding/HANDOFF.md)、[结果与后续方案](packages/daiyi_grinding/docs/status.md) | 本机感知、规划和 MuJoCo；4090 执行 GPU 推理与训练 |
| FR5 抓杯 | [astra + Jev](fr5_jev/README.md)、[pi0.5](fr5_lerobot/README.md) | 本机连接 J1 和 Gemini 335；4090 保存数据与模型 |

戴屹打磨是独立 Python 子工程，源码、依赖和运行目录位于 `packages/daiyi_grinding/`。工程结构见其[架构说明](packages/daiyi_grinding/docs/architecture.md)。

## FR5 抓杯

仓库维护 astra + Jev 阶段协作和 pi0.5 微调抓取两条主线，共用 J1、Gemini 335、采集记录与执行接口。

| 主线 | 代码与使用说明 | 结果与决策 |
| --- | --- | --- |
| astra + Jev | [fr5_jev](fr5_jev/README.md)：当前观测、阶段选择、监督抓放与连续录制 | [结果](fr5_jev/results/README.md)、[决策与案例](fr5_jev/decisions/README.md) |
| pi0.5 | [fr5_lerobot](fr5_lerobot/README.md)：验收示范导出、微调、推理和连续执行 | [当前状态](fr5_lerobot/CURRENT_PI05.md)、[产物索引](fr5_lerobot/PI05_FILES.md)、[运行说明](fr5_lerobot/PI05_CUP_RUNBOOK.md) |

`fr5_runtime/` 集中保存预览、采集网页、策略工作台、状态读取和运动学公共代码；[hardware/](hardware/README.md) 保存 SDK 状态结构、模型、工具参考和有效标定。

## 运行

在仓库目录执行：

```bash
./run-monitor.sh
./run-collection-console.sh
/home/dianrobot/anaconda3/envs/lerobot/bin/python -m fr5_runtime.policy_console
```

相机预览为 [127.0.0.1:8765](http://127.0.0.1:8765/)，采集网页为 [127.0.0.1:8768](http://127.0.0.1:8768/)，策略工作台为 [127.0.0.1:8769](http://127.0.0.1:8769/)。启动这些服务不下发运动；采集和策略执行由对应操作启动。`./fr5-system --help` 提供采集、审核、源码发布、连续示范导出、微调及策略入口。

本机执行环境为 `/home/dianrobot/anaconda3/envs/lerobot/`。相机预览环境为 `/home/dianrobot/.local/share/fr5/monitor-venv/`，依赖见 `requirements-lock.txt`；运行配置与录制缓冲位于 `~/.local/state/`。

## 数据与模型

服务器为 `4090`，大文件根目录为 `/home/data4/hts/demo1/fr5-learning/`。八条用户验收的教师示范及其观测、动作、反馈保存在 `recording-spool/`；当前连续训练数据为 `datasets/fr5-joint-stream-20261102-v1/`。pi0.5 基础权重位于 `models/pi05_base/`，候选位于 `checkpoints/`，完整来源清单见 [产物索引](fr5_lerobot/PI05_FILES.md)。

astra + Jev 的主要原始观测和结果位于服务器 `results/astra-jev/`；仓库保存摘要、用户确认与代表性图像。早期注册表有两次完整逻辑任务确认，现行 pi0.5 数据中的八条成功记录属于教师示范。

## 验证

```bash
/home/dianrobot/anaconda3/envs/lerobot/bin/python -B -m unittest -q \
  fr5_jev.test_teacher_varied fr5_jev.test_policy fr5_jev.test_decision \
  fr5_runtime.tests.test_console fr5_runtime.tests.test_policy_console \
  fr5_runtime.tests.test_system_training \
  fr5_lerobot.tests.test_stream_export fr5_lerobot.tests.test_stream_training \
  fr5_lerobot.tests.test_stream_runtime
```

## 已知问题

pi0.5 完整实机抓起、保持和放回成功尚未确认，当前候选均未放行。现有标定验证范围、早期录制缺口和模型评估结果分别集中在硬件资料及两条主线的结果说明中。
