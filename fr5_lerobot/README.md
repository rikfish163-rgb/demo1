# FR5 / Gemini π0.5 stream

本包提供 FR5、Gemini 相机的记录适配、用户验收、stream 数据导出、π0.5 微调和连续策略执行。当前使用 8 条用户确认成功的教师示范；π0.5 完整实机抓起并放回成功数为 **0**。

[当前产物](CURRENT_PI05.md) · [π0.5 原理](docs/pi05-fr5-principles.md) · [FR5/Jev 实验](docs/jev-pi05-experiment.md) · [代码、证据与服务器路径](PI05_FILES.md) · [历史抓杯记录](PI05_CUP_RUNBOOK.md)

## 环境与入口

本机采集环境为 LeRobot 0.4.4，可在工程根目录安装 `pip install -e ./fr5_lerobot`。服务器训练和模型推理使用 `/home/data4/hts/demo1/fr5-learning/.venv/bin/python`，对应 LeRobot 0.6.1、PyTorch 2.11.0+cu128、Transformers 5.5.4、PEFT 0.21.2。

公共入口为 `./fr5-system`，公共运行代码在 `fr5_runtime/`。本包的主入口如下：

| 工作 | 入口 |
| --- | --- |
| 教师采集、结果确认 | `./fr5-system collect`、`review`；`./run-collection-console.sh` |
| 用户已确认源导出 | `./fr5-system export-stream`；`lerobot_robot_fairino.export_stream_policy` |
| 真实 Jev 来源审计与训练划分 | `training/prepare_jev_pi05_dataset.py` |
| 微调 | `./fr5-system train-stream`；`training/train_stream_pi05.py` |
| 原生动作与基础权重对照 | `training/evaluate_jev_pi05.py` |
| 服务器模型推理 | `training/serve_stream_pi05.py` |
| 连续实机执行 | `./fr5-system run`；`python -m fr5_runtime.policy_console` |
| 观测记录、记录检查 | `fr5-record-observations`、`fr5-check-dataset` |

策略运行的节拍和动作参数来自完整 checkpoint。

## 教师采集

采集工作台位于 `http://127.0.0.1:8768/`。用户摆好杯子、离开动作区域后点击开始；页面默认从本轮 RGB-D 重新定位杯口，未找到或有多个候选时可手动修正。起点可选交替、左侧、右侧或当前机械臂位置，重复次数可选一轮内抓放一次或两次。

当前对话 LLM 确定杯位、抓取参考和路线，Jev 根据真实反馈选择阶段。默认流程为接近、闭爪、抬起约 50 mm、按任务悬停、原位放回、松爪、沿工具方向退开约 80 mm。教师 MoveL 速度参数默认为 10%，夹爪速度 20%、力 10%；这些数值是控制器比例。5 秒表示任务悬停时间，请求时长和实际时长分别记录。

连续记录实际 RGB-D、实测关节与夹爪、下发目标、RPC 返回、完成反馈、决策来源和时间。原始流在服务器确认保存后释放控制，录像物化可继续进行。记录整理完成后，用户依据整轮现场表现或回放确认成功或失败；`pending_review`、失败和中断记录不进入成功示范。停止按钮只停止当前任务。

每轮原始记录在服务器 `recording-spool/<episode>/`，验收和导出使用其中的 `materialized/`。八条已确认源、训练与留出划分见 [PI05_FILES.md](PI05_FILES.md)。

## 数据与策略

导出器以整条物理示范划分来源，校验用户确认、原始观测、命令和反馈，生成 6 Hz 训练锚点及来源映射。当前训练来源为 0–6，来源 7 留出，分别为 3443 和 750 个锚点。

模型输入为 Gemini RGB、8 维状态、整轮任务指令及 checkpoint 指定的观测历史。深度用于物体定位与几何计算。模型输出完整 `50×7` 动作，执行前缀由检查点配置指定；关节轨迹使用 8 ms ServoJ 节拍，夹爪使用原生整数百分比命令。

2026-11-21 的两组微调已各完成 5,000 个优化步，2,000、4,000、5,000 步完整检查点及底座对照评估均已完成。模型使用当前观测或因果状态历史、全图和工作区裁剪、锚点相对关节动作及绝对夹爪目标；采样包含真实切换邻域、运动、悬停和全轨迹。配置在重新观测前消费原样前十点。[检查点比较](results/jev-pi05-balanced-checkpoint-comparison-20261121-v1.json)保存位置、夹爪和切换时序结果。

普通关节策略运行通过 `fr5-system run --scene-json <本轮实测桌面场景.json>` 接收桌面平面；`--prefix-steps 3` 可在检查点配置的前缀内选择前三点后重新观测，并分别记录检查点配置和实际消费长度。执行前检查整个待消费前缀的关节限位和工具到桌面的净空，保存只读 FK 结果及预检耗时。默认运行只预览，`--execute` 进入实机执行。

新物体通过 `fr5_jev.prepare_object_session` 的俯抓或侧向入口准备，彩色目标可用 `--rgb-pixel`，低饱和度或复合表面可用原始 RGB 网格的 `--mask`。准备和完整采集均使用真实 `fr5_jev` 观测、决策、命令和反馈；整条试验在准备时指定训练或独立留出角色。

[离线评估](training/evaluate_jev_pi05.py)保存未经筛选的完整模型动作和真实标签，[位置与时序分析](training/summarize_jev_pi05.py)将关节误差换算为夹持点毫米误差，并统计夹爪切换时间。当前配方和评分定义见[实验文档](docs/jev-pi05-experiment.md)。

## 已知问题

当前 checkpoint 均未放行。来源 7 已多次参与模型比较，且杯位接近训练来源。RGB 与深度按各自像素模型映射，夹持中心标定与运动时遮挡跟踪仍需实测。动作标签由真实 MoveL 反馈插值与记录的夹爪命令派生。位置拟合、夹爪开度和真实执行时序共同影响任务完成；当前设备、评分与待完成的实机条件见 [CURRENT_PI05.md](CURRENT_PI05.md)。
