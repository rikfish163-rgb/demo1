# FR5 / 夹爪状态与恢复评估

## 结论

用户已确认当前没有夹持物，并授权继续夹爪恢复和策略实测。2026-11-22 19:39（北京时间），J1 各执行一次夹爪激活与全开，命令返回均为 0；最终状态包为 `active=1`、`position=100`、`motiondone=1`，夹爪故障字段为 0，按项目规则反馈有效。该次设备准备未发送机械臂运动命令。

设备准备不构成策略任务成功；完整 policy 抓放成功数仍为 0。

## 恢复记录

[本机准备日志](/home/dianrobot/.local/state/fr5-system/jev-pi05-empty-gripper-setup-20261122T193946.json)记录了用户的空爪确认、实际命令及反馈：

| 阶段 | 命令或反馈 | 结果 |
| --- | --- | --- |
| 初始状态 | 19:39:46 状态包 | `active=0`、`position=0`、`motiondone=0`，反馈无效。 |
| 激活 | `ActGripper(1, 1)` | 返回 0；随后记录到 `active=1`、`position=0`、`motiondone=0`，反馈仍无效。 |
| 全开 | `MoveGripper(1, 100, 10, 20, 10000, 1, 0, 0.0, 0, 0)` | 返回 0；19:39:48 最终状态包为 `active=1`、`position=100`、`motiondone=1`、`fault=0`、`fault_id=0`、`speed=0`，反馈有效。 |

日志标记为 `experiment_setup_only`，`arm_motion_commands=0`。开爪反馈中曾出现 `motiondone=1` 而 `position=0`，最终一帧才同时达到 100 和完成状态；应结合完整状态判断。

## 接口与记录

原始证据位于 `4090:/home/data4/hts/demo1/fr5-learning/diagnostics/`：

- `fr5-j1-gripper-readonly-20261121T090621414588Z.json`：详细版本、包字段和外设查询。
- `fr5-j1-gripper-readonly-20261121T110746576557Z.json`：19:07 的状态前后及两项夹爪 RPC，机器人和夹爪控制命令均为零。

11 月 22 日 16:40 路径检查结束时，夹爪状态为 `active=0`、`position=0`、`motiondone=0`，机械臂在自动模式、已使能、静止、无故障；该次场景及设备状态保存在 `observations/jev-purple-side-20261122T163744-v1/observation.json` 和同目录 `path-preflight.json`，同属服务器数据根目录。该次路径检查只发 IK、FK 和软限位查询，运动与夹爪控制命令均为零。J1 WebApp 适配设备表显示 DAHUAN、PGI-140、D1.0、挂载位置 0。

设备为 FR5 V5.0、Web 软件 v3.8.4.2、控制器 V3.8.15-QX，本地 SDK 为 V2.1.4。

11 月 21 日归档中的夹爪状态及控制器接口值：

| 来源 | 记录值 | 含义 |
| --- | --- | --- |
| 20004 状态包 | `gripper_active=0` | 未报告激活。状态包文档将该字段记为 0 未激活、1 激活；无论采用布尔值还是接口文档中的位图解释，零值均无激活位。 |
| 20004 状态包 | `gripper_position=0`、`gripper_motiondone=0` | 位置字段为 0%，完成信号为未完成；结合未激活状态，项目将该位置反馈判为无效。 |
| 20004 状态包 | `gripper_fault=0`、`gripper_fault_id=0` | 状态包中记录的夹爪故障字段均为 0。 |
| 控制器 XML-RPC `GetGripperMotionDone` | 原始 `[0, 1, 0]` | 调用返回码为 0；手册将中间值描述为故障标志，本设备的成功开爪记录也出现过 1，不能单凭该值认定故障。 |

[法奥 3.8.4 系列 Python 外设接口](https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/PythonRobotPeripherals.html)将返回数据描述为 `[fault,status]`，其中 `fault` 的 1 表示有错误，`status` 的 1 表示运动完成。[本地 SDK](../../hardware/sdk/Robot.py#L4861) 原样取控制器返回的第二、三个元素，包装为 `(0, [1, 0])`。[项目现场参考](/home/dianrobot/.codex/skills/fr5-jev-collaboration/references/field-notes.md)曾记录成功开爪时 `[0,1,1]`；该字段在本设备上的含义存在分歧，应结合状态包故障字段、位置和完成反馈判断。

状态包记录为 960 字节，记录中五帧的计数为 26、27、29、31、32，校验均有效；该次记录的包接收时间为 17:03:38.630，RPC 返回时间为 17:03:37.559。各字段在包中的原始字节分别为：`motiondone` 偏移 554 为 `00`、`fault` 偏移 567 为 `0000`、`active` 偏移 569 为 `0000`、`position` 偏移 571 为 `00`。项目[状态结构](../src/lerobot_robot_fairino/state_structs.py#L91)与本地 SDK 对这些字段以及 `fault_id`、速度、电流的偏移和大小一致；[解包代码](../src/lerobot_robot_fairino/state_stream.py#L39)直接取结构字段。[反馈规则](../src/lerobot_robot_fairino/feedback.py#L6)将此组合判为无效。

[本地 SDK](../../hardware/sdk/Robot.py#L12184) 的 `GetGripperActivateStatus()` 实现原文为 `return 0, self.robot_state_pkg.gripper_fault,self.robot_state_pkg.gripper_active`；`GetGripperCurPosition()` 同样从 `robot_state_pkg` 取值。这两个返回的首项 0 由本地代码直接给出，与状态包不是独立的控制器查询。相对地，`GetGripperMotionDone()` 调用 `self.robot.GetGripperMotionDone()`。

## 配置与型号

11 月 21 日 `GetGripperConfig` 的控制器原始返回为 `[0, 0, 3, 0, 0]`。[本地 SDK](../../hardware/sdk/Robot.py#L4765) 对成功返回的第一个、第二个数据元素各加 1，得到 `(0, [1, 4, 0, 0])`。它的注释以及历史版 Python 接口说明将四项写作 `[number, company, device, softversion]`；按此解释，编号为 1、厂商 4 为大寰、设备 0 对应 PGI-140。[历史版 Python 示例](https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/PythonRobotPeripherals.html)却将相同的四项命名为 `[company, device, softversion, bus]`，[Java 接口](https://fairino-doc-en.readthedocs.io/3.8.4/SDKManual/JavaRobotPeripherals.html)也采用后一组名称。因此，返回值与项目所称 PGI-140 存在相容的解释，但不足以单凭它确认当前实际型号或协议。

同一记录中 `GetAxleLuaEnableDeviceType` 返回 `[0, 0, 0, 0]`，`GetAxleLuaEnableDevice` 返回成功及全零设备列表；这反映的是末端 Lua 设备配置，不能据此断定已适配设备不存在。`GetAxleLuaEnableStatus` 返回 `[14, 0]`，`GetAxleCommunicationParam` 返回 `[14, 0, 0, 0, 0, 0, 0, 0]`；[错误码表](https://fairino-doc-en.readthedocs.io/3.8.4/SDKManual/errcode.html)将 14 列为接口执行失败，因此后续的零不能作为成功读取的启用或通信参数。

原始证据归档引用的教师记录中，编号 1 的夹爪命令曾对应 `gripper_active=1`、`gripper_motiondone=2`、`gripper_fault=0`；八条来源的初始采样均有 `active=1`、`position=100`、`motiondone=1`。这些是历史记录，不代表当前状态，也不能消除不同接口的故障值差异。

## 项目启动条件

[teacher_targets.py:257](../../fr5_jev/teacher_targets.py#L257) 的 `TeacherSession.start()` 检查夹爪有效反馈、全开及运动完成，并将 `GetGripperConfig` 与原始数组 `[0, 1, 3, 0, 0]` 比较；其 [FR5RPC 传输](../src/lerobot_robot_fairino/control_transport.py#L123)直接返回控制器结果。[项目现场参考](/home/dianrobot/.codex/skills/fr5-jev-collaboration/references/field-notes.md)记录第二个值随激活由 0 变为 1。恢复日志的最终反馈符合上述夹爪状态检查，日志未记录恢复后的原始配置；启动时仍须以实时读取的配置和其余启动条件为准。

## 后续采集与实测

继续前按当时的设备状态和场景取得新鲜 RGB-D，核对物体、路径和夹爪反馈。教师采集与策略实测分别按[采集入口](../docs/jev-pi05-first-object-trial.md)和[独立实机评估](../docs/jev-pi05-data-and-physical-evaluation.md)记录实际观测、动作和反馈；完整抓起、保持 5 秒、放回与释放仍须用户确认后才记为成功。

## 已知不确定项

- 恢复日志最后一帧达到有效全开反馈，未记录其后的持续状态或开爪实景；此前未激活的原因尚未确定。
- 历史版[状态包文档](https://fairino-doc-zhs.readthedocs.io/3.8.4/SDKManual/PythonRobotStatusFeedback.html)将 `gripper_active` 描述为 0/1，`GetGripperActivateStatus` 文档则描述为 bit0–bit15 的位图。实际已适配设备的编号到位映射尚未确定；已有编号 1 的历史命令对应包值 1。
- 历史版手册页面标示 3.8.4.1，不能视为与现场 v3.8.4.2 完全一致。
