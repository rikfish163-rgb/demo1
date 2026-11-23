# FR5/Jev 多物体数据与独立实机评估

当前已有 **8 条同一杯子的用户确认成功教师示范**；两组 π0.5 训练均已完成 5,000 步。新物体的用户确认成功示范数和 π0.5 策略完整实机抓放成功数目前均为 **0**。旧来源 7 已用于模型比较，不是独立测试。

## 任务与分配

每轮任务文本沿用采集器格式：“抓取桌面上的{真实物体描述}，在空中维持5秒后放下”。`planned_routes` 按本轮观测的物体初始支撑位规划返回；验收要求物体原位放稳并松开。模型接收真实相机图像、六轴关节反馈、夹爪位置及有效位和任务文本；历史模型还接收检查点规定的过去实测状态及其观测年龄。

采集前按**物体实体**登记训练、选模验证和最终测试三组，整条物理来源只属于一组。同一物体的不同帧、切片或重复不能伪装成新物体。最终测试预留不同于训练和选模的物体实体与摆位，不用其数据选择模型、调整参数或计算归一化统计量。每轮记录物体 ID、真实描述、形状、可核实尺寸与质量、初始摆位和姿态、任务文本及对应来源；无法测得的项目标为未知。

## 分步计划

1. **新物体 pilot：计划 12 次教师整轮尝试。** 实际选取 2 个不同物体，各设 3 个可达摆位、每个摆位尝试 2 次。当前紫块和灰色圆形物体仅作为待核实候选；确有其物、几何及夹持条件合适时才纳入。每次一轮，目标是建立从观测、采集到用户复核的完整记录。
2. **扩展：计划 48 次教师整轮尝试。** 另选 4 个训练物体，各设 3 个摆位、2 个可稳定且可区分的姿态，每种组合尝试 2 次。按实际可取得物体覆盖刚性方块、圆柱和带抓取部位的日用品，记录实际尺寸和姿态，不预设现场库存。另预留 1 个不同物体，计划以 3 个摆位各尝试 2 次作为**选模验证**，不计为最终独立测试。
3. **冻结后独立测试：计划 18 次策略整轮尝试。** 冻结检查点、归一化统计、输入处理和执行配置后，使用另外 3 个未参与训练或选模的物体，各按预先登记的 3 个未用于训练或选模的摆位尝试 2 次。事先固定顺序及复位方式；每次都计入结果，包括失败、中断和未能判定，不因失败而替换或反复试到成功。按物体和摆位列出成功、失败、中断与未知数，重复尝试不当作独立物体。

计划共进行 **60 次训练角色教师尝试、6 次选模验证尝试和 18 次独立策略评估尝试**；完成情况以真实记录和用户验收为准。实际物体或条件不足时，记录计划未完成；不拿最终测试物体补训练或选模。

## 采集、追加与验收

教师采集使用 [`fr5_jev.prepare_object_session`](../../fr5_jev/prepare_object_session.py)：指定新 `--directory`、本轮 RGB-D 的 `--observation`、真实 `--object-description`、`--rgb-pixel` 或 `--mask`、现场确认的 `--closing-percent`，以及 `--grasp-reference` 或 `--grasp-offset`；`--split` 分别使用 `training_complete_trial` 或 `held_out_complete_trial`。`--grasp-below-top-mm` 和侧抓的 `--side-retraction-mm` 根据物体及真实支撑设置接触位置，工具偏移保持其物理含义。随后由 [`whole_task_collection run`](../../fr5_jev/whole_task_collection.py) 验证会话；只有显式 `--execute` 才执行采集。准备结果是候选路线，不是抓取成功或完整物体几何证明。

记录保留任务与来源、图像和相机时间、实际关节与夹爪反馈、教师决策、实际下发命令及返回、保持区间和整轮总结。服务器录制整理为 `recording-spool/<轮次>/materialized/` 后，由用户依据现场表现或完整回放确认；只有用户明确确认抓起、保持 5 秒、原位放稳并松开，才记为成功。未完成、失败、未确认和反馈不足的记录保留原结果，不写成成功。用户原话和结论通过现有 `fr5-system review`／[`record_review`](../src/lerobot_robot_fairino/human_review.py) 绑定本轮证据。

新增教师来源使用 [`prepare_jev_pi05_dataset.py`](../training/prepare_jev_pi05_dataset.py) 的可重复 `--train-source` 和 `--validation-source` 参数；先检查来源和角色，再用 `--output` 生成**新的** `train` 与 `with-validation` 数据集，保留原数据。归一化只由训练来源计算；使用 `with-validation` 时必须遵守 provenance 划分，普通加载器不能默认排除验证 episode。最终实机运行另存为 [`policy_evaluation`](../src/lerobot_robot_fairino/stream_recording.py) 证据，保留模型输入输出、实际命令、反馈、图像和用户结论，绝不追加为教师成功示范。

新增数据使用 `with-validation` 从 history/5,000 的 `--initialize` 开始，在新的训练目录保存检查点。训练入口保留父数据的全部物理来源、用户复核和训练／留出角色；追加来源时，由当前训练来源重新计算状态和动作归一化，验证来源不参与统计。同来源初始化继续使用父统计量，`--resume` 恢复原数据、优化器和采样状态。状态离散 token 会随新统计量变化，因此新数据需继续微调后再测试。

下一批真实示范完成并验收后，使用下列配置：

```bash
./fr5-system train-stream \
  --dataset <data4内新数据集的with-validation绝对路径> \
  --run <新的训练目录名> --steps 3000 --gpu <空闲高编号GPU> \
  --initialize /home/data4/hts/demo1/fr5-learning/checkpoints/pi05-fr5-jev-history-balanced-base-20261121-v2/step-005000 \
  --rank 16 --adaptation full-expert-vision-language-lora \
  --action-representation joint-delta-from-anchor-v1 --n-action-steps 3 \
  --image-profile full-and-workspace-v1 \
  --observation-context-profile state-history-physical-v1 \
  --sampling-profile balanced-v2 --batch-size 1 --gradient-accumulation 4 \
  --learning-rate 0.00001 --decay-learning-rate 0.0000025 --lr-schedule cosine \
  --first-action-weight 1 --executed-prefix-weight 1 --gripper-weight 1 \
  --save-every 500 --seed 20261122
```

3,000 步是新增数据轮次的计划上限，每 500 步保留候选供选模验证和实机测试。冻结后再运行预留的最终测试物体。

## 当前执行条件

FR5 六轴与夹爪能表达本任务的 7 维动作，但适用对象仍受约 80 mm 夹爪开口、可达空间、实际载荷、形状与夹持部位约束。可见深度不代表隐藏体积或完整净空；夹爪百分比不是已验证的线性毫米尺寸，现有法兰夹持偏移也不是已标定 TCP。每次按真实物体、设备状态和本轮场景核对可夹持性与路线；故障或持物异常时停止并保留记录，由现场人员处理。

