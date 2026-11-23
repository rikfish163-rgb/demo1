# FR5：Jev 阶段决策与 LLM 复核

Jev 选择任务阶段，程序负责状态、坐标、IK/FK、动作参数和完成反馈。LLM 提供场景事实、处理低置信度和异常，并整理可复用案例。公共状态与几何代码位于 `fr5_runtime`，采集和执行驱动由 `fr5_lerobot` 提供。

## 入口

在工程目录使用 `lerobot` Python 环境。

| 模块 | 用途 |
|---|---|
| `workflow` | 只读 `status/observe/review/replay`。`observe/review` 加 `--call-jev` 才请求 Jev；回放读取已有决策。 |
| `continuous_workflow` | 按已准备的会话执行完整抓放，阶段间由当前对话 LLM 复核图像与异常。默认仅验证；`--execute` 才启动录制和执行。 |
| `whole_task_collection` | `prepare` 根据当前观测与目标生成路线，`run` 用公共执行驱动完成任务；Jev 每阶段选择一次，LLM 路线计划留档。`run --execute` 才执行。 |
| `prepare_object_session` | 根据当前 RGB-D、可见色块、抓取偏移和闭合目标准备一轮物体抓放；`--approach top-down/side` 选择俯抓或侧向接近，仅写采集 scope。 |

```bash
python -m fr5_jev.workflow status --output ~/.local/state/fr5-system/jev/NEW
python -m fr5_jev.workflow replay --help
```

运动入口使用 `--directory` 指定已经准备、复核且尚未开始的会话。实际观测、下发动作、反馈与阶段决定随会话保存；完整任务结束进入 `pending_review`，由用户验收。

物体准备使用 `--object-description`、`--rgb-pixel`、`--closing-percent` 和实际抓取偏移。俯抓跨可见短边闭合，侧抓沿短边接近、跨另一边闭合；两种路线分别核对真实夹持宽度。`--split held_out_complete_trial` 将整条试验预留为独立测试来源，默认是训练来源。可见表面测量与候选路线不会把物体隐藏体积、支撑关系或路径净空标为已验证。

## 结果与决策

[结果摘要与用户确认](results/)；[Jev 决策与案例](decisions/)。`experiment.py` 提供 `decide/save_new`，默认将新运行记录写入 `~/.local/state/fr5-system/jev`；调用方可显式指定会话输出目录。

仓库保留摘要 JSON、用户确认 registry、少量关键图和明确决策。主要原始观测的服务器归档目录为 `/home/data4/hts/demo1/fr5-learning/results/astra-jev/`。

已记录 **两次用户确认的完整逻辑任务**，两条 registry 均为 `training_ready=false`。第一条由两段原始录制组成，用户接受其为一条逻辑轨迹；第二条用户确认“成功，计入示范”。用户确认任务成功与数据达到训练要求分别记录。

教师和整任务执行的完成状态保持 `pending_review`。离线测试、接口返回、预抓取完成、失败及中断恢复不增加用户确认的成功次数。Jev 案例用于上下文检索；`needs_llm/needs_astra` 只表示路由结果，独立 Astra API 调用按实际记录统计。

## 录制与验证

流式录制由 `async_spool/stream_storage/stream_protocol` 保存，`materialize_stream` 整理和核对观测、动作与收尾记录。每条录制的 `wire/frame_codec.py`、`wire/stream_protocol.py` 是动态加载来源，与数据共同保留。

连续教师的 fresh recording 使用 `start_for_episode` 校验 `~/.local/state/fr5-system/stream-readiness.json`。该文件是每次 fresh recording 验证产生的运行状态；测试日志路径相对其所在的状态目录，且须位于该目录内。`whole_task_collection` 与 policy stream 使用 `start_readonly`。

```bash
python -m unittest -q fr5_jev.test_decision fr5_jev.test_policy fr5_jev.test_teacher_varied fr5_jev.test_continuous_workflow fr5_jev.test_whole_task_collection
```

这些测试使用离线样本和模拟反馈，验证决策、阶段顺序与动作记录，不计为实机试验。

## 已知范围

两次验收覆盖已记录的杯子与工作区；随机杯位泛化、独立抓取 TCP 标定和全路径精度尚未由这些结果证明。手眼外参的样本区域验证与深度尺度、夹持中心及任务验收分别记录。历史数据缺失与训练可用性见结果和案例中的具体记录。
