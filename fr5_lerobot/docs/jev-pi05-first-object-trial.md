# 新物体的一轮教师采集

在现场设备恢复并取得新 RGB-D 后，在 `/home/dianrobot/demo1` 执行。采集使用本机 `lerobot` Python、相机桥接和 FR5；4090 的 `/home/data4/hts/demo1/fr5-learning/.venv/bin/python` 用于服务器上的整理和验收。已发布代码快照为 `src/fr5-jev-pi05-tools-20261122-v10`。当前用户已授权有人现场看护时继续；采集结果仍须由用户单独验收。

## 本轮输入

- `OBS`：本机新观测目录，包含 `observation.json` 和 `scene.npz`；图像采集时间和观测完成时间在准备时均须距当前不超过 5 分钟，机器人在观测期间和启动前保持原位，物体和现场布置也须保持不变。RGB-D 必须来自工程已标定的相机，深度和时间检查须通过。
- `RGB_U`、`RGB_V`：从本轮原始 `color.jpg` 上选取物体内部像素；低饱和度或颜色分割不可靠时，改用本轮原始 RGB 网格的 PNG `--mask`，与 `--rgb-pixel` 二选一。检查生成的 `selection-mask.png` 是否覆盖目标而未混入背景或其他物体。
- `OBJECT_DESC`：真实物体的单行描述。确认物体尺寸、材质、重量、支撑方式及持物时的净空。
- `CLOSE_PCT`：针对本物体和当前 PGI-140 指尖确认的 **目标开度**，整数 0–99；100 为全开，0 为全闭。它会原样成为夹爪闭合命令的整数目标，不是闭合比例，也没有经验证的毫米线性换算。实际接触可能在目标之前发生，不要用旧杯子的数值代替本轮确认。
- `GX GY GZ`：当前夹爪法兰坐标系下的有效夹持中心偏移，单位毫米，须有适用当前指尖和抓取部位的实际依据。本轮通过 `--grasp-offset` 显式提供，代码要求 `|GX|、|GY| ≤ 30`、`120 ≤ GZ ≤ 210`。
- `GRASP_BELOW_TOP_MM`：夹持中心在可见顶部下方的距离，沿实测桌面法向计算，默认 10 mm。`SIDE_RETRACTION_MM`：侧抓中心沿工具轴反向后退距离，默认 0 mm，小于可见短边的一半。两者改变候选接触位置，工具真实偏移保持不变。
- `TRIAL_NAME`：本机和服务器均未使用过的简单文件名；本机目录必须是 `/home/dianrobot/.local/state/fr5-recording-spool/<TRIAL_NAME>`。本例为训练角色；选模验证物体改用 `--split held_out_complete_trial`。

## 准备和复核

将下列占位内容替换为现场已知值，在同一 shell 中执行：

```bash
cd /home/dianrobot/demo1
export PY=/home/dianrobot/anaconda3/envs/lerobot/bin/python
export TRIAL_NAME='填写全新轮次名称'
export SESSION="/home/dianrobot/.local/state/fr5-recording-spool/$TRIAL_NAME"
export OBS='填写本机本轮观测目录的绝对路径'
export OBJECT_DESC='填写本轮物体描述'
export RGB_U='填写原始RGB横坐标'
export RGB_V='填写原始RGB纵坐标'
export CLOSE_PCT='填写经现场确认的整数目标开度'
export GX='填写法兰系X偏移mm'
export GY='填写法兰系Y偏移mm'
export GZ='填写法兰系Z偏移mm'
export GRASP_BELOW_TOP_MM=10
export SIDE_RETRACTION_MM=0

PYTHONDONTWRITEBYTECODE=1 "$PY" -m fr5_jev.prepare_object_session \
  --directory "$SESSION" --observation "$OBS" \
  --object-description "$OBJECT_DESC" --rgb-pixel "$RGB_U" "$RGB_V" \
  --closing-percent "$CLOSE_PCT" --grasp-offset "$GX" "$GY" "$GZ" \
  --grasp-below-top-mm "$GRASP_BELOW_TOP_MM" --side-retraction-mm "$SIDE_RETRACTION_MM" \
  --approach side --split training_complete_trial

PYTHONDONTWRITEBYTECODE=1 "$PY" -m fr5_jev.whole_task_collection \
  run --directory "$SESSION"
```

选择掩膜时，将 `--rgb-pixel "$RGB_U" "$RGB_V"` 换成 `--mask '本轮PNG绝对路径'`。准备成功返回 `prepared_no_motion`；第二条命令返回 `validated_no_connection_no_motion`。它只校验 scope 和动作结构，不检查路径可达性或碰撞。

当前紫块的绿色支撑面在归档观测中约高于桌面 63 mm，原接触路线的外壳余量不足。已准备并进行控制器查询的候选使用 `GRASP_BELOW_TOP_MM=5`、`SIDE_RETRACTION_MM=21`，当前法兰系参考偏移为 `[5.208, 8.072, 162] mm`。新场景保持相同物体与指尖时，按实测结果重新生成；接触较浅，实际夹持及紫绿连接关系由现场确认。[支撑测量](../results/jev-purple-support-geometry-20261122-v1.json)和[1,409 点候选路径预检](../results/jev-purple-support-aware-side-preflight-20261122-v1.json)保留证据。

查看 `$SESSION/scene.jpg`、`$SESSION/selection-mask.png`、`$SESSION/geometry.json` 和 `$SESSION/scope.json`。侧向路线按可见短边方向接近、跨另一边闭合；当前实现按 80 mm 名义开口和可见闭合宽度加 6 mm 的余量筛选。核对实际指尖开口、物体完整体积、接触部位、起点和整条机械臂、工具、持物及线缆扫掠区域。可见表面与名义工具盒均不能证明隐藏部分或现场净空。

使用本轮路线执行只读控制器 IK/FK、关节余量和工具盒到桌面的预检，并将摘要留在新会话中。此命令会连接控制器查询，但不发送运动或夹爪命令；由持有现场设备控制权的主代理执行：

```bash
flock -n /home/dianrobot/.local/state/fr5-system/j1.lock \
  env PYTHONDONTWRITEBYTECODE=1 "$PY" - "$SESSION" <<'PY'
import json
import sys
import time
from pathlib import Path

import numpy as np
from fr5_jev.prepare_varied_session import sample_geometry, table_clearances

root = Path(sys.argv[1])
scope = json.loads((root / 'scope.json').read_text())
geometry = json.loads((root / 'geometry.json').read_text())['geometry_report']
observation = json.loads((root / 'source-observation.json').read_text())
routes = scope['planned_routes'][0]
table = geometry['table']
clearance = table_clearances(routes, np.asarray(table['normal']), table['offset_mm'])
rows, kinematics = sample_geometry(routes, observation['robot_after']['joints'])
report = {
    'checked_unix': time.time(),
    'source_observation': geometry['source_observation'],
    'sample_count': len(rows),
    'kinematics': kinematics,
    'tool_table_clearance_mm': clearance,
    'motion_commands_sent': 0,
    'gripper_commands_sent': 0,
}
with (root / 'path-preflight.json').open('x', encoding='utf-8') as stream:
    json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(report, ensure_ascii=False))
PY
```

任何检查失败都停止，不改阈值或设备检查。该预检不覆盖实际物体、整臂、线缆及环境；由现场人员结合本轮路线完成检查，并核对末端工具与物体的总负载及质心。启动时要求机械臂使能、无故障、tool0/user0、运动队列为空、相机新鲜，夹爪已激活、无故障且空夹；若夹爪未全开，执行器会先记录并下发全开命令。采集期间只能有一个控制进程持有 J1。

## 执行和验收

使用 Jev 对每个阶段的实际选择，不预写 `llm-route-plan.json`。启动前确认观测仍在 5 分钟内且起始 TCP 未改变：

```bash
flock -n /home/dianrobot/.local/state/fr5-system/j1.lock \
  env PYTHONDONTWRITEBYTECODE=1 "$PY" -m fr5_jev.whole_task_collection \
  run --directory "$SESSION" --execute
```

程序先闭合至 `CLOSE_PCT`，要求有效的夹爪接触状态 `motiondone=2`；再沿实测桌面法向从接触位试提 10 mm，继续到距接触位 50 mm，保持至少 5 秒，沿路线返回同一接触 TCP，以 100% 开度释放，再完整撤离。50 mm 是沿桌面法向的目标位移，不是从 10 mm 位置再加 50 mm。实际 `hold-01.json` 的计时从提升完成后开始，到返回指令实际发出为止，网络或决策等待可能使保持时间长于 5 秒。接触反馈、目标位置和计时都不能单独证明物体跟随、放稳或未滑落；现场异常时停止并保留记录。

完整执行后检查本机 `$SESSION/summary.json`、`$SESSION/hold-01.json`、实际命令日志及现场视频；服务器整理目录为 `4090:/home/data4/hts/demo1/fr5-learning/recording-spool/<TRIAL_NAME>/materialized/`。只有阶段、实际命令、保持时间、释放和录制均通过内部完整性检查，结果才进入 `pending_review`。该状态不是成功，且训练就绪仍为 false。

用户明确确认物体被抓起、保持至少 5 秒、原位放稳并松开后，记录用户实际原话：

```bash
./fr5-system review --episode "$TRIAL_NAME" --verdict success \
  --statement "${USER_STATEMENT:?请先填入用户对本轮结果的真实原话}"
```

失败、中断或用户未确认时保留实际结果，不填成功；有明确失败结论且记录已整理时，使用同一命令的 `--verdict failure` 记录用户原话。若持物期间 Jev 请求介入或调用失败，程序可能中断且不会自动打开夹爪；现场人员按实际状态处理。

## 决策与参考格式

仅当 Jev 实际回答的 `choice` 与当前阶段技能匹配时才继续；符合条件的已有 `llm-route-plan.json` 仅提供决策上下文。回答为 `hold`、`observe_scene`、`request_astra` 或其他不匹配选择，以及调用失败时，进入现有停止与中断记录流程。持物期间并行请求返回阶段决定，保持满 5 秒且实际回答匹配后才返回；回答未到时继续等待并监测状态。

`--grasp-reference` 的 [`load_grasp_reference`](../../fr5_jev/prepare_varied_session.py#L695) 目前仅支持旧杯源记录格式。
