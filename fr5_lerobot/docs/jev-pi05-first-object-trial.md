# 新物体的一轮教师采集

主入口为 [`./fr5-system collect-object`](../../fr5_runtime/system.py)，采集流程在 [`fr5_jev.collect_object`](../../fr5_jev/collect_object.py)。命令使用本机 `lerobot` Python、相机桥接和 FR5，阶段决策使用官方 `POST https://api.openai.com/v1/decisions`、模型 `gpt-6-luna`。默认仅准备，仍会连接服务器、采集设备观测并查询控制器 IK/FK；加 `--execute` 才执行机械臂和夹爪动作。用户已授权有人现场看护时继续，采集结果仍须由用户单独验收。

`R` 指 `4090:/home/data4/hts/demo1/fr5-learning/`；服务器整理和验收使用 `R/.venv/bin/python`。大文件、完整采集数据、视频、模型和训练输出永久保存在 `R`。本轮临时 RGB-D 可放在 `/dev/shm`，本机 recording spool 用作有界中转。执行前须满足文末的设备、存储和路径条件。

## 本轮输入

- `TRIAL_NAME`：本机 recording spool 和服务器均未使用过的简单名称，不含路径。每次调用使用新名称，包括只准备的调用和准备后的正式执行。
- `RGB_U`、`RGB_V`：当前原始 RGB 网格上物体内部的像素，使用 `--rgb-pixel U V`。低饱和度或颜色分割不可靠时，使用同一网格的 PNG `--mask`；两者必须且只能提供一个。命令内部重新采集 RGB-D，目标选择须适用于本轮现场布置。
- `--support-rgb-pixel U V`：可选，选取真实支撑表面的内部像素；例如紫块下方的绿色支撑。支撑按本轮 RGB 分割，其选择不得与目标重叠；提供后与目标一起做可见表面的工具扫掠检查。
- `OBJECT_DESC`：真实物体的单行描述。确认物体尺寸、材质、重量、支撑方式及持物时的净空。
- `CLOSE_PCT`：针对本物体和当前 PGI-140 指尖确认的 **目标开度**，整数 0–99；100 为全开，0 为全闭。它会原样成为夹爪闭合命令的整数目标，不是闭合比例，也没有经验证的毫米线性换算。实际接触可能在目标之前发生，不要用旧杯子的数值代替本轮确认。
- `GX GY GZ`：当前夹爪法兰坐标系下的有效夹持中心偏移，单位毫米，须有适用当前指尖和抓取部位的实际依据。本轮通过 `--grasp-offset` 显式提供，代码要求 `|GX|、|GY| ≤ 30`、`120 ≤ GZ ≤ 210`。
- `--grasp-below-top-mm`：夹持中心在可见顶部下方的距离，沿实测桌面法向计算，默认 10 mm；工具真实偏移保持不变。
- 默认 `--approach tilted`、`--tilt-from-down-deg 55`、`--tilted-closing-edge long`。`long` 跨可见长边闭合，`short` 跨可见短边闭合。入口还支持 `top-down` 和 `side`；`--side-retraction-mm` 仅用于 `side`，默认 0 mm，须小于可见短边的一半。
- 默认角色为 `--split training_complete_trial`；选模验证物体使用 `--split held_out_complete_trial`。准备路线固定为抓放一次、至少保持 5 秒并完整撤离。

## 一条命令准备或采集

在同一 shell 中设置本轮全新 `TRIAL_NAME` 和当前原始 RGB 像素 `RGB_U`、`RGB_V`，在工程目录执行：

```bash
cd /home/dianrobot/demo1
./fr5-system collect-object \
  --name "$TRIAL_NAME" \
  --object-description '紫色块状物体' \
  --rgb-pixel "$RGB_U" "$RGB_V" \
  --closing-percent 0 \
  --grasp-offset 5.208 8.072 162
```

示例的 `--closing-percent 0` 和法兰系偏移 `[5.208, 8.072, 162] mm` 仅适用于当前紫块及实测指尖配置。0% 是全闭目标，既有紫块实测在 68% 发生接触；新物体须重新确认目标开度和有效夹持中心，不能沿用这组值。

选择掩膜时，将 `--rgb-pixel "$RGB_U" "$RGB_V"` 换成 `--mask '本轮PNG绝对路径'`。需要检查抬高支撑时，追加 `--support-rgb-pixel U V`。默认 `long` 候选的闭合宽度取可见平面长边，80 mm 名义开口须严格大于闭合宽度加 6 mm。可见长边超过 80 mm 时，可显式使用 `--tilted-closing-edge short`，准备桌面平面内换向 90°、仍保持 55° 倾斜的候选，并按可见短边重新筛选。

不加 `--execute` 时返回 `prepared_no_motion`，包含本机会话目录、服务器观测路径、预检结果和 `task_success_verified=false`，不会下发运动或夹爪命令。准备后的正式执行须设置新的 `TRIAL_NAME`，向同一命令追加 `--execute`；新调用重新验证服务器、取得 fresh RGB-D、生成路线并做完整预检。准备目录和服务器观测名称已经占用，不能用原名称重试执行。

命令按实际流程完成：

1. **服务器验证**：在取得设备控制权和采集观测前，确认 SSH、`R` 存储根目录、至少 6 GiB 可用空间，以及服务器观测和 recording spool 均未占用本轮名称。
2. **新观测**：持有 J1 唯一控制租约，在 `/dev/shm` 临时目录采集本轮 RGB-D、标定和前后机器人状态。要求机器人就绪、夹爪反馈有效且空夹停稳；图像采集时间和观测完成时间均须满足 5 分钟新鲜度要求。
3. **完整预检**：根据本轮目标生成路线；用实测目标和可选支撑点云检查当前全开工具盒的扫掠、外壳余量和目标与指尖闭合区域的交叠，再检查工具到实测桌面的净空，以及完整路径的控制器 IK/FK 和关节余量。
4. **证据上传**：保存 `collection-preflight.json`，将本轮观测、预检报告和 `path-samples.json` 上传到 `R/observations/<TRIAL_NAME>/`。预检或上传失败时停止，正式执行器尚未启动；失败的路径预检也保留报告和观测证据。
5. **正式执行**：只有 `--execute` 才启动 [`WholeTaskSession`](../../fr5_jev/whole_task_collection.py)，由官方模型匹配阶段后完成接近、闭合、提升、保持、返回、释放和完整撤离。
6. **归档与录像**：录制关闭后释放 J1 控制权，自动调用 archive 和录像物化。完整记录返回 `pending_review`；原始流与物化数据保存在 `R/recording-spool/<TRIAL_NAME>/`，完整录像保存在 `R/reviews/<TRIAL_NAME>/full-process.mp4`。

本机会话为 `/home/dianrobot/.local/state/fr5-recording-spool/<TRIAL_NAME>/`。准备完成后可查看 `scene.jpg`、`selection-mask.png`、`geometry.json`、`scope.json` 和 `collection-preflight.json`；倾斜几何记录 `tilt_from_down_deg`、`tilted_closing_edge`、`visible_closing_width_mm`、`opening_axis_base` 和 `tool_axis_base`。启动时再次核对本轮起始 TCP、物体和现场布置，夹爪未全开时执行器记录并下发全开命令。

## 执行和验收

程序按已准备路线完成接近和接触，闭合目标为 `CLOSE_PCT`，要求有效的夹爪接触状态 `motiondone=2`；再沿实测桌面法向从接触位试提 10 mm，继续到距接触位 50 mm，保持至少 5 秒，沿路线返回同一接触 TCP，以 100% 开度释放，再完整撤离。50 mm 是相对接触位的总目标位移。

`hold-01.json` 的 `start_monotonic` 是提升命令完成后的实际保持起点，`end_monotonic` 是返回命令的实际下发时间，`actual_duration_s=end_monotonic-start_monotonic`；另记录真实提升完成时间 `lift_completed_monotonic` 及从该时刻到返回下发的 `actual_hover_from_lift_completion_s`。网络或决策等待可能使保持时间长于 5 秒。返回决策可在保持期间并行取得，返回动作仍须等待保持满 5 秒且模型选择匹配。现场异常时停止并保留记录。

完整执行后检查会话中的 `collection-result.json`、`summary.json`、`hold-01.json`、实际命令日志、决策与重观测记录，以及服务器录像；验收目录为 `R/recording-spool/<TRIAL_NAME>/materialized/`。最终命令完成后还须录到新的配对观测。只有阶段、实际命令、保持时间、释放和录制均通过内部完整性检查，结果才进入 `pending_review`，此时 `task_success_verified=false`，训练就绪仍为 `false`。

用户明确确认本轮物体被抓起、保持至少 5 秒、原位放稳并松开后，记录用户实际原话。`success` 结论及成功/训练就绪字段的 `true` 值须有该用户验收和完整来源校验：

```bash
./fr5-system review --episode "$TRIAL_NAME" --verdict success \
  --statement "${USER_STATEMENT:?请先填入用户对本轮结果的真实原话}"
```

失败、中断或用户未确认时保留实际结果；有明确失败结论且记录已整理时，使用同一命令的 `--verdict failure` 记录用户原话。`interrupted` 来源不能作为成功训练数据。

## 决策与参考格式

官方决策记录使用 `provider=openai`、`requested_model=gpt-6-luna`、`selected_by=openai_decisions`，请求和响应模型均为 `gpt-6-luna`；`scope.contract=fr5-whole-task-collection-v1`，`state.schema=fr5-cup-shadow-v1`，`state.source=live_readonly`。符合条件的已有 `llm-route-plan.json` 提供路线上下文，阶段续行依据实际模型回答。

首次决策使用连续录制已接受的 paired RGB+robot。`state.facts.scene_observation` 保留 `frame_index`、原始 `camera` 与 `robot`、`camera_time`、`robot_time_monotonic`、`previous_completion_monotonic` 和 `clock=host_monotonic`；两种时间均须严格晚于上一实际命令的完成时间。首阶段无前序实际命令时，完成时间为 `null`；若执行过 `initial_open`，则引用其实际完成时间。`captured_at_unix` 取该帧原始 `source_timestamp_unix`，`current_tcp_mm_deg` 取同一配对 `robot.tcp`。首请求保持官方 `input_json` 格式，RGB 通过帧索引关联连续录制。

模型选择 `observe_scene` 时，记录一次新观测并请求后续决策：

1. `jev-cNN-phase.json` 保存 `route=observe_scene`、`automatic_continuation=false`、`observation_record` 和 `followup_decision_id`。
2. 等待 camera 和 robot 时间均晚于 `requested_monotonic` 的新配对帧，保存 `observe-cNN-phase.json` 与同帧 RGB 编码的 `observe-cNN-phase.jpg`。JSON 记录 `requested_by_decision`、`requested_monotonic`、`requested_unix`、`record`、`image_file` 及配对字段；`source_timestamp_unix` 保留新帧原始值。
3. `jev-cNN-phase-after-observe.json` 的 `state.facts.scene_observation` 与观测 JSON 一致；请求在原 `input_json` 后追加官方 `input_image`，`image_url` 为 `data:image/jpeg;base64,...`。JPEG 使用与 continuous 帧相同的 RGB 和 OpenCV JPEG90 编码；入口核验 inline、保存文件、对应 capture JPEG 三方字节一致，以及 phase、cycle、前序命令完成和请求响应时序。
4. 后续回答的 `choice` 与当前 `PHASE_SKILLS[phase]` 匹配后继续。每阶段最多刷新一次；后续仍选择 `observe_scene`、选择 `hold`/`request_astra`、其他不匹配选择、拒绝或调用/观测失败时，停止并记录中断。返回动作同时受真实保持时长约束。

`actual_hold_duration_s` 在返回命令真实下发后填写；保持中的预取决策使用 `hold_in_progress`。`actual_return_endpoint` 来自返回阶段实测 `final_state`，残差相对计划返回 TCP 计算。`object_support_visually_verified=null`；物体是否跟随、放稳及未滑落由用户验收。

## 实测记录与当前限制

紫块教师记录 `fr5-purple-tilted-luna-teacher-20261122T235616-v1` 实测闭爪停在 68%、TCP 提升 50.54 mm、保持 5.27 秒并完成返回；松爪前模型选择 `observe_scene`，该轮以 `interrupted` 结束。183 帧、30.338 秒完整录像已归档，新增成功训练来源数为 0。该记录保留实际结果，新的完整采集仍须释放、撤离和用户验收；详见[实测结果](jev-pi05-experiment.md)。

- 2026-11-23 15:54（Asia/Shanghai）的夹爪反馈为 `active=1`、`open=100`、`done=1`、`fault=0`，夹爪已恢复。4090 SSH 入口的 `kex reset` 尚未恢复，当前不能录制或部署；连接恢复后仍须完成存储及入口就绪检查，再取新观测。
- 当前紫块位置的斜抓端点尚未通过控制器 IK；跨短边、桌面平面内换向 90° 也只作为候选。新摆位须用本轮 fresh RGB-D 和完整预检确认，不能以规划端点或旧路线代替。
- 可见表面、80 mm 名义开口和工具盒桌面预检不覆盖隐藏物体体积、真实指尖开口、整臂、线缆与现场净空。持物期间中断不会自动打开夹爪，现场人员须按实际状态处理。
- 现场人员仍须检查真实物体、整臂、线缆、环境，以及末端工具与物体的总负载及质心。准备完成后场景改变或观测过期，须用新名称重新调用 `collect-object`。
- 过期观测、未经用户验收的轨迹、失败或中断记录不能作为新成功训练来源。
