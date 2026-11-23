# 原始 RGB-D 与反馈逐帧审计

范围：stream_provenance.json 中 8 个 materialized 教师来源。结构化报告见 [raw-rgbd-report.json](success8-data-review-20261102-v1/raw-rgbd-report.json)，逐帧表与原始样例保存在服务器审计目录。

来源声明 4209 帧，实际解析 4209 帧；压缩训练集元数据为 3855 帧，两者口径不同。
RGB 文件存在/解码：4209/4209；深度 NPZ 存在/解码：4209/4209。
机械臂 joints 六个有限数：4209；TCP 六个有限数：4209；夹爪有效标志 true/false：3981/228；标志与按源码规则重算不一致/缺失：0。
JPEG 文件 SHA 匹配：4209；深度流压缩载荷 SHA 匹配：4209；NPZ 恢复 float32 源 SHA 匹配：4209；直接解码流深度与 NPZ 逐字节相同：4209。
深度有限像素：969753600/969753600；有限且非零像素：936931140/969753600；含非有限像素的帧：0；全零或非有限的帧：0。

相机的来源色彩图像与 SDK depth_to_color 网格分别缩放；元数据中的 same_pixel_correspondence_verified、raw_depth_in_source_packet、raw_depth_retained_in_bundle 按逐源分布见下表和 JSON。无损指保存/传输的 float32 深度可恢复，并不证明原生 Y16 深度被保存，也不证明 RGB 与深度严格同像素对齐。JPEG 是有损编码：编码前 RGB 的 source SHA 无法由 JPEG 恢复核实，不应作为损坏计数。

| 源序号 | 帧数 | RGB 解码 | 深度解码 | 夹爪有效 false | 原始 Y16 保留标志 | 同像素验证标志 |
|---:|---:|---:|---:|---:|---|---|
| 0 | 649 | 649 | 649 | 22 | {"false": 649} | {"false": 649} |
| 1 | 370 | 370 | 370 | 22 | {"false": 370} | {"false": 370} |
| 2 | 402 | 402 | 402 | 25 | {"false": 402} | {"false": 402} |
| 3 | 754 | 754 | 754 | 45 | {"false": 754} | {"false": 754} |
| 4 | 435 | 435 | 435 | 23 | {"false": 435} | {"false": 435} |
| 5 | 376 | 376 | 376 | 23 | {"false": 376} | {"false": 376} |
| 6 | 471 | 471 | 471 | 22 | {"false": 471} | {"false": 471} |
| 7 | 752 | 752 | 752 | 46 | {"false": 752} | {"false": 752} |

像素尺寸、dtype、深度单位、每帧有效率及时间偏移分布见 JSON；每帧关节、TCP、夹爪、文件路径、几何标志及校验结果见 JSONL。源 RGB JPEG 和 NPZ 深度样例为字节复制，未改图。

训练输入核对：数据集 info.json 和训练检查点的输入配置、预处理器及训练记录保存在 JSON。该检查点记录数据集为本次 compacted 数据集；批次构造使用前视 RGB、8 维状态和任务文本，动作作为训练标签；未发现深度特征或深度输入。模型配置中的三个 RGB 槽并不等同于三个实体相机。

逐帧表：/home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/raw-rgbd-frames.jsonl
机器可读报告：/home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/raw-rgbd-report.json


## 补充核验结果

- 八条来源的实际帧数依次为 649、370、402、754、435、376、471、752，合计 4,209，与各自声明一致；八条用户复核记录的 SHA 与来源元数据一致，记录的 verdict 均为 success。实机结果由用户确认。
- 4,209 张 JPEG 均可解码为 640×360、RGB uint8；4,209 个 NPZ 均可解码为 360×640、float32。各来源的场景预览 scene.jpg 已实际读取，均为 1920×1080；不能把它当成逐帧保存的 640×360 画面。
- 每帧 media_codec.depth_unit 和 camera.capture.depth_unit 均为 mm，aligned_depth_unit_mm 均为 1.0。总计 969,753,600 个深度像素全部有限，其中 936,931,140（96.6154%）有限且非零，32,822,460（3.3846%）为零；全部帧存在零深度像素，但没有全零帧或非有限像素。逐帧非零率范围为 96.0673%–97.1558%。
- 所有帧的 joints、TCP 均为六个有限数，夹爪位置、激活、故障、故障 ID、动作完成、速度、电流字段均存在且为有限数。228 帧的夹爪有效标志为 false，全部因为 gripper_motiondone=0 不满足项目有效性规则；记录值与重算标志不一致 0 帧。其余 3,981 帧标志为 true；有效标志不等于抓取成功。
- 机器人接收时刻减去相机源接收时刻：中位数 +0.0183 ms，绝对值 P95 4.736 ms，范围 -5.078 至 +5.033 ms；全部采用 host_monotonic，记录偏移和实际差值不一致超过 1 微秒的帧为 0。设备侧 color 时间减 depth 时间中位数 +1.562 ms，范围 -16.115 至 +17.829 ms；这些时间不能保证曝光时刻同步。
- 4,209 帧的原始几何标记全部一致：RGB 为未经校正的源色彩图缩放，深度为独立缩放的 SDK depth_to_color 网格；same_pixel_correspondence_verified=false、raw_depth_in_source_packet=false、raw_depth_retained_in_bundle=false 均为 4,209 帧。无损恢复的是传输前 float32 深度，不能推出原生 Y16 或严格同像素对齐。
- 逐帧原流的包载荷、JPEG 字节和压缩深度载荷哈希、materialized JPEG 哈希、恢复的 float32 源哈希分别全部匹配；4,209 份原流解码深度与对应 NPZ 深度逐字节相同。JPEG 有损，编码前 RGB 的 source SHA 仅有记录，无法从 JPEG 反推核实。
- 训练输入核验限定于检查点 pi05-fr5-stream-workspace-only-vision-n10-gpu9-20261102-v1/step-000250：其记录的数据集是本次 3,855 帧 compacted 集，训练 episode 0–6，保留 episode 7 作验证；批次为前视 RGB、8 维状态、任务文本及动作标签，没有深度输入。其 workspace-only 图像配置从 640×360 前视图裁剪 [240,100,640,360]，并重命名到 base RGB 输入槽。

## 原文件字节复制样例

来源 0 的帧 0、325、648 各复制了一对 JPEG 与 NPZ；帧 325 位于来源元数据记录的保持区间 309–341。复制后的六个文件与源文件 SHA 全部相同：

- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000000-rgb.jpg
- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000000-depth.npz
- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000325-rgb.jpg
- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000325-depth.npz
- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000648-rgb.jpg
- /home/data4/hts/demo1/fr5-learning/audits/success8-data-review-20261102-v1/sample-source00-frame000648-depth.npz

训练代码及检查点证据位于 /home/data4/hts/demo1/fr5-learning/checkpoints/pi05-fr5-stream-workspace-only-vision-n10-gpu9-20261102-v1/step-000250/；批次构造见 train_stream_pi05.py:1059，裁剪见 stream_image_profile.py:42，输入重命名见 policy_preprocessor.json。
