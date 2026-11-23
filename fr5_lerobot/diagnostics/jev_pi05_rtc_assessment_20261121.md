# history5000 π0.5 的 RTC 接入评估

推理时 guidance RTC 可以复用当前普通流匹配检查点。项目的七维完整 VJP 处理器已实现，并完成真实现场输入的 GPU 推理比较。**有前缀时，本地推理中位耗时为 0.776 秒，超过前三点的 0.5 秒执行窗口；当前实机候选继续使用普通十步去噪，执行前三点后重新观测。** RTC 尚未接入实际控制队列。

Physical Intelligence 的 [RTC 论文](https://www.pi.website/download/real_time_chunking.pdf)在执行旧 chunk 的同时预测新 chunk，用旧动作对重叠部分作随时间衰减的引导；其[公开 Kinetix 实现](https://github.com/Physical-Intelligence/real-time-chunking-kinetix/blob/main/src/model.py)对去噪结果计算完整向量–雅可比积（VJP）。公开仓库是仿真实验实现，不等于本项目现成的远端机器人客户端。引导是软约束，不能当成已承诺动作必然逐值不变。

## 安装版接口与检查点

服务器 `lerobot==0.6.1` 的 `PI05Policy.predict_action_chunk` 将 `inference_delay`、`prev_chunk_left_over`、`execution_horizon` 传给 `sample_actions`。首次无前缀时传 `prev_chunk_left_over=None`；有前缀时传模型空间的 `[N, 7]` 或 `[1, N, 7]` Tensor，并提供非负整数延迟步数。`select_action` 在 RTC 启用时不支持。本次 GPU 比较使用项目处理器、LINEAR schedule 和 `max_guidance_weight=10.0`。[LeRobot 0.6.1 RTC 文档](https://huggingface.co/docs/lerobot/v0.6.1/rtc)

`execution_horizon` 是**前缀引导权重的终点**，不是当前控制器执行多少点：前 `min(inference_delay, execution_horizon, N)` 点权重为 1，之后按 schedule 衰减；处理器会把终点截到传入前缀的真实长度。检查点 `n_action_steps=10` 则是现有执行前缀，两者不能混用。history5000 的 `rtc_config=null`，训练目标是普通 joint-delta π0.5，未训练 RTC 前缀条件；这里讨论的是无需改变训练目标的推理时 guidance，不是训练时硬前缀模型。

## 动作空间与重锚

history5000 使用 `state-history-physical-v1`、`full-and-workspace-v1`、50×7 输出和 10 次去噪。状态是本次观测的 6 轴绝对弧度、夹爪位置和有效位；训练的动作前 6 维是**同一观测锚点的关节差**，第 7 维是**绝对夹爪比例**。[项目转换函数](../training/relative_stream_actions.py)和检查点的 `action_representation.json` 均明确这一约定。

保存的 preprocessor 对相对动作的 7 维使用 action `q01/q99` 归一化；postprocessor 只反归一化到相对动作，随后 [serve_stream_pi05.py](../training/serve_stream_pi05.py) 用**该请求的原始状态**恢复绝对关节位置，网络回复是绝对 50×7。`use_relative_actions=false`，官方 relative/absolute processor 没有启用。通用 RTC rollout 仅在发现启用的 `RelativeActionsProcessorStep` 时自动重锚，因此不能直接复用其默认队列处理。

[`normalize_absolute_prefix`](../training/stream_rtc.py)将旧绝对目标重新表示在新观测锚点下，再复用检查点的 action normalizer。实际接线时，`old_abs` 必须来自与新观测时刻对齐、控制器尚会执行的承诺目标，包括夹爪范围映射、取整和 deadband 后的实际目标；`s_new` 是同一观测配对的原始实测状态。只使用允许执行的旧前缀。对非空 `[N,7]` 前缀：

```python
delta = relative.to_relative_actions(old_abs, s_new,
                                     input_representation=relative.ABSOLUTE)
prev_norm = normalizer({TransitionKey.ACTION: delta})[TransitionKey.ACTION].to(device)
```

即 `delta[:, :6] = old_abs[:, :6] - s_new[:6]`，`delta[:, 6] = old_abs[:, 6]`；normalizer 使用保存的 action 分位数及其零分母处理，不额外裁剪。`prev_norm` 作为 RTC 参数传入，不放入模型 batch 的 `action` 标签。模型输出仍依次经原 postprocessor 和 `to_absolute_actions(..., s_new)`，不要再开启一组官方相对动作处理器。新请求的图像和 8 秒历史仍来自真实观测。

## 完整 VJP 与真实维度

[`FR5RTCProcessor`](../training/stream_rtc.py)复用 LeRobot 0.6.1 的时间权重和 guidance 公式，在计算速度场前使潜变量参与求导。误差仅包含传入前缀的真实 N 个时间点和七个动作维度，完整 VJP 对全部潜变量求导。速度场可能跨维耦合，因此第 8–32 维的输入修正可以非零；屏蔽的是无动作语义的误差项。

外层使用 `torch.no_grad()`，局部使用 `enable_grad()`，模型权重冻结。服务器安装版 `RTCProcessor` 在计算速度场后才设置 `requires_grad`，不能得到完整 VJP；现有普通服务的 `inference_mode()` 路径也不能直接启用局部求导。[LeRobot 对应版本源码](https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/policies/rtc/modeling_rtc.py)。

CPU 定向验证共八项通过，覆盖完整 VJP、跨维耦合、真实前缀长度和七维重锚。解析例 `v(x)=2x, t=0.5` 的完整 VJP 为零；跨维例允许第八维输入产生修正。history5000 实际 normalizer 的绝对目标往返最大误差为零，夹爪保持绝对值。

## 真实输入 GPU 比较

使用已保存的五条新鲜现场请求构成四对输入，在 GPU 9 上比较普通推理、RTC 无前缀和 RTC 有前缀。同一对使用相同的显式 1×50×32 噪声、原生十次去噪和检查点后处理，保存完整 50×7 输出。前缀来自上一条预测允许的前十点，按源记录的名义时间网格截取，并应用原生夹爪范围映射与取整。

| 路径 | 本地推理与后处理中位秒 | 最大秒 | 峰值显存 MiB |
| --- | ---: | ---: | ---: |
| 普通 eager | 0.316 | 0.592 | 9,760 |
| RTC 无前缀 | 0.341 | 0.346 | 9,760 |
| RTC 有前缀 | 0.776 | 0.789 | 10,327 |

普通推理与保存的现场输出逐值一致，RTC 无前缀与普通推理也逐值一致。有前缀时的拟合结果为：

| 输入对 | 真实前缀点数 | 权重为 1 的点数 | 前缀六轴最大差 ° | 前缀夹爪最大差百分点 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4 | 4 | 0.213 | 0.143 |
| 2 | 6 | 4 | 0.246 | 0.172 |
| 3 | 7 | 4 | 0.125 | 0.918 |
| 4 | 6 | 4 | 0.121 | 0.213 |

全部权重为 1 的点中，六轴最大差为 0.213°，夹爪最大差为 0.918 个百分点。首对输入的旧预测为 76% 夹爪目标，普通新预测接近 100%；RTC 引导后前十点与普通预测的夹爪最大差为 24.16 个百分点，整块夹持点位置最大差为 69.95 mm。其余三对的整块位置最大差为 2.88–4.43 mm。这些变化体现条件对输出的影响。

完整结果保存在服务器：

`/home/data4/hts/demo1/fr5-learning/diagnostics/jev-pi05-history5000-rtc-benchmark-20261121-v1/{summary.json,runs.jsonl}`

小型数值比较见[RTC 基准结果](../results/jev-pi05-rtc-benchmark-20261121-v1.json)，可运行入口为 [`benchmark_jev_pi05_rtc.py`](benchmark_jev_pi05_rtc.py)。

## 执行取舍与限制

普通现场热推理的观测到回复为 0.512–0.578 秒、中位 0.552 秒，服务器为 0.315–0.329 秒、中位 0.322 秒。RTC 有前缀的本地耗时已经需要至少五个 6 Hz 名义周期，完整往返、预检和提交还会增加延迟。基准使用普通现场回复估计的四点引导延迟，不能直接作为 RTC 执行配置。

当前选择前三点普通执行，以缩短返回阶段未经重观测的动作范围，代价是观测与推理之间存在等待。若继续 RTC 接线，需要在真实旧队列有足够覆盖时启动推理，使用实测消费时间与观测时刻对齐候选，并保留已提交目标；可切换后缀的时间边界由控制器记录确定。

本基准的旧前缀是未执行的模型预测，未模拟控制器承诺、夹爪死区或物理接触。本地耗时不包含 JPEG 预处理、SSH 和路径预检。四对输入的夹爪反馈均无效，不能据此评估持物或任务成功。此次 GPU 基准的机器人与夹爪命令数为零，完整策略实机抓放成功确认数仍为零。
