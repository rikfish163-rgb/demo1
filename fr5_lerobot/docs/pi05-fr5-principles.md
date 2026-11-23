# π0.5 与 FR5 微调设计

## 原理和开源实现

π0.5 不是更换视觉编码器。它以 PaliGemma 的 SigLIP 视觉编码器和语言模型为骨干，配合约 300M 参数的动作专家。相对 π0，典型配置将本体状态离散化后放入语言前缀，并通过时间 MLP 和 AdaRMSNorm 将流匹配时间注入动作专家。网络默认动作内部维度为 32、预测 50 个时间点；这不要求机器人有 32 个关节。[论文，附录 E](https://www.pi.website/download/pi05.pdf)；[OpenPI 模型配置](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0_config.py#L18-L41)；[动作专家实现](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L139-L186)。

论文的预训练混合多种机器人、环境、高层子任务、语言和互联网图文数据；机器人动作经 FAST 编码为离散 token。后训练结合语言或 FAST token 的交叉熵与连续动作的流匹配；论文中的分层推理先由同一模型给出语义子任务，再以它为条件预测底层动作。单个抓取任务可以直接以任务语言条件化底层策略，不能据此宣称复现了论文的高层推理。[论文，第 IV 节](https://www.pi.website/download/pi05.pdf)。

连续头训练时，令 `x_t = t·噪声 + (1-t)·动作`，学习速度场 `噪声 - 动作` 的均方误差；推理时从噪声出发反向积分，得到整段连续动作，官方默认 10 次数值积分。这里的 10 是去噪积分步数，不是机器人控制频率。[OpenPI 损失及采样](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/pi0.py#L188-L222)。

[知识隔离方案](https://www.pi.website/research/knowledge_insulation)让离散动作与语言目标训练视觉语言骨干，并阻断连续动作专家向骨干回传的梯度；它与只训练连续头、但允许梯度进入语言适配器的微调配方不同。OpenPI 公开的 π0.5 训练和推理只提供流匹配动作头；FAST 动作解码和论文的高层子任务生成不属于该接口。[OpenPI README](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/README.md#L5-L10)。

## FR5 的输入、动作和时间

- 可用观测是前置 RGB、真实机器人状态和任务语言。当前状态为 8 维：6 个关节角（rad）、夹爪开合比例、夹爪反馈有效位；动作是 7 维：6 个关节位置目标（rad）及一个夹爪目标。第 8 个状态分量是本工程对传感器有效性的扩展，不是第 7 个机械臂关节。[接口定义](../src/lerobot_robot_fairino/stream_contract.py#L7-L23)。
- OpenPI 默认将归一化状态按 256 个区间离散化，拼入 `Task … State … Action` 文本，再使用 PaliGemma tokenizer；实际配置存在覆盖项。FR5 状态和动作需用训练来源统计量归一化，保存并在推理时复用；关注接近常量的维度以及测试时超出训练范围的状态。[状态 tokenizer](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/src/openpi/models/tokenizer.py#L22-L33)；[归一化说明](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/norm_stats.md#L1-L28)。
- `6 Hz、50 点` 表示预测未来约 8.33 秒的动作块。基础对照检查点允许执行前 10 点、名义 1.67 秒；当前实机候选按原始顺序消费前三点、名义 0.5 秒后重新观测。新的执行窗口训练对照将 `n_action_steps` 设为 3，模型仍预测完整 50 点。实际刷新间隔还包含新观测、推理、通信及反馈等待。机器人目标频率、推理刷新频率和流积分步数各自不同。[实验配置](jev-pi05-experiment.md)；[运行时](../src/lerobot_robot_fairino/stream_runtime.py)。
- 前置相机的裁剪仍是同一相机，不能视为真实腕部相机；缺失视角应使用模型支持的缺失视角掩码，训练和推理的相机顺序、预处理一致。单目视角下遮挡与深度歧义会限制接近和接触时的精度。[图像配置](../training/stream_checkpoint.py#L69-L82)。
- FR5 的夹爪约定是 `0=关闭，1=张开`；官方 π0 预训练动作空间文档写的是 `0=张开，1=关闭`。这份官方文档针对 π0，不能据此断言所有 π0.5 权重都采用相同约定；FR5 的训练标签、状态、反归一化和设备输出必须一致。[FR5 接口](../src/lerobot_robot_fairino/stream_contract.py#L12-L22)；[官方 π0 动作空间](https://github.com/Physical-Intelligence/openpi/blob/215abfb217dbac7d5f1273282331b9b1866c0479/docs/norm_stats.md#L47-L67)。

## 当前训练入口的关键取舍

1. 从基础模型开始且不传 `--resume`、`--initialize` 时，[入口](../training/train_stream_pi05.py#L1324-L1361)会严格装载完整基础权重，然后注入适配器。默认只训练语言和专家的 LoRA 及完整动作投影。`full-expert-language-lora` 训练完整动作专家、语言 LoRA 和动作投影；视觉骨干、语言骨干本体及独立时间 MLP 仍冻结。`full-expert-vision-language-lora` 额外训练视觉 LoRA。这些都是从完整基础权重初始化的真实微调，但都不是全模型参数微调。完整专家模式还要求语言 LoRA 收到梯度，不能同时假定连续头梯度已被知识隔离切断。[参数选择](../training/train_stream_pi05.py#L1158-L1175)；[梯度要求](../training/train_stream_pi05.py#L2431-L2443)；[适配模式](../training/stream_checkpoint.py#L23-L32)。
2. 5,000 步对照使用普通有效七维流匹配均方误差，所有有效时间点和动作维度权重均为 1，并采用真实开合、运动和悬停的平衡采样。执行窗口对照保持七维权重相同，比较前三点权重 1 与权重 8，其余 47 点仍为 1；两组复用相同权重起点、标签、采样和优化器设置。加权改变了流匹配目标的时间分布，模型仍须学习完整连续动作块。动作块尾部无效填充不参与损失。[损失](../training/train_stream_pi05.py#L837-L881)；[实验配置](jev-pi05-experiment.md)。
3. 当前关节偏移模型以任务文本、前置全图和同帧工作区裁剪、本体状态条件化动作专家；history 使用当前帧之前的真实物理状态与年龄。输入中没有教师阶段、未来动作、剩余悬停时间或 `cup_action_clock`。[批次构造](../training/train_stream_pi05.py#L1686-L1728)；[物理历史](../training/train_stream_pi05.py#L1023-L1060)。
4. 数据导出代码将未来网格上的真实关节反馈插值作为关节位置目标，将已下发的夹爪命令或明确记录的初始保持作为夹爪目标；关节标签并非直接记录的 ServoJ 下发命令。训练出的动作应按位置轨迹目标解释，并与执行链的时序及跟踪误差对应。[导出语义](../src/lerobot_robot_fairino/export_stream_policy.py#L362-L368)。
5. 普通关节模式读取导出的分位数覆盖入口内现算的分位数；本工程导出器按训练划分汇总该统计量。使用具体数据集时应保持其物理来源划分与统计量一致。[训练入口](../training/train_stream_pi05.py#L216-L227)；[导出器](../src/lerobot_robot_fairino/export_stream_policy.py#L312-L331)。

状态覆盖同时影响连续归一化值和语言状态 token。服务器实际 LeRobot 0.6.1 使用 `np.digitize(state, np.linspace(-1, 1, 257)[:-1]) - 1`；归一化值低于 -1 时得到 -1，高端超出范围时落在 255。11 月 22 日的实际 J2、J3 均超出了 3,443 条训练状态的完整范围，分别归一化为约 1.161、-1.314。扩大多物体示范时需同时覆盖起始姿态、接近和撤离，并由新训练来源重算统计量。[实测状态覆盖](../results/jev-pi05-live-state-training-coverage-20261122-v2.json)。

追加真实来源的 `--initialize` 保留父模型权重，使用当前训练来源的统计量和新优化器；旧来源及其用户复核、留出角色保持一致。新统计量同时改变输入状态 token 与动作归一化目标，须经继续微调适应。相同来源的初始化保留父归一化，精确 `--resume` 同时恢复原优化器和采样状态。[来源与归一化处理](../training/train_stream_pi05.py#L2049)；[新增数据训练配置](jev-pi05-data-and-physical-evaluation.md)。

## 效果与边界

FR5 的六轴加夹爪适合表达七维抓取动作，官方预训练动作空间也包含六轴平行夹爪构型；这不保证任意形状、材质、重量、遮挡或场景都可抓取。预训练基础权重已包含多来源知识，FR5 微调样本则限定为 `fr5_jev` 真实采集数据；物体、姿态、背景和照明的覆盖仍决定可迁移范围。

以物理采集来源划分训练和留出集，分别记录见过和未见过物体、位置及背景的实机抓起与放回成功次数、总尝试次数和失败阶段；成功轨迹按工程约定由用户确认。离线损失、接口返回成功或预测轨迹相似都不能替代实机任务完成率。
