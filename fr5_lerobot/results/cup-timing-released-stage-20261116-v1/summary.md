# 杯任务 released 状态证据

计时器在先前夹持之后，用有效原生夹爪反馈确认开爪，将该观测的 phase 标为 `released`。原始时钟、跟踪值和任务指令保留；`released` 表示反馈确认开爪，整轮抓起并放稳由用户确认。

[summary.json](summary.json) 保存逐源 phase 统计与时序结果；[changed-frames.jsonl](changed-frames.jsonl) 保存 227 条变化帧；[validation.json](validation.json) 保存一致性结果。[pgi-corrected-phase-evidence.json](pgi-corrected-phase-evidence.json) 保存来源 7 的四条原始反馈配对证据。

来源 7 的 195、578 帧保持接触 phase；309、692 帧由有效 99% 开度与先前夹持反馈确定为 released。release phase 的完整模型配对评分见 [paired-score](../pi05-cup-release-phase-paired-score-20261116-v1.json)。

服务器原始审计目录：`/home/data4/hts/demo1/fr5-learning/audits/cup-timing-released-stage-20261116-v1/`。
