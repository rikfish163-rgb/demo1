# astra + Jev 抓杯结果

`user-success-registry/` 保留两次用户确认的完整逻辑任务。第一条任务由两段原始录制组成；两条记录的 `training_ready` 均为 `false`。

| 结果目录 | 已记录结果 |
| --- | --- |
| `astra-jev-collection-1793147589/` | 监督抓杯、抬升 10.036 mm、放回和张开夹爪；5 次 Jev 调用，中位延迟 843.6 ms |
| `jev-collaboration-experiment/` | 6 次 Jev 调用、4 段实际运动；接触阶段转交 LLM 判断 |
| `cup-grasp-final-attempt/` | 下降过程中触发碰撞停止，保留最终摘要与故障反馈 |
| `user-calibration-comparison-1793111447/` | 标定矩阵处置及用户确认的夹爪净开口 80 mm |
| `recovery-after-reboot-20261029/` | 中断后的放回、松开和退离结果 |
| `telemetry-after-reboot-20261029/` | 重启后的只读状态摘要 |
| `pi05-reboot-audit-20261029/` | 两条早期示范的保留数据与导出审计 |
| `jev-shadow/` | Jev 阶段决策实例 |

原始观测、下发动作、控制器反馈和录制源码位于 `4090:/home/data4/hts/demo1/fr5-learning/results/astra-jev/`，其下使用上表相同的目录名。现行教师采集的八条验收示范位于服务器 `recording-spool/`，数据索引见 [pi0.5 主线](../../fr5_lerobot/PI05_FILES.md)。

## 已知问题

10 mm 监督完成记录与预抓取实验不计入两条用户验收记录。早期示范在电脑重启后失去了部分原始 RGB-D 和遥测，现有保留数据的范围见 `pi05-reboot-audit-20261029/REPORT.txt`。中断恢复不增加成功次数。
