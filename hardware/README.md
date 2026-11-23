# FR5 / Gemini 335

两条抓杯流程共用 J1 和固定相机。J1 地址为 `192.168.58.2`，设备按 FR5 V5.0、软件 `v3.8.4.2` 使用；Gemini 335 序列号为 `CP1E54200015`。

| 文件 | 用途 |
| --- | --- |
| `sdk/Robot.py`、`sdk/LICENSE` | 法奥 SDK v2.1.4 的状态结构来源与许可证 |
| `fr5.urdf`、`kinematics-verification.json` | 六关节运动链及控制器 FK 对照结果 |
| `tool-reference.json` | 当前夹爪几何与抓取路线参考 |
| `calibration/camera-to-j1-mm.json` | 彩色相机坐标到 J1 基坐标的毫米变换 |
| `calibration/calibration-report.json`、`calibration/samples/` | 标定结果与十个样本的位姿记录 |

XML-RPC 使用端口 `20003`，连续状态流使用 `20004`。执行以 `tool0/user0` 为基准，保留控制器故障、使能、队列、夹爪接触和实际运动反馈检查。

标定棋盘格边长为 24.25 mm，留出点 RMS 为 1.586 mm，最大点误差为 2.347 mm。原始 RGB-D 与补充采集记录位于 `4090:/home/data4/hts/demo1/fr5-learning/results/astra-jev/handeye-quick-1792943357/` 和同级的 `handeye-supplement-1792943852/`。

操作章节依据官网 [3.8.4 系列历史手册](https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/intro.html)：[坐标与负载](https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/base.html)、[运动指令](https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/coding.html)、[夹爪](https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/robot_peripherals.html)、[状态与故障](https://fairino-doc-zhs.readthedocs.io/3.8.4/CobotsManual/appendix.html)。

## 已知问题

标定只验证了采样区域中的几何一致性，运行时用于观测定位。深度精度、独立抓取 TCP 与全臂线缆碰撞间隙尚未完成验证。3.8.4 系列手册页面标示 3.8.4.1，官网没有独立的 3.8.4.2 文档版本。
