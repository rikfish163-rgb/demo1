"""Packed FR5 state layouts copied from the locally verified official SDK.

Source: hardware/sdk/Robot.py, classes at lines 20–136.
Source SHA256: f2e66e824a9b38b4240a1a2998c492ae04a8211eab55e75831fc44bedf604c5e
The four field lists are copied verbatim. Structure uses explicit little endian
instead of native endian; no SDK code is imported or executed at runtime.
The verified v3.8.4.2 stream has two additional trailing bytes. Only fields
preceding this tail are exposed by state_stream; checksum is read from the
actual packet end, never from the copied structure's checksum field.
"""

import ctypes
from ctypes import (
    LittleEndianStructure as Structure,
    c_byte, c_int, c_double, c_float, c_uint16, c_ushort,
)

SOURCE_SHA256 = "f2e66e824a9b38b4240a1a2998c492ae04a8211eab55e75831fc44bedf604c5e"

class ROBOT_AUX_STATE(Structure):
    _pack_ = 1
    _fields_ = [
        ("servoId", c_byte),         # 伺服驱动器ID号
        ("servoErrCode", c_int),     # 伺服驱动器故障码
        ("servoState", c_int),       # 伺服驱动器状态
        ("servoPos", c_double),      # 伺服当前位置
        ("servoVel", c_float),       # 伺服当前速度
        ("servoTorque", c_float),    # 伺服当前转矩
    ]

class EXT_AXIS_STATUS(Structure):
    _pack_ = 1
    _fields_ = [
        ("pos", c_double),        # 扩展轴位置
        ("vel", c_double),        # 扩展轴速度
        ("errorCode", c_int),     # 扩展轴故障码
        ("ready", c_byte),        # 伺服准备好
        ("inPos", c_byte),        # 伺服到位
        ("alarm", c_byte),        # 伺服报警
        ("flerr", c_byte),        # 跟随误差
        ("nlimit", c_byte),       # 到负限位
        ("pLimit", c_byte),       # 到正限位
        ("mdbsOffLine", c_byte),  # 驱动器485总线掉线
        ("mdbsTimeout", c_byte),  # 控制卡与控制箱485通信超时
        ("homingStatus", c_byte), # 扩展轴回零状态
    ]

class WELDING_BREAKOFF_STATE(Structure):
    _pack_ = 1
    _fields_ = [
        ("breakOffState", ctypes.c_uint8),        # 焊接中断状态
        ("weldArcState", ctypes.c_uint8),        # 焊接电弧中断状态
    ]

class RobotStatePkg(Structure):
    _pack_ = 1
    _fields_ = [
        ("frame_head", c_uint16),      # 帧头 0x5A5A
        ("frame_cnt", c_byte),         # 帧计数
        ("data_len", c_uint16),        # 数据长度
        ("program_state", c_byte),     # 程序运行状态，1-停止；2-运行；3-暂停
        ("robot_state", c_byte),       # 机器人运动状态，1-停止；2-运行；3-暂停；4-拖动
        ("main_code", c_int),          # 主故障码
        ("sub_code", c_int),           # 子故障码
        ("robot_mode", c_byte),        # 机器人模式，0-自动模式；1-手动模式
        ("jt_cur_pos", c_double * 6),  # 机器人当前关节位置，假设有6个关节
        ("tl_cur_pos", ctypes.c_double * 6),  # 工具当前位姿
        ("flange_cur_pos", ctypes.c_double * 6),  # 末端法兰当前位姿
        ("actual_qd", ctypes.c_double * 6),  # 机器人当前关节速度
        ("actual_qdd", ctypes.c_double * 6),  # 机器人当前关节加速度
        ("target_TCP_CmpSpeed", ctypes.c_double * 2),  # 机器人TCP合成指令速度
        ("target_TCP_Speed", ctypes.c_double * 6),  # 机器人TCP指令速度
        ("actual_TCP_CmpSpeed", ctypes.c_double * 2),  # 机器人TCP合成实际速度
        ("actual_TCP_Speed", ctypes.c_double * 6),  # 机器人TCP实际速度
        ("jt_cur_tor", ctypes.c_double * 6),  # 当前扭矩
        ("tool", ctypes.c_int),  # 工具号
        ("user", ctypes.c_int),  # 工件号
        ("cl_dgt_output_h", ctypes.c_byte),  # 数字输出15-8
        ("cl_dgt_output_l", ctypes.c_byte),  # 数字输出7-0
        ("tl_dgt_output_l", ctypes.c_byte),  # 工具数字输出7-0(仅bit0-bit1有效)
        ("cl_dgt_input_h", ctypes.c_byte),  # 数字输入15-8
        ("cl_dgt_input_l", ctypes.c_byte),  # 数字输入7-0
        ("tl_dgt_input_l", ctypes.c_byte),  # 工具数字输入7-0(仅bit0-bit1有效)
        ("cl_analog_input", ctypes.c_uint16 * 2),  # 控制箱模拟量输入
        ("tl_anglog_input", ctypes.c_uint16),  # 工具模拟量输入
        ("ft_sensor_raw_data", ctypes.c_double * 6),  # 力/扭矩传感器原始数据
        ("ft_sensor_data", ctypes.c_double * 6),  # 力/扭矩传感器数据
        ("ft_sensor_active", ctypes.c_byte),  # 力/扭矩传感器激活状态， 0-复位，1-激活
        ("EmergencyStop", ctypes.c_byte),  # 急停标志
        ("motion_done", ctypes.c_int),  # 到位信号
        ("gripper_motiondone", ctypes.c_byte),  # 夹爪运动完成信号
        ("mc_queue_len", ctypes.c_int),  # 运动队列长度
        ("collisionState", ctypes.c_byte),  # 碰撞检测，1-碰撞；0-无碰撞
        ("trajectory_pnum", ctypes.c_int),  # 轨迹点编号
        ("safety_stop0_state", ctypes.c_byte),  # 安全停止信号SI0
        ("safety_stop1_state", ctypes.c_byte),  # 安全停止信号SI1
        ("gripper_fault_id", ctypes.c_byte),  # 错误夹爪号
        ("gripper_fault", ctypes.c_uint16),  # 夹爪故障
        ("gripper_active", ctypes.c_uint16),  # 夹爪激活状态
        ("gripper_position", ctypes.c_byte),  # 夹爪位置
        ("gripper_speed", ctypes.c_byte),  # 夹爪速度
        ("gripper_current", ctypes.c_byte),  # 夹爪电流
        ("gripper_tmp", ctypes.c_int),  # 夹爪温度
        ("gripper_voltage", ctypes.c_int),  # 夹爪电压
        ("auxState", ROBOT_AUX_STATE),  # 485扩展轴状态
        ("extAxisStatus", EXT_AXIS_STATUS*4),  # UDP扩展轴状态
        ("extDIState", c_uint16*8),  # 扩展DI输入
        ("extDOState", c_uint16*8),  # 扩展DO输出
        ("extAIState", c_uint16*4),  # 扩展AI输入
        ("extAOState", c_uint16*4),  # 扩展AO输出
        ("rbtEnableState", ctypes.c_int),  # 机器人使能状态
        ("jointDriverTorque", ctypes.c_double * 6),  # 关节驱动器当前扭矩
        ("jointDriverTemperature", ctypes.c_double * 6),  # 关节驱动器当前温度
        ("year", ctypes.c_uint16),  # 年
        ("mouth", ctypes.c_uint8),  # 月
        ("day", ctypes.c_uint8),  # 日
        ("hour", ctypes.c_uint8),  # 小时
        ("minute", ctypes.c_uint8),  # 分
        ("second", ctypes.c_uint8),  # 秒
        ("millisecond", ctypes.c_uint16),  # 毫秒
        ("softwareUpgradeState", ctypes.c_int),  # 机器人软件升级状态
        ("endLuaErrCode", ctypes.c_uint16),  # 末端LUA运行状态
        ("cl_analog_output", ctypes.c_uint16 * 2),  # 控制箱模拟量输出
        ("tl_analog_output", ctypes.c_uint16),  # 工具模拟量输出
        ("gripperRotNum", ctypes.c_float),  # 旋转夹爪当前旋转圈数
        ("gripperRotSpeed", ctypes.c_uint8),  # 旋转夹爪当前旋转速度百分比
        ("gripperRotTorque", ctypes.c_uint8),  # 旋转夹爪当前旋转力矩百分比
        ("weldingBreakOffState", WELDING_BREAKOFF_STATE), # 焊接中断状态
        ("jt_tgt_tor", ctypes.c_double * 6),  # 关节指令力矩
        ("smartToolState", ctypes.c_uint16),  # SmartTool手柄按钮状态
        ("wideVoltageCtrlBoxTemp", ctypes.c_float),  # 宽电压控制箱温度
        ("wideVoltageCtrlBoxFanCurrent", ctypes.c_uint16),  # 宽电压控制箱风扇电流(ma)
        ("check_sum", c_ushort)]
