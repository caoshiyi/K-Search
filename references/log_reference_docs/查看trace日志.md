# 查看trace日志

**页面ID:** logreference_0022  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/logreference/logreference_0022.html

---

#### 简介

trace机制是指，在程序运行过程中将软件栈的维测信息记录在内存中，当程序运行出错或进程结束时落盘到文件，以免在程序运行过程中频繁产生和记录日志文件，影响性能。当前仅Ascend EP标准形态支持该功能。

#### 日志说明

trace日志落盘根目录默认为$HOME/ascend/atrace/，也可以通过环境变量ASCEND_WORK_PATH指定trace日志落盘路径，例如设置export ASCEND_WORK_PATH=/home/test，具体请参考《环境变量参考》。

trace日志具体落盘路径为：$HOME/ascend/atrace/trace_*{进程组p**id**}_**{首次加载trace动态库的进程pid**}_**{首次加载trace动态库的时间戳**}*/*{event_name}*_event_*{当前进程pid}*_*{目录生成时的时间戳}*/，其中*event_name*为事件类型，取值为：schedule（业务流程异常，如算子执行报错等）、stackcore（进程崩溃或收到异常信号）、exit（进程正常退出析构）。

**表1 **trace日志文件说明

| 存储路径 | 说明 |
| --- | --- |
| schedule_tracer_ts_{device_id}.txt | 当发生AI Core Error、notify wait超时时，Task Schedule回传到host侧的维测信息，包括寄存器、硬件buffer、bitmap等。 |
| stackcore_tracer_{signal}_{tid}_{program_name}_{time}.txt | 当Host业务进程崩溃时记录的轻量级core文件，包括栈帧地址和基地址，该文件需要使用asys工具解析，具体请参见《故障处理》。 |
| schedule_tracer_{object_name}.txt | Runtime、HCCL等模块在运行过程中上报的轨迹信息，记录进程运行过程。 |
| schedule_tracer_{object_name}.bin | AICPU等模块在运行过程中上报的轨迹信息，记录进程运行过程。以二进制格式存储，需要使用asys工具解析，具体请参见《故障处理》。 |

也可以通过环境变量ASCEND_LOG_DEVICE_FLUSH_TIMEOUT配置Device侧日志回传到Host侧的延时时间，具体请参考《环境变量参考》。

> **注意:** 

以上目录是容器或物理机内所有应用程序共同使用的，会不断增加新的应用进程，日志会不断增多，因此需要用户定期清理该目录（可以使用系统自带的logrotate实现日志切分），否则可能导致磁盘空间不足，影响业务正常运行。
