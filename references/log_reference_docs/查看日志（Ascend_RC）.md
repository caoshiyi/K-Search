# 查看日志（Ascend RC）

**页面ID:** logreference_0003  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/logreference/logreference_0003.html

---

本节介绍Ascend RC形态下，日志文件存储路径以及各日志文件记录的主要信息。

> **注意:** 

日志支持老化，如果日志超过配置的存储限制，将会自动删除最早的日志目录或文件。日志目录或文件数量、单个日志文件的大小具体请参见日志配置文件。

**表1 **日志文件介绍

| 存储路径 | 说明 |
| --- | --- |
| /var/log/npu/slog/debug/device-os/device-os_*.log | Control CPU上系统进程产生的调试日志，包括用户态日志和内核态日志。 |
| /var/log/npu/slog/debug/device-app-*pid*/device-app-*pid*_*.log | Control CPU上应用进程产生的调试日志。 |
| /var/log/npu/slog/run/event/event_*.log | Control CPU上系统进程产生的EVENT日志。 |
| /var/log/npu/slog/run/device-os/device-os_*.log | Control CPU上系统进程产生的运行日志。 |
| /var/log/npu/slog/run/device-app-*pid*/device-app-*pid*_*.log | Control CPU上应用进程产生的运行日志。 |
| /var/log/npu/slog/security/device-os/device-os_*.log | Control CPU上系统进程产生的安全日志。 |
| /var/log/npu/slog/security/device-app-*pid*/device-app-*pid*_*.log | Control CPU上应用进程产生的安全日志。 |
| /var/log/npu/slog/debug/device-*id*/device-*id*_*.log | 非Control CPU上的系统类日志，主要采集以下模块的日志：- TS- TSDUMP- LP |
| /var/log/npu/slog/slogd/slogdlog | 维测日志。记录日志工具自身的运行信息，用于日志工具自身问题定位。 日志具备老化策略，当slogdlog文件达到规定大小（1MB）后，名称变更为slogdlog.old进行备份（如果已有备份文件，则删除最早的备份文件）。 |
| 注：上述日志中*id*和*pid*分别代表Device ID和进程ID，请以实际为准；日志文件中的“*”表示该日志文件创建时的时间戳。 |  |
