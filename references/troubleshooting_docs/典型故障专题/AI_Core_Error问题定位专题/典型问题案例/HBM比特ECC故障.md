# HBM比特ECC故障

**页面ID:** troubleshooting_0009  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0009.html

---

#### 问题现象

Device的event日志（slog/dev-os-*id*/run/event/event_*.log）中存在“**event_id=0x80E01809**”或“**event_id=0x80E01801**”关键字。

Device黑匣子日志（在hisi_logs/device-*id*/*/bbox目录中）中存在“Hardware Error”报错关键字：

```
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2014[2454.830692] {6}[Hardware Error] Hardware error from APEI Generic Hardware Error Source: 0
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2015[2454.830693] {6}[Hardware Error]event severity: recoverable
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2016[2454.830694] {6}[Hardware Error] Error 0, type: recoverable
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2017[2454.830696] {6}[**Hardware Error**]  section_type: **memory error**
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2018[2454.830697] {6}[Hardware Error]  physical_address: 0x0000101efe36d3c0
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2019[2454.830699] {6}[Hardware Error]  node: 2 card: 259 module: 51 rank: 1 bank: 10 row: 30691  column: 56
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2020[2454.830701] {6}[Hardware Error]  **error_type: 3, multi-bit ECC**
/hisi_logs/device-2/20240420162417-271713000/bbox/kbox.txt:2024[2454.830702] {6}[Hardware Error]  DIMM location: not present. DMI handle: 0x0000
```

#### 故障根因

通过上述日志中的报错判断是HBM比特ECC故障导致AI Core Error。

#### 处理方法

单击《[健康管理故障定义](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743)》获取对应版本的手册，HBM比特ECC故障有如下说明（列举部分关键字段）：

| Event ID | 0x80E01809 |
| --- | --- |
| 故障事件名称 | HBM内存颗粒巡检多bit ECC错误。 |
| 故障解释/可能原因 | HBMC（HBM Control）巡检（Patrol Scrubbing和Demand Scrubbing）触发的多bit ECC错误，可能原因为HBM颗粒部分失效、HBM无法正常保持数据等，通常是硬件问题。 |
| 故障影响 | 1. 启动过程中访问到错误地址，可能会启动失败。 2. 业务访问到错误地址会返回错误数据，可能导致业务失败。 3. 业务未访问到错误地址，不会影响当前业务。 |
| 故障自处理模式 | 1.上报通知事件到设备管理，记录错误地址，尝试在线隔离，重启后进行离线隔离。 2.记录错误日志。 |
| 系统处理建议 | 无需操作。 |

| Event ID | 0x80E01801 |
| --- | --- |
| 故障事件名称 | HBM用户内存空间内存颗粒访问多bit ECC错误。 |
| 故障解释/可能原因 | HBM内存空间被访问，触发多bit ECC错误且地址热隔离失败，可能原因为HBM颗粒部分失效且失效地址仍被占用，通常是硬件问题。 |
| 故障影响 | 访问HBM返回错误数据，可能导致NPU启动失败或者业务失败。 |
| 故障自处理模式 | 1.上报故障事件到设备管理。 2.记录错误日志。 |
| 系统处理建议 | 如上报Host PID为零，建议复位SOC；如上报Host PID非零，kill进程，等待一个恢复时长后，如故障仍不恢复，建议复位SOC。 |
