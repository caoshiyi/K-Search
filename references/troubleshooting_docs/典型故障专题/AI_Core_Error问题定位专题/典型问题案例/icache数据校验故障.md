# icache数据校验故障

**页面ID:** troubleshooting_0010  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0010.html

---

#### 问题现象

Device的event日志（slog/dev-os-*id*/run/event/event_*.log）中存在“**event_id=0x80C98000**”关键字。

Atlas A2 训练系列产品/Atlas A2 推理系列产品上，trace日志（默认在$HOME/ascend/atrace/路径下）中存在“stars_print_error_pc_icache_and_hbm_info”报错关键字。

```
2024-04-22-09-06-17/hisi_logs/device-2/20240422090623-533810000/log/ts.log:5177:[ERROR] TSCH(-1,null):2024-04-20-17:02:52.772.875 35906 (dieid:0,cpuid:0) aicore.c:767 **stars_print_error_pc_icache_and_hbm_info**: stat for dump pc start, aiv_id=47, icache_miss_num=8161, hbm_miss_num=0, compare_num=32, **compare_fail_num**=0
```

Atlas 推理系列产品上，trace日志（默认在$HOME/ascend/atrace/路径下）中存在“check_error_pc_icache_and_hbm_info”报错关键字。

```
[ERROR] TSCH(-1,null):2024-09-04-00:12:52.986.322 438 (dieid:0,cpuid:0) aicore_icache_plat.c:848 **check_error_pc_icache_and_hbm_info**: stat for dump pc start, aic_id=1, icache_miss_num=8176, hbm_miss_num=0, compare_num=17, **compare_fail_num**=0
```

#### 故障根因

在报错关键字处，查看compare_fail_num打印信息，若compare_fail_num值不等于0，则说明存在icache内存跳变硬件故障。

#### 处理方法

单击《[健康管理故障定义](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743)》获取对应版本的手册，icache内存跳变故障有如下说明（列举部分关键字段）：

| Event ID | 0x80C98000 |
| --- | --- |
| 故障事件名称 | AICORE指令数据校验失败。 |
| 故障解释/可能原因 | icache数据与GM校验不一致。可能原因包括： 1. icache数据跳变。2. GM数据被改写。 |
| 故障影响 | 当前AI任务失败，如果该AI Core没有恢复正常，则后续AI任务也失败。 |
| 故障自处理模式 | 1. TSFW通过TSDrv上报故障至故障管理。2. TSFW记录错误日志。3. TSFW通过业务面返回任务失败。4. TSFW复位AIC，若复位成功则通过TSDrv上报恢复；若复位失败则进行拔核处理（隔离核后，业务调度不再使用该核）并记录错误日志。 |
| 系统处理建议 | 1. 建议AI训练任务退出重新执行或重新发起推理请求。2. 若AI任务重新执行异常则建议复位SOC，如果故障持续建议返厂送修。 |
