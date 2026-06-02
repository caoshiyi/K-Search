# 查看Device业务进程内存统计信息

**页面ID:** troubleshooting_0056  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0056.html

---

1. **查看并解读日志**关键字：host侧用户态plog日志中搜索“svm_mem_stats_show_device_proc_mem”，搜索结果示例如下所示：

```
run/plog/plog-4176893_20240629161823165.log:474:[INFO] DRV(4176893,python3.7):2024-06-29-16:19:29.085.761 [ascend][curpid: 4176893, 4185695][drv][devmm][**svm_mem_stats_show****_device_proc_mem** 358]**DEV_PROC_MEM** dev6 Mem stats (Bytes). (**module_name=AICPU**; module_id=36; **total_size=75464704**)
run/plog/plog-4176893_20240629161823165.log:475:[INFO] DRV(4176893,python3.7):2024-06-29-16:19:29.085.765 [ascend][curpid: 4176893, 4185695][drv][devmm][**svm_mem_stats_show**_**device_proc_mem** 358]**DEV_PROC_MEM** dev6 Mem stats (Bytes). (**module_name=CUSTOM**; module_id=76; **total_size=75464704**)
```

**DEV_PROC_MEM**表示Device侧进程，**module_name**用于区分Device侧的不同进程（例如AI CPU进程、自定义算子CUSTOM进程、HCCP进程等），**total_size**表示不同进程占用物理内存的情况**，**包括这些进程本身malloc占用的常驻物理内存，还包括这些进程调buff申请接口申请的sharepool类型内存。

2. **查找哪个Device业务进程申请内存增加导致的oom**

  - 如果用户是第一次运行应用，则通过日志中的内存统计信息分析各进程占用的内存是否符合预期或超出硬件物理内存，若不符合预期或超出硬件物理内存，则需联系技术支持进一步定位。
  - 如果非首次运行应用，则可将历史成功的版本与当前失败的版本对比，查看哪个业务进程申请的内存与历史成功版本差距大，可重点分析。

历史成功版本的内存统计信息示例（AI CPU进程申请约72M内存）：

```
[INFO] DRV(4052516,main_aarch64):2024-07-08-22:27:45.334.457 [ascend][curpid: 4052516, 4052516][drv][devmm][**svm_mem_stats_show_device_proc_mem** 358]DEV_PROC_MEM dev0 Mem stats (Bytes). (module_name=**AICPU**; module_id=36; total_size=**75505664**)
[INFO] DRV(4052516,main_aarch64):2024-07-08-22:27:45.334.461 [ascend][curpid: 4052516, 4052516][drv][devmm][svm_mem_stats_show_device_proc_mem 358]DEV_PROC_MEM dev0 Mem stats (Bytes). (module_name=CUSTOM; module_id=76; total_size=75505664)
```

当前问题版本的内存统计信息示例（AI CPU进程申请约150M内存）：

```
[INFO] DRV(4052533,main_aarch64):2024-07-08-22:47:49.335.448 [ascend][curpid: 4052533, 4052533][drv][devmm][**svm_mem_stats_show_device_proc_mem** 358]DEV_PROC_MEM dev0 Mem stats (Bytes). (module_name=**AICPU**; module_id=36; total_size=**157286400**)
[INFO] DRV(4052533,main_aarch64):2024-07-08-22:47:49.335.458 [ascend][curpid: 4052533, 4052533][drv][devmm][svm_mem_stats_show_device_proc_mem 358]DEV_PROC_MEM dev0 Mem stats (Bytes). (module_name=CUSTOM; module_id=76; total_size=75505664)
```
