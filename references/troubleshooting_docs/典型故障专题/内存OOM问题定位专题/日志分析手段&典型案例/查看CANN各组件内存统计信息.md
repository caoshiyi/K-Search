# 查看CANN各组件内存统计信息

**页面ID:** troubleshooting_0055  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0055.html

---

1. **查看并解读日志****关键字：**host侧用户态plog日志中搜索“_svm_mem_stats_show”，搜索结果示例如下所示：

```
run/plog/plog-4176893_20240629161823165.log:460:[INFO] DRV(4176893,python3.7):2024-06-29-16:19:28.436.386 [ascend][curpid: 4176893, 4185695][drv][devmm][_**svm_mem_stats_show** 148]**SVM_MEM** Mem stats (Bytes). (**module_name=RUNTIME**; module_id=7; current_alloced_size=19927040; alloced_peak_size=19927040; alloc_cnt=37; free_cnt=0)
run/plog/plog-4176893_20240629161823165.log:461:[INFO] DRV(4176893,python3.7):2024-06-29-16:19:28.436.393 [ascend][curpid: 4176893, 4185695][drv][devmm][_**svm_mem_stats_show** 148]**DEV_MEM** dev6 Mem stats (Bytes). (**module_name=HCCL**; module_id=3; current_alloced_size=419467264; alloced_peak_size=419500032; alloc_cnt=6; free_cnt=2)
run/plog/plog-4176893_20240629161823165.log:463:[INFO] DRV(4176893,python3.7):2024-06-29-16:19:28.436.401 [ascend][curpid: 4176893, 4185695][drv][devmm][_**svm_mem_stats_show** 148]**DEV_MEM** dev6 Mem stats (Bytes). (**module_name=APP**; module_id=33; current_alloced_size=64680361984; alloced_peak_size=64680361984; alloc_cnt=3113; free_cnt=0)
```

**日志级别：**ERROR级别（申请内存失败时）、进程退出时也会打印内存统计信息

**打印格式：**[内存属性] Mem stats (Bytes). (module_name=[模块名]; module_id=[id]; current_alloced_size=[size]; alloced_peak_size=[size];  alloc_cnt=[size]; free_cnt=[size])

  - **内存属性：**SVM_MEM（支持缺页的地址，CANN内部使用）、DEV_MEM（dev内存）、HOST_MEM（host内存）、DVPP_MEM（dvpp内存）；
  - **module_name：**例如GE、RUNTIME、DVPP等；
  - **current_alloced_size****：**该模块当前已占用的内存大小(Bytes)；
  - **alloced_peak_size****：**该模块已占用的内存峰值大小(Bytes)；
  - **alloc_cnt：**申请内存的次数；
  - **free_cnt：**释放内存的次数。如果释放次数与申请次数不匹配，则需排查内存使用是否合理，例如上层业务框架通过内存池方式管理内存、内存泄漏。

2. **查找哪个组件申请内存增加导致的oom**

  - 如果用户是第一次运行应用，则通过日志中的内存统计信息分析各组件占用的内存是否符合预期或超出硬件物理内存，若不符合预期或超出硬件物理内存，则需要调整代码逻辑或重新规划内存使用。

例如，以参数量13B大模型为例，其中，B是Billion，代表十亿参数，13B就是130亿参数，每个参数全精度是float32，占用32位bit，也就是4Byte字节，1GB=10243Byte，那么13B模型占用13 * 109 * 4Byte ÷ 10243 ≈ 48.4GB，如果说当前硬件内存只有50G左右，那运行模型时大概率会超出硬件物理内存，而导致OOM。

  - 如果非首次运行应用，则可将历史成功的版本与当前失败的版本对比，查看对应内存属性（大多数问题都是申请dev内存不足，查看DEV_MEM属性的打印），各模块的alloced_peak_size峰值内存，找到峰值增加的组件，哪个组件申请的内存与历史成功版本差距大，可重点分析。

历史成功版本的内存统计信息示例（APP申请约2G内存）：

```
[INFO] DRV(4052516,main_aarch64):2024-07-08-22:27:45.354.856 [ascend][curpid: 4052516, 4052516][drv][devmm][_svm_mem_stats_show 148]DEV_MEM dev0 Mem stats (Bytes). (module_name=RUNTIME; module_id=7; current_alloced_size=44138496; alloced_peak_size=44138496; alloc_cnt=18; free_cnt=0)
[INFO] DRV(4052516,main_aarch64):2024-07-08-22:27:45.354.866 [ascend][curpid: 4052516, 4052516][drv][devmm][**_svm_mem_stats_show** 148]DEV_MEM dev0 Mem stats (Bytes). (module_name=**APP**; module_id=33; current_alloced_size=**2078195712**; alloced_peak_size=2078195712; alloc_cnt=996; free_cnt=0)
```

当前问题版本的内存统计信息示例（APP申请约20G内存）：

```
[INFO] DRV(4052522,main_aarch64):2024-07-09-22:47:56.352.884 [ascend][curpid: 4052522, 4052522][drv][devmm][_svm_mem_stats_show 148]DEV_MEM dev0 Mem stats (Bytes). (module_name=RUNTIME; module_id=7; current_alloced_size=44138496; alloced_peak_size=44138496; alloc_cnt=18; free_cnt=0)
[INFO] DRV(4052522,main_aarch64):2024-07-09-22:47:56.352.894 [ascend][curpid: 4052522, 4052522][drv][devmm][**_svm_mem_stats_show** 148]DEV_MEM dev0 Mem stats (Bytes). (module_name=**APP**; module_id=33; current_alloced_size=20781957120; alloced_peak_size=**20781957120**; alloc_cnt=996; free_cnt=0)
```

**注意事项：**

  - 若统计信息处module_name为APP的组件占用内存多，表示用户的应用进程占用内存多，用户需分析已申请内存和预估值的差距，若差距较大，用户需分析原因、排查并优化应用代码中内存申请的逻辑。
  - 若module_name为GE、RUNTIME、HCCL等组件，表示CANN组件占用的内存，在分析CANN各组件内存是否占用过多时，可参考如下内存占用值：

    - **训练场景下，针对不同框架CANN各组件占用的内存不同，以Pytorch框架场景为例**，关键组件占用的内存参考值如下，供分析问题时参考，若CANN组件占用内存较大，则需联系技术支持分析：

      - GE：约3M
      - RUNTIME：约26M
      - HCCL：HCCL占用内存 = 通信链路占用内存 + 缓冲区内存，通信链路占用的内存与集群规模、通信链路有关，缓冲区占用的内存与通信域个数、单个通信域占用的缓冲区大小有关。

例如，集群中有1024个server，需建立10个通信链路，3个通信域，每个通信域占用“2 * HCCL_BUFFSIZE”大小的收发内存（HCCL_BUFFSIZE是环境变量，由用户配置，默认值200M），**则**：单算子模式下通信链路占用的内存 = 通信链路个数 * 4M =10 * 4M = 40M，图模式下通信链路占用的内存 = 通信链路个数 * 图里的算子个数 * 0.3M = 10 * 图里的算子个数 * 0.3M = 图里的算子个数 * 3M，缓冲区内存 = 通信域个数 * 每个通信域占用的收发内存 =3 * 2 * 200M

    - **推理场景下，****以Pytorch模型为例，转换为适配昇腾AI处理器的离线模型进行推理时**，关键组件占用的内存参考值如下，供分析问题时参考，若CANN组件占用内存较大，则需联系技术支持分析：

      - GE：约86M
      - RUNTIME：约18M

  - 可使用《算子开发工具用户指南》中的msSanitizer内存检测工具排查用户应用的内存问题，不过该工具当前仅支持Atlas A2训练系列产品/Atlas 800I A2推理产品、Atlas 推理系列产品。
