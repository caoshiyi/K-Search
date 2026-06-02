# 使用msnpureport工具收集更多AI Core Error信息

**页面ID:** troubleshooting_0021  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0021.html

---

联系技术支持定位一些疑难AI Core Error问题时，除了收集AI Core Error问题信息中收集的信息外，还需获取Device配置信息、设置TaskSchedule是否自动复位加速器、导出寄存器信息、设置AI Core上任务串联或并行执行、屏蔽指定AI Core或Vector Core上的任务执行等，以便更进一步排查算子问题、硬件问题。

本节仅给出收集更多AI Core Error信息的命令示例，详细使用方法及约束请参见《[msnpureport工具使用指南](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)》。

- 获取当前Device配置信息。

命令示例如下，表示获取Device 0的配置信息：

```
**msnpureport config --get --device** 0
```

- 设置TaskSchedule是否自动复位加速器，以便导出更详细、更精准的寄存器信息定位问题，但该配置后会影响执行性能。

命令示例如下，表示在Device 1上设置不自动复位加速器：

```
**msnpureport config --set** **--device** 1 **--accelerator_recover** 0
```

- 设置AI Core的singlecommit，设置后，需再次收集故障信息、分析问题。

命令示例如下，表示在Device 1上关闭AI Core singlecommit模式，此时AI Core内部多指令并行：

```
**msnpureport config --set --device** 1 **--singlecommit** 0
```

- 屏蔽指定AI Core或Vector Core上的任务执行，以便排查哪个AI Core或Vector Core故障。设置后，需再次收集故障信息、分析问题。

命令示例如下：

```
# 在Device 0上，屏蔽core id为3的AI Core
**msnpureport config --set --aic_switch** 0 **--coreid** 3 **--device** 0

# 在Device 0上，屏蔽core id为5的Vector Core
**msnpureport config --set --aiv_switch** 0 **--coreid** 5 **--device** 0
```

- 设置icache bit翻转检验范围，以便定位算子问题。设置后，需再次收集故障信息、分析问题。

在设置该参数值前，可先参见•获取当前Device配置信息。获取当前环境上的icache翻转校验范围（即Icache check range字段值）。

设置icache bit翻转检验范围的命令示例如下，表示在Device 0上将--icachecheck设置为128，这时校验从出错PC往前128K到出错PC往后128K的icache与GM（Global Memory）是否一致：

```
**msnpureport config --set --icachecheck** 128 **--device** 0
```

- 导出寄存器信息，便于后续辅助定位硬件问题。

命令示例如下：

```
**msnpureport report --type 2**
```

> **注意:** 

- 基于昇腾AI处理器的AI应用进程运行过程中不支持使用本步骤中的定位方法，可能会导致应用进程运行异常或本步骤中的命令执行异常，需应用进程退出后才能使用。
