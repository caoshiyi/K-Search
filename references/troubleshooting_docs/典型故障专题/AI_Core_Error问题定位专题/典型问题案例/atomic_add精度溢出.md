# atomic add精度溢出

**页面ID:** troubleshooting_0015  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0015.html

---

#### 分析结果

如果info.txt中给出如下结论，说明是精度溢出导致的AI Core error。

```
Analysis result: success.
"**********************Root cause conclusion******************"
Atomic add has a precision overflow. Check the operator precision.Note that if tasks are concurrently executed on the NPU, a false warning may be reported.
```

在Device的slog日志（report/*/slog/dev-os-*id*/[run|debug]/device-os/device-os_*.log）中，检查是否出现"**Vm fault failed**"关键字，如果没有出现，则确定是atomic精度溢出问题导致的AI Core Error问题；如果出现，则说明是内存越界问题，不是atomic精度溢出。slog日志示例如下：

```
2024-02-20-17-07-25/slog/dev-os-0/debug/device-os/device-os_20240122045443091.log:461:[EVENT] KERNEL(4128,sklogd):2024-01-22-05:03:02.259.649 [klogd.c:253][2572550.901383] [ascend] [ERROR] [devmm] [devmm_svm_device_fault 438] <kworker/u16:186:9871,9871> Vm fault failed. (hostpid=1885445; devid=0; vfid=0; ret=64; fault_addr=0x1240f1fa0000; start=0x1240f1fa0000)
```

Atlas A2 训练系列产品/Atlas A2 推理系列产品上，由于硬件优化，不会出现atomic add精度溢出问题。

#### 故障根因

出现该问题，是由于在算子运算过程中，有极端的数据遇到了atomic累加指令，atomic累加时如果出现溢出，则会报0x800000错误。

#### 处理方法

此类问题一般为输入数据错误导致，应通过精度调优手段进行下一步定位。

**推理**场景下，请参考《应用开发指南 (C&C++)》中的“精度/性能优化”章节调优精度。

**训练**场景下，对于TensorFlow框架，请参考《TensorFlow 2.6.5模型迁移指南》的“精度调优”章节调优精度。

**训练**场景下，对于PyTorch框架，请参考《PyTorch 训练模型迁移调优指南》调优精度。
