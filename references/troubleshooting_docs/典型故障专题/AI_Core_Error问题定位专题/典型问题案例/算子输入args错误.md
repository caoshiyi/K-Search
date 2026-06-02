# 算子输入args错误

**页面ID:** troubleshooting_0019  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0019.html

---

#### 问题现象

Host应用类日志（log/[run|debug]/plog/plog-*pid*_*.log）中存在如下报错：

```
[**ERROR**] RUNTIME(85483,python):2024-04-19-18:13:17.186.318 [task_info.cc:1678]85658 PrintErrorInfoForDavinciTask:Aicore kernel execute failed, device_id=0, stream_id=2, report_stream_id=2, task_id=57783, flip_num=3, fault kernel_name=RealDiv_ee98c6628030785f610b924ab1557b31_high_precision_210000000, fault kernel info ext=none, program id=9, hash=10612039229658031084.
[**ERROR**] RUNTIME(85483,python):2024-04-19-18:13:17.186.336 [task_info.cc:1617]85658 GetArgsInfo:[AIC_INFO] **args**(0 to 9) **after execute**:**0x4f453840**, 0x124201ea7400, 0x12420240cc00, 0x1241c006dc28, 0x124100011000, 0x1, 0x1, 0x1, 0,
```

#### 故障根因

在plog日志的报错信息中，args(*xxxx*) after execute部分的日志很关键，需检查args after execute处的参数地址是否合理，如果出现0，可以认定是地址分配错误导致。

在Atlas 训练系列产品上，地址是0x1240开头，如果出现不是0x1240开头的地址则可能异常，需排查。以上报错日志中，在Atlas 训练系列产品上，args after execute处0x4f453840、0x124201ea7400、0x12420240cc00、0x1241c006dc28等信息，其中第一个参数**0x4f453840**，非0x1240开头，怀疑该参数地址是Host内存地址而不是Device内存地址，导致AI Core读取错误。

在Atlas A2 训练系列产品/Atlas A2 推理系列产品上，地址不是0x1240开头，如果出现0x1240开头的地址则可能异常，需排查。

#### 处理方法

需排查训练脚本，对于使用除法的地方，例如使用了cpu_tensor/npu_tensor，由于cpu_tensor是Host内存地址，而npu_tensor又是Device内存地址，这会导致AI Core读取错误，应将两张都修改为Device内存地址。
