# AI Core硬件故障

**页面ID:** troubleshooting_0011  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0011.html

---

#### 问题现象

Host应用类日志（log/[run|debug]/plog/plog-*pid*_*.log）中存在如下报错：

```
plog/plog-377228_20240402154414713.log:70:[ERROR] RUNTIME(377228,ascend-dmi):2024-04-02-15:49:01.883.834 [device_error_proc.cc:1164] 377265 ProcessStarsCoreErrorInfo:The error from device(**chipId**:2, dield:0), serial number is 4, there is an fftsplus aivector error exception, core id is 34, error code = 0, dump info: pc start: 0x12406ce78f54, current: 0x12406ce7C430, vec error info: 0xef1d17c368, mte error info: 0xf3dff1cf9b, ifu error info: 0x63f70dc00e200, ccu error info : 0x6f6d518f0004d700, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd000288, para base: 0x1240403e5000． 
plog/plog-377228_20240402154414713.log:71:[ERROR] RUNTIME(377228,ascend-dmi):2024-04-02-15:49:01.883.875 [device_error_proc.cc:1176] 377265 ProcessStarsCoreErrorInfo: report error module_type=5, module_name=EZ9999
plog/plog-377228_20240402154414713.log:72:[ERROR] RUNTIME(377228,ascend-dmi):2024-04-02-15:49:01.883.876 [device_error_proc.cc:1176] 377265 ProcessStarsCoreErrorInfo: The extend info: **errcode:(0, 0x80000000, 0)** **errorStr: A 2-bit ECC error occurs in the data-cache data-ram**. fixp_error0 info: 0xff1cf9b, fixp_error1 info: 0xf3 fsmId:0, tslot:0, ctxid:0, blk:16, subblk:0, subErrType:4.
```

#### 故障根因

使用ascend-dmi工具压测AI Core，压测异常，提示**EMERGENCY_WARN**，表示AI Core故障。

ascend-dmi工具需要单独安装，压测AI Core的命令示例如下：

```
ascend-dmi --dg -i aicore
```

ascend-dmi工具在MindCluster ToolBox软件包中，该软件与CANN的配套关系请单击Link查询，ascend-dmi工具的安装及详细使用指导请参见[Link](https://hiascend.com/document/redirect/mindxdl-ascenddmiug)。

#### 处理方法

AI Core故障，需联系技术支持更换硬件。您可以获取日志后单击Link联系技术支持。
