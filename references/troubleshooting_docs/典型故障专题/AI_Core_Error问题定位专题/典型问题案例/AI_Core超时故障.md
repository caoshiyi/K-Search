# AI Core超时故障

**页面ID:** troubleshooting_0500  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0500.html

---

#### 问题现象

Host应用类日志（log/[run|debug]/plog/plog-*pid*_*.log）中存在如下报错：

```
[ERROR] RUNTIME(3813,python3):2025-01-10-17:53:52.846.403 [device_error_proc.cc:1409] 3813 ProcessStarsCoreErrorInfo:[INIT][DEFAULT]report error module_type=5, module_name=EZ9999
[ERROR] RUNTIME(3813,python3):2025-01-10-17:53:52.846.427 [device_error_proc.cc:1409] 3813 ProcessStarsCoreErrorInfo:[INIT][DEFAULT]The error from device(**chipId**:4, dield:0), serial number is 67, there is an aivector error exception, core id is 18, error code = 0, dump info: pc start: 0x124800000000, current: 0x12480000008c, vec error info: 0x5a1e95250c, mte error info: 0xf3ff16bb18, ifu error info: 0xcd34130e0080, ccu error info : 0x8405bf692593f308, cube error info: 0, biu error info: 0, aic error mask: 0x6500020bd00028c, para base: 0x12c1c001c00． 
[ERROR] RUNTIME(3813,python3):2025-01-10-17:53:52.846.526 [device_error_proc.cc:1421] 3813 ProcessStarsCoreErrorInfo: report error module_type=5, module_name=EZ9999
[ERROR] RUNTIME(3813,python3):2025-01-10-17:53:52.846.528 [device_error_proc.cc:1421] 3813 ProcessStarsCoreErrorInfo: The extend info: **errcode:(0, 0, 0)** **errorStr: timeout or trap error**. fixp_error0 info: 0xf16bb18, fixp_error1 info: 0xf3 fsmId:0, tslot:5, thread:0, ctxid:0, blk:11, subblk:0, subErrType:4.
```

Device的event日志（slog/dev-os-*id*/run/event/event_*.log）中存在“**event_id=0x80C98001**”或“**event_id=0x80CB8001**”关键字。

#### 故障根因

使用ascend-dmi工具压测AI Core，压测异常，提示**a timeout error occurred**，表示AI Core超时。报错示例如下所示：

```
Hardware:
    aicore:
        FAIL
        *** Some processes are seizing the NPU. Test results may be affected.
        *** Device 4: a timeout error occurred.
```

ascend-dmi工具需要单独安装，压测AI Core的命令示例如下：

```
ascend-dmi --dg -i aicore -s -q
```

ascend-dmi工具在MindCluster ToolBox软件包中，该软件与CANN的配套关系请单击Link查询，ascend-dmi工具的安装及详细使用指导请参见[Link](https://hiascend.com/document/redirect/mindxdl-ascenddmiug)。

#### 处理方法

建议下电重启后再使用ascend-dmi工具压测，如果对应设备依旧出现timeout报错，则判断为硬件故障，需联系技术支持更换硬件。

您可以获取日志后单击Link联系技术支持。
