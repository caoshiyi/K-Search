# AI Core Error问题现象描述

**页面ID:** troubleshooting_0004  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0004.html

---

用户应用程序报错退出，终端屏幕日志错误码为EZ9999，且日志中包含“**there is an aivec error exception**”或“**there is an aicore error exception**”；或者plog日志中存在报错日志“**Aicore kernel execute failed**”。

**报错示例如下所示**：

```
-----------------------------------------
   Ascend Error Message:
-----------------------------------------
EZ9999: Inner Error!
EZ9999: The error from device(**chipid**:4, **dieId**:0), serial number is 2, **there is an aivec error exception**, **core id** is 11, error code = 0x10, dump info: pc start: 0x1240c46650b8, vec error info: 0xd019ddc1a, mte error info: 0x2ffeba07af, ifu error info: 0x4e5c097530000, ccu error info: 0x30c255954a000023, cude error info: 0,0, aic error mask: 0x65000020bd000288, para base: 0x1240c51b1dd0.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1100]
        TraceBack (most recent call last):
        The extend info: **errcode**:(0x10, 0, 0) **errorStr**: Illegal instruction, which is usually caused by unaligned UUB addresses, fixp_error0 info: 0xeba07af, fixp_error, mId:0, tslot:0, threadId:0, ctxid:0, blk:0, sublk:0, subErrType:4.[FUNC:ProcessStarsCoreErrorInfo][FILE:device_error_proc.cc][LINE:1112]
       ** Aicore kernel execute failed**, device_id=4, stream_id=450, report_stream_id=2, task_id=442, flip_num=0, **fault kernel_name**=00_131_Grandients/Default/AddN.op56419/program id=2089, hash=16296079633597215637.[FUNC:GetError][FILE:stream.cc][LINE:1467]
        [AIC_INFO] after execute:args print end[FUNC:GetError][FILE:stream.cc][LINE:1467]
        rtStreamSynchronize execute failed, reason=[The model stream execute failed][FUNC:FuncErrorReason][FILE:error_message_manage.cc][LINE:50]
(Please search "Ascend Error Message" at https://www.mindspore.cn for error code description)
-----------------------------------------
   C++ Call Stack: (For framework developers)
-----------------------------------------
```

**报错日志解读：**

- **chipId、****dieId**：分别表示报错芯片chipId及dieid，可用于判定是否固定chipId报错；
- **core id：**表示报错芯片核id，可用于判定是否在同一个core上执行报错；
- **errcode**、**errorStr**：分别表示AI Core Error的报错错误码、错误描述；
- **fault kernel_name/fault kernel info ext****：**表示报错kernel名字，可用于查看报错算子。

异步任务执行场景下，例如连续下发多个算子执行任务，可能会有多个算子报错，错误信息中可能包含多个算子错误，用户需从首报错算子开始排查问题。
