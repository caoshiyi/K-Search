# 调用SetDevice接口错误导致无可用Context

**页面ID:** troubleshooting_0065  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0065.html

---

#### 问题现象

参见收集进程中断问题信息收集日志文件，收集的日志所存放的目录，下文以${HOME}/err_log_info/为例。

Host应用类日志（${HOME}/err_log_info/log/[run|debug]/plog/plog-*pid*_*.log）日志中存在“ctx is NULL” 、“context pointer null”关键词，日志示例如下：

```
104069:[ERROR]RUNTIME(2977549,test_incre):2024-02-21-10:24:27.965.879[api_impl.cc:5544]2978321 CtxGetSysParamOpt:report error module_type=3,
module_name=EE8888
104070:[ERROR]RUNTIME(2977549,test_incre):2024-02-21-10:24:27.965.886[api_impl.cc:5544]2978321 CtxGetSysParamOpt:ctx is null!
104072:[ERROR]RUNTIME(2977549,test_incre):2024-02-21-10:24:27.966.091[api_c.cc:5200]2978321 **rtCtxGetSysParamOpt:ErrCode=107002, desc=[context pointer null]**,InnerCode=0x7070001
104074:[ERROR]RUNTIME(2977549,test_incre):2024-02-21-10:24:27.966.116[error_message_manage.cc:48]2978321 FuncErrorReason:rtCtxGetSysParamOpt execute
failed, reason=[context pointer null]
104080:[ERROR]RUNTIME(2977549,test_incre):2024-02-21-10:24:27.966.770[api_impl.cc:5553]2978321 CtxGetOverflowAddr:report error module_type=3,
module_name=EE8888
```

#### 故障根因

SetDevice操作相关的接口调用异常，导致无可用Context。

acl对应接口：aclrtSetDevice

runtime对应接口为：rtSetDevice/rtSetDeviceEx

#### 解决方法

执行**export ASCEND_GLOBAL_LOG_LEVEL=1**命令将日志级别设置为info后，重新执行业务、收集日志后，在plog日志中搜索关键字“SetDevice”，排查是否存在如下错误的SetDevice调用方式，并根据排查场景优化代码逻辑。

- 如果能搜索到“SetDevice”，需排查以下代码逻辑：

  - 是否运行环境上Device异常，导致无法创建Context；

**错误场景举例**：做了SetDevice，但是运行环境上没有Device设备。

```
[root@localhost plog]# **grep -rn "SetDevice"**
plog-27441_20240221060831113.log:2210:[INFO] ASCENDCL(28980,python3):2024-02-21-06:08:31.254.432 [device.cpp:148]28980 aclrtSetDevice: start to execute aclrtSetDevice, deviceId = 0
plog-27441_20240221060831113.log:2231:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:31.254.750 [api_c.cc:1798] 28980 **rtSetDevice: There is no devId, do nothing**.
plog-27441_20240221060831113.log:2235:[WARNING] ASCENDCL(28980,python3):2024-02-21-06:08:31.254.818 [device.cpp:157]28980 aclrtSetDevice: update platform info with device failed, deviceId = 0
plog-27441_20240221060831113.log:2236:[INFO] ASCENDCL(28980,python3):2024-02-21-06:08:31.254.828 [device.cpp:160]28980 aclrtSetDevice: successfully execute aclrtSetDevice, deviceId = 0
plog-27441_20240221060831113.log:6861:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:31.529.819 [api_c.cc:1798] 29157 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:11191:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:40.390.084 [api_c.cc:1798] 29157 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:36695:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:58.721.291 [api_c.cc:1798] 28980 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:36738:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:58.729.533 [api_c.cc:1798] 28980 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:36781:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:58.736.987 [api_c.cc:1798] 28980 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:36796:[INFO] RUNTIME(28980,python3):2024-02-21-06:08:58.754.905 [api_c.cc:1798] 28980 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:561:[INFO] RUNTIME(30104,host_cpu_executor):2024-02-21-06:08:41.913.634 [api_c.cc:1798] 30104 rtSetDevice: There is no devId, do nothing.
plog-27441_20240221060831113.log:9522:[INFO] HCCL(30104,host_cpu_executor):2024-02-21-06:08:55.769.345 [hccl_impl_base.cc:2644] [30104][SetDevice] entry
plog-27441_20240221060831113.log:9526:[INFO] HCCL(30104,host_cpu_executor):2024-02-21-06:08:55.769.494 [hccl_impl_base.cc:2622] [31143][SetDeviceThread]ctx[(nil)]
plog-27441_20240221060831113.log:9527:[INFO] HCCL(30104,host_cpu_executor):2024-02-21-06:08:55.769.497 [hccl_impl_base.cc:2637] [31143][SetDeviceThread]exit
plog-27441_20240221060831113.log:9528:[INFO] HCCL(30104,host_cpu_executor):2024-02-21-06:08:55.769.527 [hccl_impl_base.cc:2657] [30104][SetDevice]exit
```

```
[root@localhost plog]# **grep -rn "ERROR"**
plog-27441_20240221060831113.log:36899:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.527) [api_impl.cc:1038]28980 GetMaxStreamAndTask:[FINAL][FINAL]report error module_type=3, module_name=EE8888
plog-27441_20240221060831113.log:36900:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.531) [api_impl.cc:1038]28980 GetMaxStreamAndTask:[FINAL][FINAL]**ctx is NULL!**
plog-27441_20240221060831113.log:36902:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.625) [logger.cc:417]28980 GetMaxStreamAndTask:[FINAL][FINAL]GetMax stream and task failed, streamType=0.
plog-27441_20240221060831113.log:36903:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.668) [api_c.cc:850]28980 rtGetMaxStreamAndTask:[FINAL][FINAL]ErrCode=107002, desc[contextpointer null], InnerCode=0x7070001
plog-27441_20240221060831113.log:36904:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.675) [error_message_manage.cc:48]28980 FuncErrorReason:[FINAL][FINAL]report error module_name=EE1001
plog-27441_20240221060831113.log:36905:[ERROR] RUNTIME(28980,python3:2024-02-21-06:08:59.095.683) [error_message_manage.cc:48]28980 FuncErrorReason:[FINAL][FINAL]rtGetMaxStreanAndTask execute failed, reason=[context pointer null]
```

  - 多线程场景下，排查一下SetDevice操作的线程和报错的线程是否一致，是否只对主线程（线程ID：2977549）做了SetDevice、子线程（线程ID：2978321）没有做SetDevice，导致子线程中无可用Context；

**错误场景举例**：只对主线程做了SetDevice，其它线程没有做SetDevice，导致报错。

```
[root@localhost plog]# grep -rn "ERROR"
104069:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.965.879  [api_impl.cc:5544]2978321
CtxGetSysParamOpt:report error module_type=3, module_name=EE8888
104070:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.965.886 [api_impl.cc:5544]2978321 CtxGetSysParamOpt:**ctx is NULL!**
104072:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.091 [api_c.cc:5200]2978321 rtCtxGetSysParamOpt:ErrCode=107002, desc[context pointer null], InnerCode=0x7070001
104073:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.104 [error_message_manage.cc:48]2978321 FuncErrorReason:report error module_name=EE1001
104074:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.116 [error_message_manage.cc:48]2978321 FuncErrorReason:rtCtxGetSysParamOpt execute failed, reason=[context pointer null]
104080:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.770  [api_impl.cc:5553]2978321
CtxGetOverflowAddr:report error module_type=3, module_name=EE8888
104081:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.776  [api_impl.cc:5553]2978321 CtxGetOverflowAddr:ctx is NULL!
104083:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.832 [api_c.cc:5210]2978321 rtCtxGetOverflowAddr:ErrCode=107002, desc[context pointer null], InnerCode=0x7070001
104084:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.840 [error_message_manage.cc:48]2978321 FuncErrorReason:report error module_name=EE1001
104085:[ERROR] RUNTIME(2977549, test_incre):2024-03-21-10:24:27.966.847 [error_message_manage.cc:48]2978321 FuncErrorReason:rtCtxGetOverflowAddr execute failed, reason=[context pointer null]
104087:[ERROR] OP(2977549, test_incre):2024-03-21-10:24:27.966.882 [nnopbase_executor.cpp:73][NNOP][NnopbaseSetOverFlowAddr][2978321] errno[361001] Assert ((rtCtxGetOverflowAddr(&addr)) == 0) failed
104089:[ERROR] OP(2977549, test_incre):2024-03-21-10:24:27.968.636 [nnopbase_executor.cpp:216][NNOP][NnopbaseSetExecutorSetGlobalConfig][2978321]
errno[361001] Check NnopbaseSetOverFlowAddr(g_nnopbaseSysCfgParams.overflowAddr) failed
104091:[ERROR] OP(2977549, test_incre):2024-03-21-10:24:27.968.800 [nnopbase_api.cpp:44][NNOP][NnopbaseInit][2978321] errno[361001] Check
NnopbaseExecutorSetGlobalConfig() failed
104093:[ERROR] OP(2977549, test_incre):2024-03-21-10:24:27.968.822 [nnopbase_api.cpp:53][NNOP][NnopbaseCreateExecutorSpace][2978321]  errno[361001] Assert ((NnopbaseInit()) == 0) failed
104095:[ERROR] OP(2977549, test_incre):2024-03-21-10:24:27.968.841 [nnopbase_api.cpp:19][NNOP][NnopbaseOpLogE][2978321] errno[361001] Check
NnopbaseCreateExecutorSpace(&executorSpace) failed
```

  - 是否在报错后才调用SetDevice，导致无可用Context；

**错误场景举例**：申请内存前，没有调用SetDevice相关接口，导致报错。

```
[ERROR] RUNTIME(44525,python) :2024-01-22-04:26:23.625.738 [api_impl.cc:1401]45179 DevMalloc: [DUMP] [DEFAULT] report error module_type=3, module_name=EE8888
[ERROR] RUNTIME(44525,python) :2024-01-22-04:26:23.625.743 [api_impl.cc:1401]45179 DevMalloc: [DUMP] [DEFAULT] **ctx is NULL!**
[INFO] GE(44525,python) :2024-01-22-04:26:23.625.761 [error_manager.cc:296]45179 ReportInterErrMessage:report error message, error_code:EE8888, work_stream_id:4452545179
[ERROR] RUNTIME(44525,python) :2024-01-22-04:26:23.625.784 [logger.cc:581]45179 DevMalloc:[DUMP][DEFAULT]Device malloc failed, size=9(Byte), type=2.
[ERROR] RUNTIME(44525,python) :2024-01-22-04:26:23.625.805 [api_c.cc:1173]45179 rtMalloc:[DUMP][DEFAULT]ErrCode=107002, desc=[**context pointer null**], InnerCode=0x7070001
[ERROR] RUNTIME(44525,python) :2024-01-22-04:26:23.625.811 [error_message_manage.cc:48]45179 FuncErrorReason:[DUMP][DEFAULT] report error module_name=EE1001
[ERROR]RUNTIME(44525,python) :2024-01-22-04:26:23.625.816 [error_message_manage.cc:48]45179 FuncErrorReason:[DUMP][DEFAULT] **rtMallocexecute failed, reason=[context pointer null]**
```

- 如果搜索不到“SetDevice”，说明没有做SetDevice操作，导致无可用Context。
