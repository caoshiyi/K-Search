# troubleshooting 文档索引

## 典型故障专题

### AI Core Error问题定位专题

#### 典型问题案例

- [AI Core硬件故障](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/AI_Core硬件故障.md)
- [AI Core超时故障](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/AI_Core超时故障.md)
- [Dump数据失败](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/Dump数据失败.md)
- [HBM比特ECC故障](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/HBM比特ECC故障.md)
- [atomic add精度溢出](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/atomic_add精度溢出.md)
- [icache数据校验故障](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/icache数据校验故障.md)
- [单算子运行报错](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/单算子运行报错.md)
- [算子输入args下发前后不一致](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/算子输入args下发前后不一致.md)
- [算子输入args错误](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/算子输入args错误.md)
- [算子输入输出数据地址异常](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/算子输入输出数据地址异常.md)
- [系统环境/硬件问题](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/系统环境_硬件问题.md)
- [索引类算子索引越界](典型故障专题/AI_Core_Error问题定位专题/典型问题案例/索引类算子索引越界.md)

#### 参考信息

- [使用msnpureport工具收集更多AI Core Error信息](典型故障专题/AI_Core_Error问题定位专题/参考信息/使用msnpureport工具收集更多AI_Core_Error信息.md)
- [手动收集算子编译信息（算子.o和.json文件）](典型故障专题/AI_Core_Error问题定位专题/参考信息/手动收集算子编译信息（算子.o和.json文件）.md)

- [AI Core Error问题定位思路](典型故障专题/AI_Core_Error问题定位专题/AI_Core_Error问题定位思路.md)
- [AI Core Error问题现象描述](典型故障专题/AI_Core_Error问题定位专题/AI_Core_Error问题现象描述.md)
- [使用msaicerr工具分析AI Core Error问题](典型故障专题/AI_Core_Error问题定位专题/使用msaicerr工具分析AI_Core_Error问题.md)
- [收集AI Core Error问题信息](典型故障专题/AI_Core_Error问题定位专题/收集AI_Core_Error问题信息.md)

### 内存OOM问题定位专题

#### 日志分析手段&典型案例

- [使用第三方asan工具检测内存错误问题](典型故障专题/内存OOM问题定位专题/日志分析手段&典型案例/使用第三方asan工具检测内存错误问题.md)
- [查看CANN各组件内存统计信息](典型故障专题/内存OOM问题定位专题/日志分析手段&典型案例/查看CANN各组件内存统计信息.md)
- [查看Device业务进程内存统计信息](典型故障专题/内存OOM问题定位专题/日志分析手段&典型案例/查看Device业务进程内存统计信息.md)

- [内存OOM问题定位思路](典型故障专题/内存OOM问题定位专题/内存OOM问题定位思路.md)
- [内存OOM问题现象描述](典型故障专题/内存OOM问题定位专题/内存OOM问题现象描述.md)
- [收集内存OOM问题信息](典型故障专题/内存OOM问题定位专题/收集内存OOM问题信息.md)

### 进程中断问题定位专题

#### 典型案例

- [rtMemcpyAsync异步参数校验报错](典型故障专题/进程中断问题定位专题/典型案例/rtMemcpyAsync异步参数校验报错.md)
- [环境变量访问冲突，导致应用程序异常终止](典型故障专题/进程中断问题定位专题/典型案例/环境变量访问冲突，导致应用程序异常终止.md)
- [算子执行过程中D2D拷贝出错](典型故障专题/进程中断问题定位专题/典型案例/算子执行过程中D2D拷贝出错.md)
- [调用SetDevice接口错误导致无可用Context](典型故障专题/进程中断问题定位专题/典型案例/调用SetDevice接口错误导致无可用Context.md)

- [收集进程中断问题信息](典型故障专题/进程中断问题定位专题/收集进程中断问题信息.md)
- [进程中断问题定位思路](典型故障专题/进程中断问题定位专题/进程中断问题定位思路.md)
- [进程中断问题现象描述](典型故障专题/进程中断问题定位专题/进程中断问题现象描述.md)

### 进程卡住问题定位专题

#### 典型案例

- [fork方式创建子进程导致应用进程卡死](典型故障专题/进程卡住问题定位专题/典型案例/fork方式创建子进程导致应用进程卡死.md)

- [收集进程卡住问题信息](典型故障专题/进程卡住问题定位专题/收集进程卡住问题信息.md)
- [进程卡住问题定位思路](典型故障专题/进程卡住问题定位专题/进程卡住问题定位思路.md)

## 故障定位工具

### asys工具使用指导

- [AI Core Error故障信息解析](故障定位工具/asys工具使用指导/AI_Core_Error故障信息解析.md)
- [FAQ](故障定位工具/asys工具使用指导/FAQ.md)
- [asys工具功能及约束](故障定位工具/asys工具使用指导/asys工具功能及约束.md)
- [coredump文件解析](故障定位工具/asys工具使用指导/coredump文件解析.md)
- [coretrace文件解析](故障定位工具/asys工具使用指导/coretrace文件解析.md)
- [stackcore文件解析](故障定位工具/asys工具使用指导/stackcore文件解析.md)
- [trace文件解析](故障定位工具/asys工具使用指导/trace文件解析.md)
- [业务复跑+故障信息收集](故障定位工具/asys工具使用指导/业务复跑+故障信息收集.md)
- [健康检查](故障定位工具/asys工具使用指导/健康检查.md)
- [实时堆栈导出](故障定位工具/asys工具使用指导/实时堆栈导出.md)
- [性能数据采集](故障定位工具/asys工具使用指导/性能数据采集.md)
- [故障信息收集](故障定位工具/asys工具使用指导/故障信息收集.md)
- [环境配置](故障定位工具/asys工具使用指导/环境配置.md)
- [组件检测](故障定位工具/asys工具使用指导/组件检测.md)
- [综合检测](故障定位工具/asys工具使用指导/综合检测.md)
- [软硬件、Device状态信息展示](故障定位工具/asys工具使用指导/软硬件、Device状态信息展示.md)

- [msaicerr工具使用指导](故障定位工具/msaicerr工具使用指导.md)

## 错误码参考

### AI CPU Errors

- [E30005 Device_Connection_Failure](错误码参考/AI_CPU_Errors/E30005_Device_Connection_Failure.md)
- [E30006 Package_Error_Verify_OPP](错误码参考/AI_CPU_Errors/E30006_Package_Error_Verify_OPP.md)
- [E30008 Execution_Error_AICPU_Operator_Timeout](错误码参考/AI_CPU_Errors/E30008_Execution_Error_AICPU_Operator_Timeout.md)
- [E39009 Inner_Error_Device_Status_Abnormal](错误码参考/AI_CPU_Errors/E39009_Inner_Error_Device_Status_Abnormal.md)

### Auto Tune Errors

- [EC0009 Resource_Error](错误码参考/Auto_Tune_Errors/EC0009_Resource_Error.md)

### Driver Errors

- [EL0005 Resource_Busy](错误码参考/Driver_Errors/EL0005_Resource_Busy.md)
- [EL0007 No_Permission](错误码参考/Driver_Errors/EL0007_No_Permission.md)
- [EL0015 Invalid_Device_Access](错误码参考/Driver_Errors/EL0015_Invalid_Device_Access.md)

### FE Errors

- [E21001 File_Operation_Error_Open](错误码参考/FE_Errors/E21001_File_Operation_Error_Open.md)

### GE Errors

- [E10501 Not_Supported_Operator](错误码参考/GE_Errors/E10501_Not_Supported_Operator.md)
- [E13029 Compilation_Error_Load_Custom_Fusion_Pass](错误码参考/GE_Errors/E13029_Compilation_Error_Load_Custom_Fusion_Pass.md)

### HCCL Errors

- [EI0002  Communication_Error_Timeout](错误码参考/HCCL_Errors/EI0002_Communication_Error_Timeout.md)
- [EI0012 Execution_Error_SDMA](错误码参考/HCCL_Errors/EI0012_Execution_Error_SDMA.md)
- [EI0013 Execution_Error_ROCE_CQE](错误码参考/HCCL_Errors/EI0013_Execution_Error_ROCE_CQE.md)
- [EI0015 Ranktable_Detect_Failed](错误码参考/HCCL_Errors/EI0015_Ranktable_Detect_Failed.md)

### RTS Errors

- [EE4002 Model_Unbinding_Errors](错误码参考/RTS_Errors/EE4002_Model_Unbinding_Errors.md)

### TBE Pass Compiler Errors

- [EB1000 Compilation_Error_Invalid_Primitives](错误码参考/TBE_Pass_Compiler_Errors/EB1000_Compilation_Error_Invalid_Primitives.md)

### TEFusion Errors

- [E40003 File_Operation_Error_Open](错误码参考/TEFusion_Errors/E40003_File_Operation_Error_Open.md)
- [E40024 Environment_Error_Call_Python_Function_Failed](错误码参考/TEFusion_Errors/E40024_Environment_Error_Call_Python_Function_Failed.md)

- [使用说明](错误码参考/使用说明.md)

- [故障处理简介](故障处理简介.md)

