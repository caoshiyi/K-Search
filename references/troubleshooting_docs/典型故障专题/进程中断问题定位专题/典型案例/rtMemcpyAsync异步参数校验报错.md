# rtMemcpyAsync异步参数校验报错

**页面ID:** troubleshooting_0066  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0066.html

---

#### 问题现象

参见收集进程中断问题信息收集日志文件，收集的日志所存放的目录，下文以${HOME}/err_log_info/为例。

Host应用类日志（${HOME}/err_log_info/log/[run|debug]/plog/plog-*pid*_*.log）日志中存在“Memory async failed” 、“kind=x is invalid”关键词，日志示例如下：

```
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.792 [api_error.cc:963]291577 MemcpyAsyncCheckKindAndLocation:[EXEC][EXEC]report error module_name=EE1001
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.799 [api_error.cc:963]291577 MemcpyAsyncCheckKindAndLocation:[EXEC][EXEC]**Memory async failed, src loc type=2, dst loc type=2, kind=1 is invalid!**
[INFO] GE(291088,python3):2024-01-29-19:2856.519.818 [error_manager.cc:301]291577 ReportErrMessage:report error_message, error_code:EE1001, work_stream_id:11.
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.868 [api_error.cc:881]291577 MemcpyAsync:[EXEC][EXEC]report error module_type=3, module_name=EE8888
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.877 [api_error.cc:881]291577 MemcpyAsync:[EXEC][EXEC]Memory async failed, check kind and loc,retCode=0x7110001
[INFO] GE(291088,python3):2024-01-29-19:2856.519.887 [error_manager.cc:2551]291577 ReportInterErrMessage:report error_message, error_code:EE8888, work_stream_id:11
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.911 [api_c.cc:1428]291577 rtMemcpyAsync:[EXEC][EXEC]ErrCode=107000, desc=[invalid value], InnertCode=0x7110001
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.920 [error_message_manage.cc:48]291577 FuncErrorReason:[EXEC][EXEC]report error module_name=EE1001
[ERROR] RUNTIME(291088,python3):2024-01-29-19:2856.519.929 [error_message_manage.cc:48]291577 FuncErrorReason:[EXEC][EXEC]rtMemcpyAsync execute failed, reason=[invalid value]
[INFO] GE(291088,python3):2024-01-29-19:2856.519.938 [error_manager.cc:301]291577 ReportErrMessage:report error_message, error_code:EE1001, work_stream_id:11
[ERROR] ASCENDCL(291088,python3):2024-01-29-19:2856.519.952 [memory.cc:336]291577 aclrtMemcpyAsync:[EXEC][EXEC]asynchronized memcpy failed, kind = 1, runtime result = 107000
```

日志中关键字段解释如下：

- loc type（地址类型）参数值解释如下：

  - 1：Host内存地址
  - 2：Device内存地址
  - 3：SVM内存地址
  - 4：DVPP内存地址

- kind（拷贝类型）参数值解释如下：

  - 0：Host内的数据拷贝
  - 1：Host->Device的数据拷贝
  - 2：Device->Host的数据拷贝
  - 3：Device内的数据拷贝或两个Device之间的数据拷贝
  - 4：管理内存，内部组件使用
  - 5：Device内的数据拷贝或两个Device之间的数据拷贝，二级指针拷贝，内部组件使用

#### 故障根因

进行异步拷贝时，用户需指定src和dst地址、以及拷贝类型kind(比如H2D/D2H等)，然后runtime会根据用户的kind校验src和dst地址的有效性，比如H2D，这种情况下src地址应该是Host内存地址，dst地址应该是Device内存地址，若src和dst的地址类型和kind不匹配，则会引起报错。

#### 解决方法

排查异步拷贝的代码逻辑，指定正确的src和dst地址、以及拷贝类型kind。
