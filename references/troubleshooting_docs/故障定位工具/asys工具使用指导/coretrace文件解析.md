# coretrace文件解析

**页面ID:** troubleshooting_0512  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0512.html

---

- **功能说明**

解析coretrace文件。

若需获取coretrace文件，请参见《[msnpureport工具使用指南](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)》中的“导出Device侧系统类日志和其他维测信息 > 单次导出Device侧系统类日志和其他维测信息”章节导出coretrace文件。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | ☓ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | ☓ |
| Atlas 200I/500 A2 推理产品 | ☓ |
| Atlas 推理系列产品 | ☓ |
| Atlas 训练系列产品 | ☓ |

- **注意事项**

coretrace解析功能依赖addr2line进行堆栈函数名和行号的解析，这是linux系统自带工具，请确保addr2line已安装且执行该脚本的用户有权限执行。

- **命令格式**

```
**asys analyze -r=coretrace --file=***filename* **--symbol_path=***path1,path2* **--output=***path3*
```

或

```
**asys analyze -r=coretrace --path=***directory* **--symbol_path=***path1,path2* **--output=***path3*
```

- **参数说明**

  - **r **： 必选参数，解析模式，此处设置为coretrace，用于解析coretrace格式的文件（coretrace.*文件），供后续定位使用。
  - **file**：用于解析单个文件，此处设置为包含路径的文件名，coretrace模式必选。
  - **path**：指定目录，用于解析指定目录及其子目录下的多个文件，coretrace模式下选择path参数、file参数中的一个，两者不能同时存在。
  - **symbol_path**：coretrace模式解析所需要的动态库目录，可传多个目录，用逗号隔开，symbol_path只扫描当前目录下的动态库，先找路径1，再找路径2，不扫描子目录，为防止误解析建议将相关动态库放在一个路径下。coretrace模式可选，不指定symbol_path参数时从coretrace文件中获取所需要的动态库路径，为防止找不到动态库文件，建议仅在发生coredump错误的环境上使用。
  - **output**：可选参数，其值作为asys工具的结果输出目录的路径前缀，即最终输出目录为{output}/asys_output_timestamp。命令行中不带output参数时，输出结果存放在命令行执行目录下；若output指定值为空、无效字符串、或指定路径目录无写权限、或创建目录失败，则asys工具退出执行并报错。

- **使用示例及输出说明**

```
**asys analyze -r=coretrace --file=**coretrace.log-daemon.12335.11.1749181305 **--symbol_path=***$HOME/test1,$HOME/test2* **--output=***$HOME/dfx_info*
```

解析后的文件示例如下，其中，线程信息以"PID *num* (*线程id*，*线程名*)"开头，解析后的内容按“{地址} {函数名} {二进制名}”格式输出，例如：“0xdfffcac78a68    0x00000000000b4a68: clock_nanosleep at ??:?    /usr/lib64/libc.so.6”。若函数名获取失败，则显示解析过程中自动计算出来的十六进制值，此时需检查symbol_path配置目录是否正确以及该目录下的动态库文件是否正确。

```
Signal 11 pid 12335

PID 12335 TGID 12335 comm log-daemon
0xdfffcac78a68    0x00000000000b4a68: clock_nanosleep at ??:?    /usr/lib64/libc.so.6
0xdfffcac7dc4c    0x00000000000b9c48: __nanosleep at ??:?    /usr/lib64/libc.so.6
0xdfffcaca6d88    0x00000000000e2d84: usleep at ??:?    /usr/lib64/libc.so.6
0xaaaad1e22dec    0x0000000000012de8: ToolSleep at log_system_api.c:704    /var/log-daemon
0xaaaad1e1cb58    0x000000000000cb54: main at log_daemon.c:217    /var/log-daemon
0xdfffcabef040    0x000000000002b03c: __libc_init_first at ??:?    /usr/lib64/libc.so.6
0xdfffcabef118    0x000000000002b114: __libc_start_main at ??:?    /usr/lib64/libc.so.6
0xaaaad1e1c680    0x000000000000c67c: $x at start.os:?    /var/log-daemon

PID 12356 TGID 12335 comm adx_get_file_th
0xdfffcaca550c    0x00000000000e150c: ioctl at ??:?    /usr/lib64/libc.so.6
0xdfffcbc413fc    0x00000000000543f8: HiIam::AppIoctl(int, unsigned long, void*, unsigned int&, bool) at ??:?    /usr/lib64/libiam.so.0.1.0.0
0xdfffcbc415ec    0x00000000000545e8: ioctl at ??:?    /usr/lib64/libiam.so.0.1.0.0
0xdfffcb2d61ec    0x000000000006f1e8: mmIoctl at hdc_pcie_drv.c:?    /usr/lib64/libascend_hal.so
0xdfffcb2d678c    0x000000000006f788: hdcIoctl at hdc_pcie_drv.c:?    /usr/lib64/libascend_hal.so
0xdfffcb2d8550    0x000000000007154c: hdcPcieEpollWait at ??:?    /usr/lib64/libascend_hal.so
0xdfffcb2dabf4    0x0000000000073bf0: drvHdcPcieEpollWait at hdc_pcie_epoll.c:?    /usr/lib64/libascend_hal.so
0xdfffcb2da7f4    0x00000000000737f0: drvHdcEpollWait at ??:?    /usr/lib64/libascend_hal.so
0xaaaad1e6dc60    0x000000000005dc5c: Adx::AdxHdcEpoll::EpollWait(std::vector<Adx::EpollEvent, std::allocator<Adx::EpollEvent> >&, int, int) at ??:?    /var/log-daemon
0xaaaad1e6c51c    0x000000000005c518: Adx::AdxServerManager::ComponentWaitEvent() at :?    /var/log-daemon
0xaaaad1e6c780    0x000000000005c77c: Adx::AdxServerManager::Run() at ??:?    /var/log-daemon
0xaaaad1e6ea0c    0x000000000005ea08: Adx::Runnable::Process(void*) at ??:?    /var/log-daemon
0xdfffcac46168    0x0000000000082164: pthread_condattr_setpshared at ??:?    /usr/lib64/libc.so.6
0xdfffcacad8dc    0x00000000000e98d8: clone at ??:?    /usr/lib64/libc.so.6
......
```

  - 解析后的文件如果存在?这个字符，可能存在以下原因：

    - 编译选项：该动态库文件编译时没有使用 -g选项，以在文件中保留调试信息
    - 未添加链接参数：未使用-rdynamic来通知链接器将所有符号添加到动态符号表中
    - 未找到动态库：未找到相匹配的动态库

  - coretrace解析函数名和行号时，部分动态库解析出的行号会有少许偏差，原因如下：

    - 编译选项： 不同的编译选项，特别是与调试信息相关的选项，可能会造成影响。
    - 优化级别：较高的优化级别可能会导致代码的重组和优化，从而使行号与原始源代码的对应关系发生了偏差。
