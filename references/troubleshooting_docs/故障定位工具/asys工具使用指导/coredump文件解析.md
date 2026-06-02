# coredump文件解析

**页面ID:** troubleshooting_0510  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0510.html

---

- **功能说明**

解析coredump文件。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **注意事项**

coredump解析功能依赖gdb，需提前安装gdb，可通过包管理（如apt-get install gdb、yum install gdb）进行安装，详细安装步骤及使用方法请参见[GDB官方文档](https://sourceware.org/gdb/)。

- **命令格式**

```
**asys analyze -r=coredump --exe_file=***filename* **--core_file=***filename* **--reg=***reglevel* **--symbol=***value* **--output=***path*
```

- **参数说明**

  - **r** ：必选参数，解析模式，此处设置为coredump。在执行任务过程中进程中断退出，软件在退出时会报Segmentation fault等错误，可以使用该功能解析coredump生成的core文件，获取stackcore格式的堆栈文件（*.txt文件），供后续定位使用。
  - **exe_file**：程序coredump时的可执行文件，此处设置为包含路径的文件名，coredump模式必选，需要保证与core_file相匹配，否则解析结果错误。
  - **core_file**：程序coredump时生成的core文件，此处设置为包含路径的文件名，coredump模式必选，需要保证与exe_file相匹配，否则解析结果错误。
  - **reg**：coredump功能添加寄存器数据的模式，只支持0、1和2，默认为0，coredump模式可选。

    - 0：不添加寄存器数据。
    - 1：每个线程加一条寄存器数据。
    - 2：线程的所有栈都添加寄存器数据（占用Host资源较多，比较耗时）。

  - **symbol**：coredump功能解析模式，只支持0和1，默认为0，coredump模式可选，地址不存在或栈溢出可能导致asys工具无法解析。

    - 0：将所有带地址行解析成stackcore格式的文件，其他行显示Ignore，表示跳过不解析。
    - 1：只解析in ?? () 行，其他行保留gdb堆栈的原数据

  - **output**：可选参数，其值作为asys工具的结果输出目录的路径前缀，即最终输出目录为{output}/asys_output_timestamp。命令行中不带output参数时，输出结果存放在命令行执行目录下；若output指定值为空、无效字符串、或指定路径目录无写权限、或创建目录失败，则asys工具退出执行并报错。

- **使用示例及输出说明**

  - 将symbol指定为0（解析所有地址行）

```
**asys analyze -r=coredump --exe_file=***atrace_test* **--core_file=***core_atrace_test_001* **--symbol=**0 **--output=***$HOME/dfx_info*
```

解析后的stackcore格式文件示例如下，后续可以使用asys工具的stackcore文件解析功能进一步解析。

```
[process]
crash reason: SIGABRT
crash pid: 37246
crash tid: 37246

[stack]
Thread 1 (37246)
#00 0x00007fbad83792bf 0x00007fbad830b000 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0
#01                                       Ignore
#02 0x00007fbad83d8c22 0x00007fbad830b000 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0
#03 0x00007fbad83e9648 0x00007fbad830b000 /usr/local/python3.7.5/lib/libpython3.7m.so.1.0
......

[maps]
    Start Addr           End Addr       Size     Offset objfile
0x562677ed1000     0x562677ed2000     0x1000        0x0 /usr/local/python3.7.5/bin/python3.7
0x562677ed2000     0x562677ed3000     0x1000     0x1000 /usr/local/python3.7.5/bin/python3.7
0x562677ed3000     0x562677ed4000     0x1000     0x2000 /usr/local/python3.7.5/bin/python3.7
0x562677ed4000     0x562677ed5000     0x1000     0x2000 /usr/local/python3.7.5/bin/python3.7
......
```

  - 将symbol指定为1（只解析in ?? () 行）

```
**asys analyze -r=coredump --exe_file=***atrace_test* **--core_file=***core_atrace_test_002* **--symbol=**1 **--output=***$HOME/dfx_info*
```

解析后的stackcore格式文件示例如下，后续可以使用asys工具的stackcore文件解析功能进一步解析。

```
[process]
crash reason: SIGABRT
crash pid: 37246
crash tid: 37246

[stack]
Thread 1 (37246)
#00 0x00007fbad83792bf in lookdict_unicode (value_addr=0x7ffea1e917e8, hash=<optimized out>, key=<optimized out>, mp=0x7fba98907fa0) at Objects/dictobject.c:811
#01 lookdict_unicode (mp=0x7fba98907fa0, key=<optimized out>, hash=<optimized out>, value_addr=0x7ffea1e917e8) at Objects/dictobject.c:783
#02 0x00007fbad83d8c22 in PyDict_GetItem (op=op@entry=0x7fba98907fa0, key=key@entry=0x7fba9a15b570) at Objects/dictobject.c:1327
#03 0x00007fbad83e9648 in _PyObject_GenericGetAttrWithDict (obj=obj@entry=0x7fba989083b0, name=name@entry=0x7fba9a15b570, dict=0x7fba98907fa0, dict@entry=0x0, suppress=suppress@entry=0) at Objects/object.c:1268
......

[maps]
    Start Addr           End Addr       Size     Offset objfile
0x562677ed1000     0x562677ed2000     0x1000        0x0 /usr/local/python3.7.5/bin/python3.7
0x562677ed2000     0x562677ed3000     0x1000     0x1000 /usr/local/python3.7.5/bin/python3.7
0x562677ed3000     0x562677ed4000     0x1000     0x2000 /usr/local/python3.7.5/bin/python3.7
0x562677ed4000     0x562677ed5000     0x1000     0x2000 /usr/local/python3.7.5/bin/python3.7
......
```
