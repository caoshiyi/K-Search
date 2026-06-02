# stackcore文件解析

**页面ID:** troubleshooting_0511  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0511.html

---

- **功能说明**

解析stackcore文件。

stackcore文件有多种获取来源：

  - Host侧stackcore文件（stackcore_tracer_*.txt），请参见《日志参考》中的“查看trace日志”章节获取stackcore文件。
  - Device侧stackcore文件，请参见《[msnpureport工具使用指南](https://support.huawei.com/enterprise/zh/ascend-computing/ascend-hdk-pid-252764743?category=reference-guides&subcategory=command-reference)》中的“导出Device侧系统类日志和其他维测信息 > 单次导出Device侧系统类日志和其他维测信息”章节导出stackcore文件。
  - 使用asys工具的coredump文件解析功能获得的stackcore文件。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品             /              Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品             /              Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **注意事项**

stackcore解析功能依赖readelf进行文件信息的获取、依赖addr2line进行堆栈函数名和行号的解析，两者都是linux系统自带工具，请确保readelf、addr2line已安装且执行该脚本的用户有权限执行。

- **命令格式**

```
**asys analyze -r=stackcore --file=***filename* **--symbol_path=***path1,path2* **--output=***path3*
```

或

```
**asys analyze -r=stackcore --path=***directory* **--symbol_path=***path1,path2* **--output=***path3*
```

- **参数说明**

  - **r **： 必选参数，解析模式，此处设置为stackcore，用于解析stackcore格式的文件（*.txt文件），供后续定位使用。
  - **file**：用于解析单个文件，此处设置为包含路径的文件名，stackcore模式必选。

  - **path**：指定目录，用于解析指定目录及其子目录下的多个文件，stackcore模式下选择path参数、file参数中的一个，两者不能同时存在。
  - **symbol_path**：stackcore模式解析所需要的动态库目录，可传多个目录，用逗号隔开，symbol_path只扫描当前目录下的动态库，先找路径1，再找路径2，不扫描子目录，为防止误解析建议将相关动态库放在一个路径下。stackcore模式可选，不指定symbol_path参数时从stackcore文件中获取所需要的动态库路径，为防止找不到动态库文件，建议仅在发生coredump错误的环境上使用。
  - **output**：可选参数，其值作为asys工具的结果输出目录的路径前缀，即最终输出目录为{output}/asys_output_timestamp。命令行中不带output参数时，输出结果存放在命令行执行目录下；若output指定值为空、无效字符串、或指定路径目录无写权限、或创建目录失败，则asys工具退出执行并报错。

- **使用示例及输出说明**

```
**asys analyze -r=stackcore --file=***stackcore_tracer_test.txt* **--symbol_path=***$HOME/test1,$HOME/test2* **--output=***$HOME/dfx_info*
```

解析后的txt文件示例如下，其中，线程信息以"Thread *num* (*线程id*，*线程名*)"开头，线程名获取失败则显示"unknown"：

```
[process]
crash reason:6
crash pid:37246
crash tid:37246
crash stack base:0x00007ffea1e96000
crash stack top:0x00007ffea1e91770

[stack]
Thread 1 (37246, python3.7)
#00 0x00007fbad83792bf lookdict_unicode in dictobject.c:811 from libpython3.7m.so.1.0
#01                    lookdict_unicode in dictobject.c:783 from libpython3.7m.so.1.0
#02 0x00007fbad83d8c22 PyDict_GetItem in dictobject.c:1328 from libpython3.7m.so.1.0
#03 0x00007fbad83e9648 _PyObject_GenericGetAttrWithDict in object.c:1269 from libpython3.7m.so.1.0
#04 0x00007fbad83e6729 module_getattro in moduleobject.c:704 from libpython3.7m.so.1.0
#05 0x00007fbad83e937b _PyObject_GetMethod in object.c:1137 from libpython3.7m.so.1.0
......

[maps]
e0000380000-e0000381000 rw-p 00000000 00:00 0 
e00003c0000-e00003c1000 rw-p 00000000 00:00 0 
562677ed1000-562677ed2000 r--p 00000000 fd:00 13113992                   /usr/local/python3.7.5/bin/python3.7
562677ed2000-562677ed3000 r-xp 00001000 fd:00 13113992                   /usr/local/python3.7.5/bin/python3.7
562677ed3000-562677ed4000 r--p 00002000 fd:00 13113992                   /usr/local/python3.7.5/bin/python3.7
......
```

  - 解析后的txt文件如果存在?这个字符，可能存在以下原因：

    - 编译选项：该动态库文件编译时没有使用 -g选项，以在文件中保留调试信息
    - 未添加链接参数：未使用-rdynamic来通知链接器将所有符号添加到动态符号表中
    - 未找到动态库：未找到相匹配的动态库

  - stackcore解析函数名和行号时，部分动态库解析出的行号会有少许偏差，原因如下：

    - 编译选项： 不同的编译选项，特别是与调试信息相关的选项，可能会造成影响。
    - 优化级别：较高的优化级别可能会导致代码的重组和优化，从而使行号与原始源代码的对应关系发生了偏差。
