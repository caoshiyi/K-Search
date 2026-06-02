# trace文件解析

**页面ID:** troubleshooting_0507  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0507.html

---

- **功能说明**

解析trace文件。

若需获取trace文件（*.bin格式），请参见《日志参考》中的“查看trace日志”。

- **产品支持情况**

| 产品 | 是否支持 |
| --- | --- |
| Atlas A3 训练系列产品             /              Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品             /              Atlas A2 推理系列产品 | √ |
| Atlas 200I/500 A2 推理产品 | √ |
| Atlas 推理系列产品 | √ |
| Atlas 训练系列产品 | √ |

- **命令格式**

```
**asys analyze -r=trace --file=***filename* **--output=***path*
```

- **参数说明**

  - **r** ： 必选参数，解析模式，此处设置为trace，用于将trace日志文件（*.bin文件）解析为.txt格式的文件，使用asys工具的环境版本要与产生trace日志的环境版本保持一致。
  - **file**：用于解析单个文件，此处设置为包含路径的文件名，trace模式必选。
  - **output**：可选参数，其值作为asys工具的结果输出目录的路径前缀，即最终输出目录为{output}/asys_output_timestamp。命令行中不带output参数时，输出结果存放在命令行执行目录下；若output指定值为空、无效字符串、或指定路径目录无写权限、或创建目录失败，则asys工具退出执行并报错。

- **使用示例**

```
**asys analyze -r=trace --file=***schedule_tracer_demo.bin* **--output=***$HOME/dfx_info*
```

- **输出说明**

解析后的txt文件内容示例：

```
2024-05-09 19:08:12.408.800 demo0: tid0[0], count0[0], tag0[struct0 tag], streamId0[0], deviceIdArray0[0, 1], hostIdArray0[1, 2, 3, 4]
2024-05-09 19:08:12.408.804 demo1: tag1[struct1 tag], streamId1[0], deviceIdArray1[0, 1], hostIdArray1[1, 2, 3, 4]
```
