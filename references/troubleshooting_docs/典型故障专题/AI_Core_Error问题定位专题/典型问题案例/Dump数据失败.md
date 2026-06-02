# Dump数据失败

**页面ID:** troubleshooting_0018  
**来源:** https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/maintenref/troubleshooting/troubleshooting_0018.html

---

#### 分析结果

如果在info.txt中给出如下分析结论，则推断为dump数据失败：

```
"**********************Root cause conclusion******************"
The input/output memory address of the operator is abnormal (or the original dumped data fails). Check the framework or application.
```

同时在info.txt文件中“Dump info”处会有如下信息：

```
***********************5. Operator Dump File Parsing*************************
tiling data in int32: [1, 1, 64, 1, 2, 1, 1]
tiling data in int64: Cannot decode in this dtype
tiling data in float16: [5.960464477539063e-08, 0.0, 5.960464477539063e-08, 0.0, 3.814697265625e-06, 0.0, 5.960464477539063e-08, 0.0, 1.1920928955078125e-07, 0.0, 5.960464477539063e-08, 0.0, 5.960464477539063e-08, 0.0]

Failed to get dump data of error op!
```

#### 故障根因

AI Core Error发生后，框架会在错误的回调里尝试通过输入/输出的地址和大小去dump一份数据供开发者定位。如果dump失败，则可以推断算子的输入/输出地址有错误：AI Core无法读取这段地址导致了AI Core Error，同时，错误回调中dump功能也无法读取这段地址导致了dump数据失败。

#### 处理方法

出现上面情况，参考以下方法处理：

查看info.txt文件中“4. Operator Input/Output Memory”处的输入/输出地址，同时判断该算子是否为推理中的首算子。

如果为推理首算子，则需要查看用户推理脚本，确认是否为网络分配了足够的地址空间；如果不是首算子，则需要联系技术支持分析错误地址的来源。您可以获取日志后单击Link联系技术支持。

```
****************4. Operator Input/Output Memory*******************
input[0] addr: 0x1240c0047000 end_addr:0x1240c0047100 size: 0x100
input[1] addr: 0x1240c0027000 end_addr:0x1240c0027008 size: 0x8
input[2] addr: 0x1240c0037000 end_addr:0x1240c0037004 size: 0x4
output[0] addr: 0x1240c0057000 end_addr:0x1240c0057008 size: 0x8
```
