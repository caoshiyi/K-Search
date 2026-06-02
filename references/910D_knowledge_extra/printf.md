# printf<a name="ZH-CN_TOPIC_0000002523344766"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p12300735171314"><a name="p12300735171314"></a><a name="p12300735171314"></a><span id="ph730011352138"><a name="ph730011352138"></a><a name="ph730011352138"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section259105813316"></a>

该接口提供CPU域/NPU域调试场景下的格式化输出功能。

在算子kernel侧实现代码中需要输出日志信息的地方调用printf接口打印相关内容。样例如下：

```
#include "kernel_operator.h"
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);
```

> **注意：** 
>printf（PRINTF）接口打印功能会对算子实际运行的性能带来一定影响，通常在调测阶段使用。开发者可以按需通过设置ASCENDC\_DUMP=0的方式关闭打印功能。

## 函数原型<a name="section2067518173415"></a>

```
template <class... Args>
__aicore__ inline void printf(__gm__ const char* fmt, Args&&... args)
template <class... Args>
__aicore__ inline void PRINTF(__gm__ const char* fmt, Args&&... args)
```

## 参数说明<a name="section158061867342"></a>

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p45208478318"><a name="p45208478318"></a><a name="p45208478318"></a>fmt</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p135196472314"><a name="p135196472314"></a><a name="p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1965816506312"><a name="p1965816506312"></a><a name="p1965816506312"></a>格式控制字符串，包含两种类型的对象：普通字符和转换说明。</p>
<a name="ul419411543310"></a><a name="ul419411543310"></a><ul id="ul419411543310"><li>普通字符将原样不动地打印输出。</li><li>转换说明并不直接输出而是用于控制printf中参数的转换和打印。每个转换说明都由一个百分号字符（%）开始，以转换说明结束，从而说明输出数据的类型 。<div class="p" id="p158820115597"><a name="p158820115597"></a><a name="p158820115597"></a>支持的转换类型包括：<a name="ul541124915329"></a><a name="ul541124915329"></a><ul id="ul541124915329"><li>%d / %i：输出十进制数，支持打印的数据类型：bool/int8_t/int16_t/int32_t/int64_t</li><li>%f：输出实数，支持打印的数据类型：float/half</li><li>%x：输出十六进制整数，支持打印的数据类型：int8_t/int16_t/int32_t/int64_t/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%s：输出字符串</li><li>%u：输出unsigned类型数据，支持打印的数据类型：bool/uint8_t/uint16_t/uint32_t/uint64_t</li><li>%p：输出指针地址</li></ul>
</div>
</li></ul>
<p id="p532441419813"><a name="p532441419813"></a><a name="p532441419813"></a><strong id="b164621818125011"><a name="b164621818125011"></a><a name="b164621818125011"></a>注意</strong>：</p>
<a name="ul63292141989"></a><a name="ul63292141989"></a><ul id="ul63292141989"><li>上文列出的数据类型是NPU域调试支持的数据类型，CPU域调试时，支持的数据类型和C/C++规范保持一致。</li><li>在转换类型为%x，即输出十六进制整数时，NPU域上的输出为64位，CPU域上的输出为32位。</li></ul>
</td>
</tr>
<tr id="row163919564263"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="p1563916565265"><a name="p1563916565265"></a><a name="p1563916565265"></a>args</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="p59396564285"><a name="p59396564285"></a><a name="p59396564285"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="p1257115311337"><a name="p1257115311337"></a><a name="p1257115311337"></a>附加参数，个数和类型可变的参数列表：根据不同的fmt字符串，函数可能需要一系列的附加参数，每个参数包含了一个要被插入的值，替换了fmt参数中指定的每个%标签。参数的个数应与%标签的个数相同。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section43265506459"></a>

-   本接口不支持打印除换行符之外的其他转义字符。
-   如果开发者需要包含标准库头文件stdio.h和cstdio，请在kernel\_operator.h头文件之前包含，避免printf符号冲突。
-   单次调用本接口打印的数据总量不可超过1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。在使用自定义算子工程进行工程化算子开发时，一个算子所有使用Dump功能的接口在每个核上Dump的数据总量不可超过1MB。请开发者自行控制待打印的内容数据量，超出则不会打印。
-   根据算子执行方式的不同，printf的打印结果输出方式不同。动态图或者单算子直调场景下，待输出内容会被解析并打印在屏幕上；静态图场景下，整图算子需要全下沉到NPU侧执行，无法直接调用接口打印出单个算子的信息，因此需要在模型执行完毕后，将待输出内容落盘在dump文件中，dump文件需要通过工具解析为可读内容。
    -   dump文件落盘路径按照优先级排列如下：
        -   如果开启了Data Dump功能，dump文件落盘到开发者配置的dump\_path路径下。如何开启Dump功能依赖于具体的网络运行方式。以TensorFlow在线推理为例，通过enable\_dump、dump\_path、dump\_mode等参数进行配置。配置方式可参考《TensorFlow 2.6.5模型迁移指南》中的API参考 \> TF Adapter 接口（2.x）\> npu.global\_options \> 配置参数说明章节。
        -   如果未开启Data Dump功能，但配置了ASCEND\_WORK\_PATH环境变量，dump文件落盘到ASCEND\_WORK\_PATH下的printf目录下。ASCEND\_WORK\_PATH环境变量的配置方式可参考《环境变量参考》。
        -   如果未开启Data Dump功能也没有配置ASCEND\_WORK\_PATH环境变量，dump文件落盘到当前程序执行目录下的printf路径下。

    -   落盘dump文件需要使用工具解析为用户可读内容：

        使用show\_kernel\_debug\_data工具将dump二进制文件解析为用户可读内容，命令格式如下。show\_kernel\_debug\_data的具体使用方法请参考[show\_kernel\_debug\_data工具](show_kernel_debug_data工具.md)。

        ```
        show_kernel_debug_data  bin_file  output_dir
        ```

-   Ascend 950PR/Ascend 950DT，暂不支持静态图场景下的打印。
-   算子入图场景，若一个动态Shape模型中有可下沉的部分，框架内部会将模型拆分为动态调度和下沉调度（静态子图）两部分，静态子图中的算子不支持该printf特性。

## 调用示例<a name="section82241477610"></a>

```
#include "kernel_operator.h"

// 整型打印：
AscendC::printf("fmt string %d\n", 0x123);
AscendC::PRINTF("fmt string %d\n", 0x123);

// 浮点型打印：
float a = 3.14;
AscendC::printf("fmt string %f\n", a);
AscendC::PRINTF("fmt string %f\n", a);

// 指针打印：
int *b;
AscendC::printf("TEST %p\n", b);
AscendC::PRINTF("TEST %p\n", b);
```

NPU模式下，程序运行时打印效果如下（CANN Version和TimeStamp仅在使用自定义算子工程时才会打印）：

```
CANN Version: XXX.XX, TimeStamp: 202408
fmt string 291
fmt string 291
fmt string 3.140000
fmt string 3.140000
TEST 0x12c08001a000
TEST 0x12c08001a000
```

