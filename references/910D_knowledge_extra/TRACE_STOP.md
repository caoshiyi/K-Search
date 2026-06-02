# TRACE\_STOP<a name="ZH-CN_TOPIC_0000002523343808"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p39198265239"><a name="p39198265239"></a><a name="p39198265239"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section259105813316"></a>

通过CAModel进行算子性能仿真时，可对算子任意运行阶段打点，从而分析不同指令的流水图，以便进一步性能调优。

用于表示终止位置打点，一般与[TRACE\_START](TRACE_START.md)配套使用。

> **注意：** 
>该功能主要用于**调试和性能分析**，开启后会对算子性能产生一定影响，通常在调测阶段使用，**生产环境建议关闭**。
>默认情况下，该功能关闭，开发者可以按需通过如下方式开启打点功能。
>修改Kernel直调工程cmake目录下的npu\_lib.cmake文件，在ascendc\_compile\_definitions命令中增加-DASCENDC\_TRACE\_ON编译选项，来开启打点功能。示例如下：
>```
>// 打开算子的打点功能
>ascendc_compile_definitions(ascendc_kernels_${RUN_MODE} PRIVATE
>    -DASCENDC_TRACE_ON
>)
>```

## 函数原型<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section2067518173415"></a>

```
#define TRACE_STOP(TraceId apid)
```

## 参数说明<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section158061867342"></a>

<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.93%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.58%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_row42461942101815"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002000199857_p3634mcpsimp"><a name="zh-cn_topic_0000002000199857_p3634mcpsimp"></a><a name="zh-cn_topic_0000002000199857_p3634mcpsimp"></a>apid</p>
</td>
<td class="cellrowborder" valign="top" width="11.93%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_p135196472314"><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_p135196472314"></a><a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001656094169_p135196472314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.58%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002000199857_p1098217404"><a name="zh-cn_topic_0000002000199857_p1098217404"></a><a name="zh-cn_topic_0000002000199857_p1098217404"></a>取值需与<a href="TRACE_START.md">TRACE_START</a>参数取值保持一致，否则影响打点结果。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section640mcpsimp"></a>

无

## 约束说明<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section43265506459"></a>

-   TRACE\_START/TRACE\_STOP需配套使用，若Trace图上未显示打点，则说明两者没有配对。
-   不支持跨核使用，例如TRACE\_START在AI Cube打点，则TRACE\_STOP打点也需要在AI Cube上，不能在AI Vector上。
-   宏支持所有的产品型号，但实际调用时需与调测工具支持的型号保持一致。
-   仅支持Kernel直调工程，不支持自定义算子工程下开启打点功能。

## 调用示例<a name="zh-cn_topic_0000002000199857_zh-cn_topic_0000001997078721_section82241477610"></a>

在Kernel代码中特定指令位置打上TRACE\_START/TRACE\_STOP：

```
TRACE_START(0x1);
DataCopy(zGm, zLocal, this->totalLength);
TRACE_STOP(0x1);
```

