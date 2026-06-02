# WaitPreTaskEnd<a name="ZH-CN_TOPIC_0000002554423927"></a>

> **说明：** 
>本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 产品支持情况<a name="section17196114513104"></a>

<a name="zh-cn_topic_0000002523344214_table38301303189"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000002523344214_row20831180131817"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="zh-cn_topic_0000002523344214_p1883113061818"><a name="zh-cn_topic_0000002523344214_p1883113061818"></a><a name="zh-cn_topic_0000002523344214_p1883113061818"></a><span id="zh-cn_topic_0000002523344214_ph20833205312295"><a name="zh-cn_topic_0000002523344214_ph20833205312295"></a><a name="zh-cn_topic_0000002523344214_ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="24.6%" id="mcps1.1.4.1.2"><p id="zh-cn_topic_0000002523344214_p783113012187"><a name="zh-cn_topic_0000002523344214_p783113012187"></a><a name="zh-cn_topic_0000002523344214_p783113012187"></a>是否支持</p>
</th>
<th class="cellrowborder" valign="top" width="21.759999999999998%" id="mcps1.1.4.1.3"><p id="zh-cn_topic_0000002523344214_p182842352418"><a name="zh-cn_topic_0000002523344214_p182842352418"></a><a name="zh-cn_topic_0000002523344214_p182842352418"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000002523344214_row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="zh-cn_topic_0000002523344214_p17301775812"><a name="zh-cn_topic_0000002523344214_p17301775812"></a><a name="zh-cn_topic_0000002523344214_p17301775812"></a><span id="zh-cn_topic_0000002523344214_ph2272194216543"><a name="zh-cn_topic_0000002523344214_ph2272194216543"></a><a name="zh-cn_topic_0000002523344214_ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="24.6%" headers="mcps1.1.4.1.2 "><p id="zh-cn_topic_0000002523344214_p22421043141916"><a name="zh-cn_topic_0000002523344214_p22421043141916"></a><a name="zh-cn_topic_0000002523344214_p22421043141916"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="21.759999999999998%" headers="mcps1.1.4.1.3 "><p id="zh-cn_topic_0000002523344214_p32421543131916"><a name="zh-cn_topic_0000002523344214_p32421543131916"></a><a name="zh-cn_topic_0000002523344214_p32421543131916"></a>该接口生效</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在SuperKernel的子Kernel中调用，调用前的指令可以和前序其他的子Kernel实现并行，提升整体性能。如[图1](#fig99271836191110)所示，SuperKernel按序调用子Kernel，为保证子Kernel之间数据互不干扰，会在子Kernel间插入算子间同步进行保序，子Kernel<sub>N+1</sub>调用该接口之前的指令会和前序子Kernel<sub>N</sub>实现并行。

SuperKernel是一种算子的二进制融合技术，与源码融合不同，它聚焦于内核函数 \(Kernel\) 的二进制的调度方案，展开深度优化，于已编译的二进制代码基础上融合创建一个超级Kernel函数（SuperKernel），以调用子函数的方式调用多个其他内核函数，也就是子Kernel。相对于单算子下发，SuperKernel技术可以减少任务调度等待时间和调度开销，同时利用Task间隙资源进一步优化算子头开销。

**开发者需要自行保证调用此接口前的指令不会与前序算子互相干扰而导致精度问题，推荐在整个算子第一条搬运指令前调用此接口。**

**图 1**  通过WaitPreTaskEnd实现并行示意图<a name="fig99271836191110"></a>  
<!-- img2text -->
```text
┌──────────────────────┐      ┌────────┐      ┌──────────────────────┐      ┌────────┐
│      子kernelₙ₋₁      │      │ 算子   │      │       子kernelₙ       │      │ 算子   │
└──────────────────────┘      │ 间同   │      └──────────────────────┘      │ 间同   │
                              │ 步     │                                    │ 步     │
                              └────────┘                                    └────────┘
                                              ┌────────────────┐
                                              │  子kernelₙ₊₁   │
                                              └────────────────┘

                                                                     ↑
                                                                     │
                                                             ┌───────────────┐
                                                             │ WaitPreTaskEnd │
                                                             └───────────────┘

                                                                     ┌──────────────────────────────┐
                                                                     │        子kernelₙ₊₁           │
                                                                     └──────────────────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

```
__aicore__ inline void WaitPreTaskEnd()
```

## 参数说明<a name="section622mcpsimp"></a>

无

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   该接口适用于TorchAir图模式开发场景，且需在启用SuperKernel特性后方可生效。相关信息可参考《PyTorch图模式使用指南\(TorchAir\)》中的“max-autotune模式功能 \> 图内标定SuperKernel范围”章节。
-   在算子运行过程中，需要保证此接口在每个核上都被调用，且每个核上仅被调用一次。
-   若子Kernel某个TilingKey分支调用了此接口，则开发者需要保证当前算子可能会运行的所有TilingKey均调用了此接口，否则会出现因同步指令数量不匹配而卡住的现象。

## 调用示例<a name="section837496171220"></a>

```
#include "kernel_operator.h"

AscendC::LocalTensor<half> src0Local = inQueueSrc0.AllocTensor<half>();
// 算子第一条搬运指令前插入，且保证只调用一次
AscendC::WaitPreTaskEnd();
AscendC::DataCopy(src0Local, src0Global, 512);
inQueueSrc0.EnQue(src0Local);;
```

