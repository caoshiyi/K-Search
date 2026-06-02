# SetNextTaskStart<a name="ZH-CN_TOPIC_0000002523344214"></a>

> **说明：** 
>本接口为试验接口，在后续版本中可能会调整或改进，不保证后续兼容性。请开发者在使用过程中关注后续版本更新。

## 产品支持情况<a name="section17196114513104"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="53.64%" id="mcps1.1.4.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="24.6%" id="mcps1.1.4.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
<th class="cellrowborder" valign="top" width="21.759999999999998%" id="mcps1.1.4.1.3"><p id="p182842352418"><a name="p182842352418"></a><a name="p182842352418"></a>备注</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="53.64%" headers="mcps1.1.4.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="24.6%" headers="mcps1.1.4.1.2 "><p id="p22421043141916"><a name="p22421043141916"></a><a name="p22421043141916"></a>√</p>
</td>
<td class="cellrowborder" valign="top" width="21.759999999999998%" headers="mcps1.1.4.1.3 "><p id="p32421543131916"><a name="p32421543131916"></a><a name="p32421543131916"></a>该接口生效</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

在SuperKernel的子Kernel中调用，调用后的指令可以和后续其他的子Kernel实现并行，提升整体性能。如[图1](#fig37581010773)所示，SuperKernel按序调用子Kernel，为保证子Kernel之间数据互不干扰，会在子Kernel间插入算子间同步进行保序，子Kernel<sub>N-1</sub>调用该接口后，之后的指令会和后续子Kernel<sub>N</sub>实现并行。

SuperKernel是一种算子的二进制融合技术，与源码融合不同，它聚焦于内核函数 \(Kernel\) 的二进制的调度方案，展开深度优化，于已编译的二进制代码基础上融合创建一个超级Kernel函数（SuperKernel），以调用子函数的方式调用多个其他内核函数，也就是子Kernel。相对于单算子下发，SuperKernel技术可以减少任务调度等待时间和调度开销，同时利用Task间隙资源进一步优化算子头开销。

**开发者需要自行保证调用此接口后的指令不会与后序算子互相干扰而导致精度问题，推荐在整个算子最后一条搬运指令后调用此接口。**

**图 1**  通过SetNextTaskStart实现并行示意图<a name="fig37581010773"></a>  
<!-- img2text -->
```text
                                     ┌───────────────────────┐
                                     │       子KernelN       │
                                     └───────────────────────┘

┌───────────────────────┐     ╭──────────╮     ┌───────────────┐     ╭──────────╮     ┌───────────────────────┐
│     子KernelN-1       │     │ 算子间同步 │     │   子KernelN-1  │     │ 算子间同步 │     │      子KernelN+1      │
└───────────────────────┘     ╰──────────╯     └───────────────┘     ╰──────────╯     └───────────────────────┘
               ↑
               │
               │
       ┌─────────────────┐
       │ SetNextTaskStart│
       └─────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   该原型在如下产品型号支持：

    Ascend 950PR/Ascend 950DT

    ```
    template<pipe_t AIV_PIPE = PIPE_MTE3, pipe_t AIC_PIPE = PIPE_FIX>
    __aicore__ inline void SetNextTaskStart()
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="50%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p11864712175417"><a name="p11864712175417"></a><a name="p11864712175417"></a>AIV_PIPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p13502115305410"><a name="p13502115305410"></a><a name="p13502115305410"></a>SetNextTaskStart之后运行的指令，如果位于AIV上的AIV_PIPE流水，可以与后序算子并行。AIV_PIPE的取值范围为PIPE_MTE2、PIPE_MTE3、PIPE_S、PIPE_V，流水类型介绍可参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
</td>
</tr>
<tr id="row168561422132317"><td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.1 "><p id="p14624101795415"><a name="p14624101795415"></a><a name="p14624101795415"></a>AIC_PIPE</p>
</td>
<td class="cellrowborder" valign="top" width="50%" headers="mcps1.2.3.1.2 "><p id="p1170619208225"><a name="p1170619208225"></a><a name="p1170619208225"></a>SetNextTaskStart之后运行的指令，如果位于AIC上的AIC_PIPE流水，可以与后序算子并行。AIC_PIPE的取值范围为PIPE_MTE1、PIPE_MTE2、PIPE_MTE3、PIPE_FIX、PIPE_M，流水类型介绍可参考<a href="同步控制简介.md#section1272612276459">硬件流水类型</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   该接口适用于TorchAir图模式开发场景，且需在启用SuperKernel特性后方可生效。相关信息可参考《PyTorch图模式使用指南\(TorchAir\)》中的“max-autotune模式功能 \> 图内标定SuperKernel范围”章节。
-   在算子运行过程中，需要保证此接口在每个核上都被调用，且每个核上仅被调用一次。
-   若子Kernel某个TilingKey分支调用了此接口，则开发者需要保证当前算子可能会运行的所有TilingKey均调用了此接口，否则会出现因同步指令数量不匹配而卡住的现象。

## 调用示例<a name="section837496171220"></a>

```
#include "kernel_operator.h"

AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
AscendC::DataCopy(dstGlobal, dstLocal, 512);
// 算子最后一条搬运指令后插入，且保证只调用一次
AscendC::SetNextTaskStart();
outQueueDst.FreeTensor(dstLocal);
```

