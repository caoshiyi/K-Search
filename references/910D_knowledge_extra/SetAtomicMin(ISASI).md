# SetAtomicMin\(ISASI\)<a name="ZH-CN_TOPIC_0000002554344077"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

原子操作函数，设置后续从VECOUT传输到GM的数据是否执行原子比较，将待拷贝的内容和GM已有内容进行比较，将最小值写入GM。

可通过设置模板参数来设定不同的数据类型。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T>
__aicore__ inline void SetAtomicMin() 
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="8.63%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="91.36999999999999%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="8.63%" headers="mcps1.2.3.1.1 "><p id="p611771320276"><a name="p611771320276"></a><a name="p611771320276"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="91.36999999999999%" headers="mcps1.2.3.1.2 "><p id="p3844958114318"><a name="p3844958114318"></a><a name="p3844958114318"></a>设定不同的数据类型。</p>
<p id="p1109123434013"><a name="p1109123434013"></a><a name="p1109123434013"></a><span id="ph910919343409"><a name="ph910919343409"></a><a name="ph910919343409"></a>Ascend 950PR/Ascend 950DT</span>，支持int8_t/int16_t/half/bfloat16_t/int32_t/float。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

使用完后，建议通过[DisableDmaAtomic](DisableDmaAtomic.md)关闭原子最小操作，以免影响后续相关指令功能。

## 调用示例<a name="section177231425115410"></a>

```
#include "kernel_operator.h"

uint32_t size = 256;
AscendC::LocalTensor<half> dst0Local = queueDst0.DeQue<half>();
AscendC::LocalTensor<half> dst1Local = queueDst1.DeQue<half>();
AscendC::DataCopy(dstGlobal, dst1Local, size);
AscendC::PipeBarrier<PIPE_MTE3>();
AscendC::SetAtomicMin<half>();
AscendC::DataCopy(dstGlobal, dst0Local, size);
queueDst0.FreeTensor(dst0Local);
queueDst1.FreeTensor(dst1Local);
AscendC::DisableDmaAtomic();

每个核的输入数据为: 
Src0: [1,1,1,1,1,...,1] // 256个1
Src1: [2,2,2,2,2,...,2] // 256个2
最终输出数据: [1,1,1,1,1,...,1] // 256个1
```

