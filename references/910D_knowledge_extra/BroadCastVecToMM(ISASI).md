# BroadCastVecToMM\(ISASI\)<a name="ZH-CN_TOPIC_0000002523304888"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>x</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将矢量数据广播到矩阵中，每个数据块中的每16个elements会被连续复制16次；当前支持的数据传输通路：VECIN/VECCALC/VECOUT-\>CO1。

**图 1**  功能示例<a name="fig1730933122314"></a>  
<!-- img2text -->
```
                    blockCount = 1
                    blockLen = 3

src
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│    │   block 0    │      │   block 1    │      │   block 2    │              │
│    └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

          Copy                    Copy                    Copy
           │                       │                       │
           ▼                       ▼                       ▼

dst
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│    │ block 0 - 0  │      │ block 1 - 0  │      │ block 2 - 0  │              │
│    └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│    │ block 0 - 1  │      │ block 1 - 1  │      │ block 2 - 1  │              │
│    └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│    │ block 0 - 2  │      │ block 1 - 2  │      │ block 2 - 2  │              │
│    └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
│           ...                     ...                     ...                 │
│                                                                              │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐              │
│    │ block 0 - 15 │      │ block 1 - 15 │      │ block 2 - 15 │              │
│    └──────────────┘      └──────────────┘      └──────────────┘              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

说明:
- src 中有 3 个连续数据块：block 0、block 1、block 2。
- 每个 src block 经过一次 Copy，在 dst 中连续复制为 16 份：
  - block 0 → block 0 - 0 ～ block 0 - 15
  - block 1 → block 1 - 0 ～ block 1 - 15
  - block 2 → block 2 - 0 ～ block 2 - 15
- 图中参数保留：
  - blockCount = 1
  - blockLen = 3

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, typename U>
__aicore__ inline void BroadCastVecToMM(const LocalTensor<T> &dst, const LocalTensor<U> &src, const int32_t blockCount, const uint8_t blockLen, const uint8_t srcGap, const uint8_t dstGap)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="17.43%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.57%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>dst的数据类型。</p>
</td>
</tr>
<tr id="row118213273213"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p161827233218"><a name="p161827233218"></a><a name="p161827233218"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p56871237135217"><a name="p56871237135217"></a><a name="p56871237135217"></a>src的数据类型。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table284254116544"></a>
<table><thead align="left"><tr id="row10842204115543"><th class="cellrowborder" valign="top" width="18.5018501850185%" id="mcps1.2.4.1.1"><p id="p565185175414"><a name="p565185175414"></a><a name="p565185175414"></a><strong id="b16565112547"><a name="b16565112547"></a><a name="b16565112547"></a>参数名称</strong></p>
</th>
<th class="cellrowborder" valign="top" width="13.67136713671367%" id="mcps1.2.4.1.2"><p id="p7651751165417"><a name="p7651751165417"></a><a name="p7651751165417"></a><strong id="b1365351135416"><a name="b1365351135416"></a><a name="b1365351135416"></a>类型</strong></p>
</th>
<th class="cellrowborder" valign="top" width="67.82678267826783%" id="mcps1.2.4.1.3"><p id="p36519513546"><a name="p36519513546"></a><a name="p36519513546"></a><strong id="b16545185411"><a name="b16545185411"></a><a name="b16545185411"></a>说明</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row1484210419547"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p19287714181617"><a name="p19287714181617"></a><a name="p19287714181617"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p192871614151615"><a name="p192871614151615"></a><a name="p192871614151615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p114820562258"><a name="p114820562258"></a><a name="p114820562258"></a>目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。</p>
<p id="p122771447172412"><a name="p122771447172412"></a><a name="p122771447172412"></a><span id="ph14913134718242"><a name="ph14913134718242"></a><a name="ph14913134718242"></a>LocalTensor的起始地址需要256个元素对齐。</span></p>
<p id="p16287121461618"><a name="p16287121461618"></a><a name="p16287121461618"></a>支持的数据类型为：half/float/int32_t。</p>
</td>
</tr>
<tr id="row16259173555"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p327217511273"><a name="p327217511273"></a><a name="p327217511273"></a>源操作数，输入矢量，类型为LocalTensor，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p0287191420164"><a name="p0287191420164"></a><a name="p0287191420164"></a>支持的数据类型需要与dst一致。</p>
</td>
</tr>
<tr id="row14842841145419"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p1648712150175"><a name="p1648712150175"></a><a name="p1648712150175"></a>blockCount</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p19487171515178"><a name="p19487171515178"></a><a name="p19487171515178"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p3487131516175"><a name="p3487131516175"></a><a name="p3487131516175"></a>指定该指令包含的连续广播数据块个数，取值范围：blockCount∈[1, 255]。</p>
</td>
</tr>
<tr id="row23311816165517"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p1728791441620"><a name="p1728791441620"></a><a name="p1728791441620"></a>blockLen</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>指定该指令每个连续广播数据块长度，单位为16个elements。取值范围：blockLen∈[1, 255]。</p>
</td>
</tr>
<tr id="row11842164195417"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p6694183118501"><a name="p6694183118501"></a><a name="p6694183118501"></a>srcGap</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p46231641144814"><a name="p46231641144814"></a><a name="p46231641144814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p19780547162614"><a name="p19780547162614"></a><a name="p19780547162614"></a>源操作数，相邻连续数据块的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为datablock(32Bytes)。</p>
</td>
</tr>
<tr id="row18843104116541"><td class="cellrowborder" valign="top" width="18.5018501850185%" headers="mcps1.2.4.1.1 "><p id="p14691531145016"><a name="p14691531145016"></a><a name="p14691531145016"></a>dstGap</p>
</td>
<td class="cellrowborder" valign="top" width="13.67136713671367%" headers="mcps1.2.4.1.2 "><p id="p41147426481"><a name="p41147426481"></a><a name="p41147426481"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="67.82678267826783%" headers="mcps1.2.4.1.3 "><p id="p36901631125020"><a name="p36901631125020"></a><a name="p36901631125020"></a>目的操作数，相邻连续数据块间的间隔（前面一个数据块的尾与后面数据块的头的间隔），单位为256个elements。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

本示例中，输入bias形状为\[1, 32\]，输出c的形状为\[32, 32\]，格式为Nz。

**图 2**  调用示例图<a name="fig496292825418"></a>  
<!-- img2text -->
```text
bias
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
│ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │ 11 │ 12 │ 13 │ 14 │ 15 │ 16 │ 17 │ 18 │ 19 │ 20 │ 21 │ 22 │ 23 │ 24 │ 25 │ 26 │ 27 │ 28 │ 29 │ 30 │ 31 │ 32 │
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┴────┘
                                              │
                                              │
                                              ↓

c
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
│  1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32 │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

红色标记路径:
↗ 左下 → 中上(约第16列) → ↓ 到底部 → ↗ 右上
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

```
#include "kernel_operator.h"

constexpr int32_t INPUT_LENGTH = 32;
constexpr int32_t OUTPUT_LENGTH = 32 * 32;
class KernelBroadCastVecToMM {
public:
    __aicore__ inline KernelBroadCastVecToMM() {}
    __aicore__ inline void Init(GM_ADDR bias, GM_ADDR c)
    {
        biasGm.SetGlobalBuffer((__gm__ float*)bias);
        cGm.SetGlobalBuffer((__gm__ float*)c);
        pipe.InitBuffer(inQueueBias, 1, INPUT_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueueC, 1, OUTPUT_LENGTH * sizeof(float));
        pipe.InitBuffer(broadCastQue, 1, OUTPUT_LENGTH * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        BroadCast();
        Aggregate();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<float> biasLocal = inQueueBias.AllocTensor<float>();
        AscendC::DataCopy(biasLocal, biasGm, INPUT_LENGTH);
        inQueueBias.EnQue(biasLocal);
    }
    __aicore__ inline void BroadCast()
    {
        AscendC::LocalTensor<float> biasLocal = inQueueBias.DeQue<float>();
        AscendC::LocalTensor<float> brcLocal = broadCastQue.AllocTensor<float>();
        AscendC::BroadCastVecToMM(brcLocal, biasLocal, 2, 1, 0, 1);
        AscendC::BroadCastVecToMM(brcLocal[16 * 16], biasLocal, 2, 1, 0, 1);
        broadCastQue.EnQue<float>(brcLocal);
        inQueueBias.FreeTensor(biasLocal);
    }
    __aicore__ inline void Aggregate()
    {
        AscendC::LocalTensor<float> brcLocal = broadCastQue.DeQue<float>();
        AscendC::LocalTensor<float> cLocal = outQueueC.AllocTensor<float>();

        AscendC::DataCopyParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = 4;
        AscendC::DataCopyEnhancedParams enhancedParams;
        enhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy(cLocal, brcLocal, dataCopyParams, enhancedParams);

        outQueueC.EnQue<float>(cLocal);
        broadCastQue.FreeTensor(brcLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<float> cLocal = outQueueC.DeQue<float>();
        AscendC::DataCopy(cGm, cLocal, OUTPUT_LENGTH);
        outQueueC.FreeTensor(cLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueBias;
    AscendC::TQueBind<AscendC::TPosition::VECIN, TPosition::CO1, 1> broadCastQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueC;
    AscendC::GlobalTensor<float> biasGm;
    AscendC::GlobalTensor<float> cGm;
};

extern "C" __global__ __aicore__ void broadcast_vec_to_mm_custom(GM_ADDR bias, GM_ADDR c)
{
    KernelBroadCastVecToMM op;
    op.Init(bias, c);
    op.Process();
}
```

