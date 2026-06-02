# Fill<a name="ZH-CN_TOPIC_0000002523343692"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.96%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42.04%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.96%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42.04%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

将特定TPosition的LocalTensor初始化为某一具体数值。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
__aicore__ inline void Fill(const LocalTensor<T>& dst, const InitConstValueParams<U>& initConstValueParams)
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
<p id="p14564164211111"><a name="p14564164211111"></a><a name="p14564164211111"></a><span id="ph156494218115"><a name="ph156494218115"></a><a name="ph156494218115"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half/int16_t/uint16_t/bfloat16_t/float/int32_t/uint32_t</p>
</td>
</tr>
<tr id="row118213273213"><td class="cellrowborder" valign="top" width="17.43%" headers="mcps1.2.3.1.1 "><p id="p161827233218"><a name="p161827233218"></a><a name="p161827233218"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="82.57%" headers="mcps1.2.3.1.2 "><p id="p15182229327"><a name="p15182229327"></a><a name="p15182229327"></a>初始化值的数据类型。</p>
<a name="ul17416131483320"></a><a name="ul17416131483320"></a><ul id="ul17416131483320"><li>当dst使用基础数据类型时， U和dst的数据类型T需保持一致，否则编译失败。</li><li>当dst使用<a href="TensorTrait.md">TensorTrait</a>类型时，U和dst的数据类型T的LiteType需保持一致，否则编译失败。</li></ul>
<p id="p472714219338"><a name="p472714219338"></a><a name="p472714219338"></a>最后一个模板参数仅用于上述数据类型检查，用户无需关注。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="17.661766176617665%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="8.670867086708672%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="73.66736673667367%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="17.661766176617665%" headers="mcps1.2.4.1.1 "><p id="p19287714181617"><a name="p19287714181617"></a><a name="p19287714181617"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="8.670867086708672%" headers="mcps1.2.4.1.2 "><p id="p192871614151615"><a name="p192871614151615"></a><a name="p192871614151615"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="73.66736673667367%" headers="mcps1.2.4.1.3 "><p id="p232011551739"><a name="p232011551739"></a><a name="p232011551739"></a>目的操作数，结果矩阵，类型为LocalTensor。</p>
<p id="p46455161649"><a name="p46455161649"></a><a name="p46455161649"></a><span id="ph864519166416"><a name="ph864519166416"></a><a name="ph864519166416"></a>Ascend 950PR/Ascend 950DT</span>， 支持的TPosition为A1/B1。</p>
<p id="p16287121461618"><a name="p16287121461618"></a><a name="p16287121461618"></a>如果TPosition为A1/B1，起始地址需要满足32B对齐；如果TPosition为A2/B2，起始地址需要满足512B对齐。</p>
</td>
</tr>
<tr id="row1075785651510"><td class="cellrowborder" valign="top" width="17.661766176617665%" headers="mcps1.2.4.1.1 "><p id="p486615557145"><a name="p486615557145"></a><a name="p486615557145"></a>InitConstValueParams</p>
</td>
<td class="cellrowborder" valign="top" width="8.670867086708672%" headers="mcps1.2.4.1.2 "><p id="p11287151451610"><a name="p11287151451610"></a><a name="p11287151451610"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="73.66736673667367%" headers="mcps1.2.4.1.3 "><p id="p17376814155615"><a name="p17376814155615"></a><a name="p17376814155615"></a>初始化相关参数，类型为InitConstValueParams。</p>
<p id="p595519531047"><a name="p595519531047"></a><a name="p595519531047"></a>具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p12287014111614"><a name="p12287014111614"></a><a name="p12287014111614"></a>参数说明请参考<a href="#table15780447181917">表3</a>。</p>
<p id="p125181030327"><a name="p125181030327"></a><a name="p125181030327"></a><span id="ph122310317217"><a name="ph122310317217"></a><a name="ph122310317217"></a>Ascend 950PR/Ascend 950DT</span>，支持配置所有参数。</p>
<a name="ul1670213162021"></a><a name="ul1670213162021"></a><ul id="ul1670213162021"><li><span>仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，其他参数配置无效。每次迭代处理固定数据量（512字节），迭代间无间隔。</span></li><li>支持配置所有参数场景下，支持配置迭代次数（repeatTimes）、初始化值（initValue）、每个迭代处理的数据块个数（blockNum）和迭代间间隔（dstGap）。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  InitConstValueParams结构体参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="15.25%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="84.75%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row10780647151919"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p7662195112120"><a name="p7662195112120"></a><a name="p7662195112120"></a>repeatTimes</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p1140916508219"><a name="p1140916508219"></a><a name="p1140916508219"></a>迭代次数。默认值为0。</p>
<a name="ul98581524450"></a><a name="ul98581524450"></a><ul id="ul98581524450"><li><span>仅支持配置迭代次数（repeatTimes）和初始化值（initValue）场景下，repeatTimes∈[0, 255]。</span></li><li>支持配置所有参数场景下，repeatTimes∈[0, 32767] 。</li></ul>
</td>
</tr>
<tr id="row6780947191919"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p1778310122113"><a name="p1778310122113"></a><a name="p1778310122113"></a>blockNum</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p1634012352218"><a name="p1634012352218"></a><a name="p1634012352218"></a>每次迭代初始化的数据块个数，取值范围：blockNum∈[0, 32767] 。默认值为0。</p>
<a name="ul16932118103412"></a><a name="ul16932118103412"></a><ul id="ul16932118103412"><li>dst的位置为A1/B1时，每一个block（数据块）大小是32B；</li><li>dst的位置为A2/B2时，每一个block（数据块）大小是512B。</li></ul>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p012671315216"><a name="p012671315216"></a><a name="p012671315216"></a>dstGap</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p71261136859"><a name="p71261136859"></a><a name="p71261136859"></a>目的操作数前一个迭代结束地址到后一个迭代起始地址之间的距离。</p>
<a name="ul1196025153512"></a><a name="ul1196025153512"></a><ul id="ul1196025153512"><li>dst的位置为A1/B1时，单位是32B；</li><li>dst的位置为A2/B2时，单位是512B。</li></ul>
<p id="p1734053517219"><a name="p1734053517219"></a><a name="p1734053517219"></a>取值范围：dstGap∈[0, 32767] 。默认值为0。</p>
</td>
</tr>
<tr id="row1761285762117"><td class="cellrowborder" valign="top" width="15.25%" headers="mcps1.2.3.1.1 "><p id="p4579316202119"><a name="p4579316202119"></a><a name="p4579316202119"></a>initValue</p>
</td>
<td class="cellrowborder" valign="top" width="84.75%" headers="mcps1.2.3.1.2 "><p id="p193068319220"><a name="p193068319220"></a><a name="p193068319220"></a>初始化的value值，支持的数据类型与dst保持一致。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

```
#include "kernel_operator.h"
uint32 mLength = 16;
uint32 kLength = 16;
template <typename Src0T>
...
TPipe pipe;
TQue<TPosition::A1, 1> qidA1_;
pipe.InitBuffer(qidA1_, 1, mLength  * kLength * sizeof(float));
LocalTensor<float> leftMatrix = qidA1_.template AllocTensor<float>();
Fill(leftMatrix, {1, static_cast<uint16_t>(mLength * kLength * sizeof(Src0T) / 32), 0, 1});
qidA1_.EnQue(leftMatrix);
```

