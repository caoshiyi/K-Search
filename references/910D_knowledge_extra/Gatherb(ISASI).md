# Gatherb\(ISASI\)<a name="ZH-CN_TOPIC_0000002554343625"></a>

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

## 功能说明<a name="section17600329101418"></a>

给定一个输入的张量和一个地址偏移张量，本接口根据偏移地址按照DataBlock的粒度将输入张量收集到结果张量中。

<!-- img2text -->
```text
offset
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│ offset[0]│ offset[1]│ offset[2]│ offset[3]│ offset[4]│ offset[5]│ offset[6]│ offset[7]│
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                                       │
                                       ▼
src的基地址 ─────────────────────────→  ⊕
                                       │
                                       ▼

┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│  addr0   │  addr1   │  addr2   │  addr3   │  addr4   │  addr5   │  addr6   │  addr7   │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

src
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│DataBlock2│DataBlock0│DataBlock4│DataBlock5│DataBlock1│DataBlock6│DataBlock3│DataBlock7│
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
                                       │
                                       ▼
                                     Gatherb
                                       │
                                       ▼

dst
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│DataBlock0│DataBlock1│DataBlock2│DataBlock3│DataBlock4│DataBlock5│DataBlock6│DataBlock7│
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

说明:
- `offset[i]` 与 `src的基地址` 相加生成 `addri`
- `addr0~addr7` 对应从 `src` 中按地址收集的数据块，最终通过 `Gatherb` 写入 `dst`
- 图中可读出的收集结果顺序为:
  - `addr0 → src` 中的 `DataBlock2 → dst` 的 `DataBlock0`
  - `addr1 → src` 中的 `DataBlock0 → dst` 的 `DataBlock1`
  - `addr2 → src` 中的 `DataBlock4 → dst` 的 `DataBlock2`
  - `addr3 → src` 中的 `DataBlock5 → dst` 的 `DataBlock3`
  - `addr4 → src` 中的 `DataBlock1 → dst` 的 `DataBlock4`
  - `addr5 → src` 中的 `DataBlock6 → dst` 的 `DataBlock5`
  - `addr6 → src` 中的 `DataBlock3 → dst` 的 `DataBlock6`
  - `addr7 → src` 中的 `DataBlock7 → dst` 的 `DataBlock7`

## 函数原型<a name="section15660625202219"></a>

```
template <typename T>
__aicore__ inline void Gatherb(const LocalTensor<T>& dst, const LocalTensor<T>& src0, const LocalTensor<uint32_t>& offset, const uint8_t repeatTime, const GatherRepeatParams& repeatParams)
```

## 参数说明<a name="section1619484392111"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="14.729999999999999%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.27%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="14.729999999999999%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001429830437_p5835457165816"><a name="zh-cn_topic_0000001429830437_p5835457165816"></a><a name="zh-cn_topic_0000001429830437_p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.27%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>操作数数据类型。</p>
<p id="p8391415163219"><a name="p8391415163219"></a><a name="p8391415163219"></a><span id="ph13391315113210"><a name="ph13391315113210"></a><a name="ph13391315113210"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/int16_t/uint16_t/int32_t/uint32_t/half/float/bfloat16_t/uint64_t/int64_t</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table917mcpsimp"></a>
<table><thead align="left"><tr id="row923mcpsimp"><th class="cellrowborder" valign="top" width="15.02%" id="mcps1.2.4.1.1"><p id="p925mcpsimp"><a name="p925mcpsimp"></a><a name="p925mcpsimp"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10%" id="mcps1.2.4.1.2"><p id="p927mcpsimp"><a name="p927mcpsimp"></a><a name="p927mcpsimp"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.98%" id="mcps1.2.4.1.3"><p id="p929mcpsimp"><a name="p929mcpsimp"></a><a name="p929mcpsimp"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row930mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p2925016172518"><a name="p2925016172518"></a><a name="p2925016172518"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p199251416112517"><a name="p199251416112517"></a><a name="p199251416112517"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p12471855121311"><a name="p12471855121311"></a><a name="p12471855121311"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p78204190123"><a name="p78204190123"></a><a name="p78204190123"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row937mcpsimp"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p3926171610253"><a name="p3926171610253"></a><a name="p3926171610253"></a>src0</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p4926121682518"><a name="p4926121682518"></a><a name="p4926121682518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p142923151419"><a name="p142923151419"></a><a name="p142923151419"></a>源操作数。</p>
<p id="p15547145121412"><a name="p15547145121412"></a><a name="p15547145121412"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p086691541612"><a name="p086691541612"></a><a name="p086691541612"></a><span id="ph12866151511161"><a name="ph12866151511161"></a><a name="ph12866151511161"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1484485824312"><a name="p1484485824312"></a><a name="p1484485824312"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row18516194102416"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p85164422415"><a name="p85164422415"></a><a name="p85164422415"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p10516104162418"><a name="p10516104162418"></a><a name="p10516104162418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p7545916142117"><a name="p7545916142117"></a><a name="p7545916142117"></a>每个datablock在源操作数中对应的地址偏移。</p>
<p id="p15812123272020"><a name="p15812123272020"></a><a name="p15812123272020"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p18431833185"><a name="p18431833185"></a><a name="p18431833185"></a><span id="ph359873013185"><a name="ph359873013185"></a><a name="ph359873013185"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p1331252141512"><a name="p1331252141512"></a><a name="p1331252141512"></a>该偏移量是相对于src0的基地址而言的。每个元素值要大于等于0，单位为字节；且需要保证偏移后的地址满足32字节对齐。</p>
</td>
</tr>
<tr id="row4736114341415"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p47360437147"><a name="p47360437147"></a><a name="p47360437147"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p773619438142"><a name="p773619438142"></a><a name="p773619438142"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p13484943122219"><a name="p13484943122219"></a><a name="p13484943122219"></a>重复迭代次数，每次迭代完成8个datablock的数据收集，数据范围：repeatTime∈（0,255]。</p>
</td>
</tr>
<tr id="row20730549195712"><td class="cellrowborder" valign="top" width="15.02%" headers="mcps1.2.4.1.1 "><p id="p1473034913577"><a name="p1473034913577"></a><a name="p1473034913577"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="10%" headers="mcps1.2.4.1.2 "><p id="p9730649125711"><a name="p9730649125711"></a><a name="p9730649125711"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.98%" headers="mcps1.2.4.1.3 "><p id="p13186514145119"><a name="p13186514145119"></a><a name="p13186514145119"></a>用于控制指令迭代的相关参数。</p>
<p id="p271423619448"><a name="p271423619448"></a><a name="p271423619448"></a>类型为GatherRepeatParams，具体定义可参考<span id="ph156517114113"><a name="ph156517114113"></a><a name="ph156517114113"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_gather.h。<span id="ph206514115119"><a name="ph206514115119"></a><a name="ph206514115119"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p197301549165720"><a name="p197301549165720"></a><a name="p197301549165720"></a>其中dstBlkStride、dstRepStride支持用户配置，参数说明参考<a href="Gatherb(ISASI).md#table2166248155314">表3</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  GatherRepeatParams结构体参数说明

<a name="table2166248155314"></a>
<table><thead align="left"><tr id="row5166144885316"><th class="cellrowborder" valign="top" width="14.510000000000002%" id="mcps1.2.3.1.1"><p id="p1116674814535"><a name="p1116674814535"></a><a name="p1116674814535"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="85.49%" id="mcps1.2.3.1.2"><p id="p216614815538"><a name="p216614815538"></a><a name="p216614815538"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1716618489531"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.3.1.1 "><p id="p199988584557"><a name="p199988584557"></a><a name="p199988584557"></a>dstBlkStride</p>
</td>
<td class="cellrowborder" valign="top" width="85.49%" headers="mcps1.2.3.1.2 "><p id="p69921747165514"><a name="p69921747165514"></a><a name="p69921747165514"></a>单次迭代内，矢量目的操作数不同datablock间的地址步长。</p>
</td>
</tr>
<tr id="row18688164155717"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.3.1.1 "><p id="p489432135811"><a name="p489432135811"></a><a name="p489432135811"></a>dstRepStride</p>
</td>
<td class="cellrowborder" valign="top" width="85.49%" headers="mcps1.2.3.1.2 "><p id="p1568874112576"><a name="p1568874112576"></a><a name="p1568874112576"></a>相邻迭代间，矢量目的操作数相同datablock间的地址步长。</p>
</td>
</tr>
<tr id="row416744895313"><td class="cellrowborder" valign="top" width="14.510000000000002%" headers="mcps1.2.3.1.1 "><p id="p332710615819"><a name="p332710615819"></a><a name="p332710615819"></a>blockNumber</p>
</td>
<td class="cellrowborder" rowspan="7" valign="top" width="85.49%" headers="mcps1.2.3.1.2 "><p id="p1601381273"><a name="p1601381273"></a><a name="p1601381273"></a>预留参数。为后续的功能做保留，开发者暂时无需关注，使用默认值即可。</p>
</td>
</tr>
<tr id="row516717487537"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p953516579556"><a name="p953516579556"></a><a name="p953516579556"></a>src0BlkStride</p>
</td>
</tr>
<tr id="row191689487531"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p7880131485719"><a name="p7880131485719"></a><a name="p7880131485719"></a>src1BlkStride</p>
</td>
</tr>
<tr id="row9168148155312"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p3513164125610"><a name="p3513164125610"></a><a name="p3513164125610"></a>src0RepStride</p>
</td>
</tr>
<tr id="row1045731112566"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p1345718114561"><a name="p1345718114561"></a><a name="p1345718114561"></a>src1RepStride</p>
</td>
</tr>
<tr id="row1824291417569"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p359183495619"><a name="p359183495619"></a><a name="p359183495619"></a>repeatStrideMode</p>
</td>
</tr>
<tr id="row1946916160567"><td class="cellrowborder" valign="top" headers="mcps1.2.3.1.1 "><p id="p15634427135614"><a name="p15634427135614"></a><a name="p15634427135614"></a>strideSizeMode</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section11276201527"></a>

```
#include "kernel_operator.h"
AscendC::TPipe tpipe;
AscendC::TQue<AscendC::TPosition::VECIN, 2> vecIn;
AscendC::TQue<AscendC::TPosition::VECIN, 2> vecOffset;
AscendC::TQue<AscendC::TPosition::VECOUT, 2> vecOut;

uint32_t bufferLen = 0;

uint32_t len = 128;
bufferLen = len;
tpipe.InitBuffer(vecIn, 2, bufferLen * sizeof(uint16_t));
tpipe.InitBuffer(vecOffset, 2, 8 * sizeof(uint32_t));
tpipe.InitBuffer(vecOut, 2, bufferLen * sizeof(uint16_t));

auto x_buf = vecIn.AllocTensor<uint16_t>();
auto offset_buf = vecOffset.AllocTensor<uint32_t>();
AscendC::DataCopy(x_buf, x_gm[index * bufferLen], bufferLen);
AscendC::DataCopy(offset_buf, offset_gm[0], 8);
vecIn.EnQue(x_buf);
vecOffset.EnQue(offset_buf);

auto y_buf = vecOut.DeQue<uint16_t>();
AscendC::DataCopy(y_gm[index * bufferLen], y_buf, bufferLen);
vecOut.FreeTensor(y_buf);

auto x_buf = vecIn.DeQue<uint16_t>();
auto offset_buf = vecOffset.DeQue<uint32_t>();
auto y_buf = vecOut.AllocTensor<uint16_t>();
AscendC::GatherRepeatParams params{1, 8};
uint8_t repeatTime = bufferLen * sizeof(uint16_t) / 256;
AscendC::Gatherb<uint16_t>(y_buf, x_buf, offset_buf, repeatTime, params);
vecIn.FreeTensor(x_buf);
vecOffset.FreeTensor(offset_buf);
vecOut.EnQue(y_buf);
```

结果示例：

```
输入数据(offsetLocal): [224 192 160 128 96 64 32 0]
输入数据(srcLocal): [0 1 2 3 4 5 6 7 ... 120 121 122 123 124 125 126 127]
输出数据(dstGlobal):[
112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 
96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111
... 
0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
]
```

