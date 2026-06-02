# Histograms<a name="ZH-CN_TOPIC_0000002554344099"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1622020982912"><a name="p1622020982912"></a><a name="p1622020982912"></a><span id="ph1522010992915"><a name="ph1522010992915"></a><a name="ph1522010992915"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

对直方图数据进行统计，在目的操作数dstReg的基础数据上加上源操作数srcReg数据的统计结果，包括数据的频率统计和累计统计。

-   频率统计

    如下图所示，dst0和dst1分别统计\[0-127\]和\[128-255\]区间内的数据，dst数据中的第n位数据代表src中n出现的频率，并在dst源数据基础上累加所统计出的数据。

    **图 1**  频率统计<a name="fig1436621813216"></a>  
    <!-- img2text -->
```text
                                   ┌──────────────────────────────────────────────────────────────────────────────→
                                   │
src
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  0  │  1  │  1  │  1  │  4  │ ... │ 128 │ 129 │ 130 │ 131 │ 132 │ ... │ 252 │ 255 │ 255 │ 255 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
                                    └──────────────────────────────────────────────┐
                                                                                   │

dst0
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    1     │    1     │    1     │   ...    │   ...    │    1     │    1     │    1     │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

dst1
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    1     │    1     │    1     │   ...    │   ...    │    1     │    1     │    1     │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

                                               ↓

dst0
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    2     │    4     │    1     │   ...    │   ...    │    2     │    2     │    2     │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘

dst1
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
│    2     │    2     │    2     │   ...    │   ...    │    2     │    2     │    4     │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

说明:
- 图中表示频率统计：dst0 和 dst1 分别统计 `[0-127]` 和 `[128-255]` 区间内的数据。
- src 中出现的值会映射到对应 dst 的第 n 位，并在 dst 原有数据基础上累加。
- 示例中：
  - `src` 里有 `0` 1次、`1` 3次、`4` 1次，因此 `dst0` 对应位置由 `1` 累加后变为 `2、4、1...`
  - `src` 里有 `128、129、130、131、132、252` 各1次，`255` 3次，因此 `dst1` 对应位置累加后末尾变为 `... 2、2、4`

-   累计统计

    如下图所示，dst0和dst1分别统计\[0-127\]和\[128-255\]区间内的数据，dst数据中的第n位数据代表src中0-n内所有数据出现的频率，并在dst源数据基础上累加所统计出的数据。

    **图 2**  累计统计<a name="fig117001116340"></a>  
    <!-- img2text -->
```text
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────→

src
┌─────┬─────┬─────┬─────┬─────┬─────┬──────┬──────┬──────┬──────┬──────┬─────┬──────┬──────┬──────┬──────┐
│  0  │  1  │  1  │  1  │  4  │ ... │ 128  │ 129  │ 130  │ 131  │ 132  │ ... │ 252  │ 255  │ 255  │ 255  │
└─────┴─────┴─────┴─────┴─────┴─────┴──────┴──────┴──────┴──────┴──────┴─────┴──────┴──────┴──────┴──────┘
                                  └──────────────────────────────────────────────────────────────┘

dst0
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│    1    │    1    │    1    │   ...   │   ...   │    1    │    1    │    1    │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

dst1
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│    1    │    1    │    1    │   ...   │   ...   │    1    │    1    │    1    │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

                                               ↓

dst0
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│    2    │    5    │    5    │   ...   │   ...   │   127   │   128   │   129   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

dst1
┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│   130   │   131   │   132   │   ...   │   ...   │   254   │   254   │   257   │
└─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

说明:
- src 中示例数据为：0, 1, 1, 1, 4, ..., 128, 129, 130, 131, 132, ..., 252, 255, 255, 255
- dst0 和 dst1 初始值均为 1
- dst0 统计 [0-127] 区间内的数据，并做累计统计后叠加到原 dst0
- dst1 统计 [128-255] 区间内的数据，并做累计统计后叠加到原 dst1
- 结果示例：
  - dst0: 2, 5, 5, ..., ..., 127, 128, 129
  - dst1: 130, 131, 132, ..., ..., 254, 254, 257

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T = DefaultType, typename U = DefaultType, HistogramsBinType mode, HistogramsType type, typename S, typename V>
__simd_callee__ inline void Histograms(V& dstReg, S& srcReg, MaskReg& mask)
```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="17.94%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="82.06%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>源操作数的数据类型。</p>
<p id="p188929596549"><a name="p188929596549"></a><a name="p188929596549"></a><span id="ph1479120155519"><a name="ph1479120155519"></a><a name="ph1479120155519"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t</p>
</td>
</tr>
<tr id="row18835145716587"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p1383515717581"><a name="p1383515717581"></a><a name="p1383515717581"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><p id="p1775118537244"><a name="p1775118537244"></a><a name="p1775118537244"></a>目的操作数的数据类型。</p>
<p id="p18114175555419"><a name="p18114175555419"></a><a name="p18114175555419"></a><span id="ph811495510548"><a name="ph811495510548"></a><a name="ph811495510548"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t</p>
</td>
</tr>
<tr id="row18128134152619"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p2012917416262"><a name="p2012917416262"></a><a name="p2012917416262"></a><span id="ph1854911301319"><a name="ph1854911301319"></a><a name="ph1854911301319"></a>mode</span></p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><p id="p9905165812547"><a name="p9905165812547"></a><a name="p9905165812547"></a>HistogramsBinType枚举类型，用于控制统计src前半部分还是后半部分的数据。VL长度为256Byte，dst数据类型为uint16_t，一个dst可以存储128个数据，因此需要两个dst。</p>
<a name="ul59931348346"></a><a name="ul59931348346"></a><ul id="ul59931348346"><li>BIN0表示低位模式，统计src中[0-127]范围内的数据写入dst0。</li><li>BIN1表示高位模式，统计src中[128-255]范围内的数据写入dst1。</li></ul>
</td>
</tr>
<tr id="row134244441918"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p916243141912"><a name="p916243141912"></a><a name="p916243141912"></a><span id="ph1324910258318"><a name="ph1324910258318"></a><a name="ph1324910258318"></a>type</span></p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><div class="p" id="p1663311238518"><a name="p1663311238518"></a><a name="p1663311238518"></a>HistogramsType 枚举类型，表示统计模式。<a name="ul36112038046"></a><a name="ul36112038046"></a><ul id="ul36112038046"><li>FREQUENCY：频率统计模式，统计src中[0, 255]每个数的数量。每个dst有128个元素，其中dst0中每个元素对应src中[0, 127]每个元素的累加个数，dst1中每个元素对应src中[128,255]每个元素的累加个数。</li><li>ACCUMULATE：累计统计模式，统计src中x&lt;=0、x&lt;=1、x&lt;=2、x&lt;=3......x&lt;=254、x&lt;=255每个区间的元素个数。每个dst有128个元素，其中dst0中每个元素对应src中[0, 127]每个元素区间累加个数，dst1中每个元素对应src中[128，255]每个元素区间累加个数。</li></ul>
</div>
</td>
</tr>
<tr id="row1237743311380"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p6377173353817"><a name="p6377173353817"></a><a name="p6377173353817"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><p id="p2304311103410"><a name="p2304311103410"></a><a name="p2304311103410"></a>目的操作数RegTensor类型，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
<tr id="row175751956133812"><td class="cellrowborder" valign="top" width="17.94%" headers="mcps1.2.3.1.1 "><p id="p357565618387"><a name="p357565618387"></a><a name="p357565618387"></a>V</p>
</td>
<td class="cellrowborder" valign="top" width="82.06%" headers="mcps1.2.3.1.2 "><p id="p12801441174416"><a name="p12801441174416"></a><a name="p12801441174416"></a>源操作数RegTensor类型，由编译器自动推导，用户不需要填写。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p284425844311"><a name="p284425844311"></a><a name="p284425844311"></a>dstReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p158449584436"><a name="p158449584436"></a><a name="p158449584436"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p6389114616514"><a name="p6389114616514"></a><a name="p6389114616514"></a>目的操作数。</p>
<p id="p66093533169"><a name="p66093533169"></a><a name="p66093533169"></a><span id="ph134278176129"><a name="ph134278176129"></a><a name="ph134278176129"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p6844125874315"><a name="p6844125874315"></a><a name="p6844125874315"></a>srcReg</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p128442058144312"><a name="p128442058144312"></a><a name="p128442058144312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p172083541517"><a name="p172083541517"></a><a name="p172083541517"></a>源操作数。</p>
<p id="p157411581277"><a name="p157411581277"></a><a name="p157411581277"></a><span id="ph890017117407"><a name="ph890017117407"></a><a name="ph890017117407"></a>类型为<a href="RegTensor.md">RegTensor</a>。</span></p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.4.1.1 "><p id="p1484519586432"><a name="p1484519586432"></a><a name="p1484519586432"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p484514581433"><a name="p484514581433"></a><a name="p484514581433"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p11541143920"><a name="p11541143920"></a><a name="p11541143920"></a><span id="ph15776181222"><a name="ph15776181222"></a><a name="ph15776181222"></a>源操作数元素操作的有效指示，详细说明请参考<a href="MaskReg.md">MaskReg</a>。</span></p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section642mcpsimp"></a>

```
template <typename T, typename U>
__simd_vf__ inline void HistogramsVF(__ubuf__ U* dstAddr, __ubuf__ T* srcAddr, uint32_t oneRepeatSize, uint16_t repeatTimes, AscendC::MicroAPI::HistogramsBinType mode, AscendC::MicroAPI::HistogramsType type)
{
    AscendC::MicroAPI::RegTensor<T> srcReg;
    AscendC::MicroAPI::RegTensor<U> dstReg;
    AscendC::MicroAPI::MaskReg mask0 = AscendC::MicroAPI::CreateMask<T>();
    AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::CreateMask<T>();
    for (uint16_t i = 0; i < repeatTimes; ++i){
        AscendC::MicroAPI::LoadAlign(srcReg, srcAddr + oneRepeatSize * i);
        AscendC::MicroAPI::Histograms<T, U, mode, type>(dstReg, srcReg, mask0);
        AscendC::MicroAPI::StoreAlign(dstAddr, dstReg, mask1);
    }
}
```

