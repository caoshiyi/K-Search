# Select<a name="ZH-CN_TOPIC_0000002523343522"></a>

## 产品支持情况<a name="section1586581915393"></a>

<a name="table169596713360"></a>
<table><thead align="left"><tr id="row129590715369"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p17959971362"><a name="p17959971362"></a><a name="p17959971362"></a><span id="ph895914718367"><a name="ph895914718367"></a><a name="ph895914718367"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p89594763612"><a name="p89594763612"></a><a name="p89594763612"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row18959673369"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p1595910763613"><a name="p1595910763613"></a><a name="p1595910763613"></a><span id="ph1595918753613"><a name="ph1595918753613"></a><a name="ph1595918753613"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p1695957133611"><a name="p1695957133611"></a><a name="p1695957133611"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

给定两个源操作数src0和src1，根据maskTensor相应位置的值（非bit位）选取元素，得到目的操作数dst。选择的规则为：当Mask的值为0时，从src0中选取，否则从src1选取。

**该接口支持多维Shape，需满足maskTensor和源操作数Tensor的前轴（非尾轴）元素个数相同，且maskTensor尾轴元素个数大于等于源操作数尾轴元素个数，maskTensor多余部分丢弃不参与计算。**

-   **maskTensor尾轴需32字节对齐且元素个数为16的倍数。**
-   **源操作数Tensor尾轴需32字节对齐。**

如下图样例，源操作数src0为Tensor，shape为\(2,16\)，数据类型为half，尾轴长度满足32字节对齐；源操作数src1为scalar，数据类型为half；maskTensor的数据类型为bool，为满足对齐要求shape为\(2,32\)，仅有图中蓝色部分的mask掩码生效，灰色部分不参与计算。输出目的操作数dstTensor如下图所示。

<!-- img2text -->
```text
src0Tensor:  shape(2,16)
datatype: half

┌───┬───┬────┬────┬────┬────┐
│ 1 │ 2 │ ...│ ...│ 15 │ 16 │
├───┼───┼────┼────┼────┼────┤
│ 1 │ 2 │ ...│ ...│ 15 │ 16 │
└───┴───┴────┴────┴────┴────┘

                          src1Scalar
                          datatype: half
                          ┌───┐
                          │ 0 │
                          └───┘

maskTensor:  shape(2,32)
datatype: bool

┌───┬───┬────┬────┬───┬───┬────┬────┬───┬───┐
│ 1 │ 0 │ ...│ ...│ 1 │ 0 │ ...│ ...│ 1 │ 0 │
├───┼───┼────┼────┼───┼───┼────┼────┼───┼───┤
│ 1 │ 0 │ ...│ ...│ 1 │ 0 │ ...│ ...│ 1 │ 0 │
└───┴───┴────┴────┴───┴───┴────┴────┴───┴───┘
                                   ↑
                                   │
                    ┌────────────────────────────────────┐
                    │ 尾轴需要32字节对齐                │
                    │ 且元素个数为16的倍数。            │
                    │ 灰色部分是为了保证                │
                    │ 对齐多开辟的空间，不              │
                    │ 参与计算。                        │
                    └────────────────────────────────────┘

dstTensor

┌───┬───┬────┬────┬───┬────┐
│ 0 │ 2 │ ...│ ...│ 0 │ 16 │
├───┼───┼────┼────┼───┼────┤
│ 0 │ 2 │ ...│ ...│ 0 │ 16 │
└───┴───┴────┴────┴───┴────┘
```

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[m, k1\]的source输入Tensor，shape为\[m, k2\]的mask Tensor为例，描述Select高阶API内部算法框图，如下图所示。

**图 1**  Select算法框图<a name="fig46711722165110"></a>  
<!-- img2text -->
```text
                         ┌────────────┐
                         │ mask[m, k2]│
                         └────────────┘
                                │
                                ↓
      ┌──────────────────────────────────────────────────────────────┐
      │                                                              │
      │                     ┌───────────┐                            │
      │                     │  k1 = k2  │────── False ──────┐        │
      │                     └───────────┘                    │        │
      │                          │ True                      ↓        │
      │                          │                    ┌──────────────────────────┐
      │                          │                    │       GatherMask         │
      │                          │                    │   [m, k2] -> [m, k1]     │
      │                          │                    └──────────────────────────┘
      │                          │                               │
      │                          └───────────────←───────────────┘
      │                                          │
      │                                          ↓
      │                           ┌──────────────────────────┐
      │                           │        Cast(mask)        │
      │                           │      U->half[m, k1]      │
      │                           └──────────────────────────┘
      │                                          │
      │                                          ↓
      │                           ┌──────────────────────────┐
      │                           │      Compare(EQ 0)       │
      │                           │      cmpmask([m * k1])   │
      │                           └──────────────────────────┘
      │                                          │
      │                                          ↓
┌──────────┐                                    ┌──────────────────────────┐
│  scalar  │──────────────────────────────────→│  Select(src, scalar,     │
└──────────┘                                    │         cmpmask)         │
┌──────────┐──────────────────────────────────→│                          │
│src[m, k1]│                                    └──────────────────────────┘
└──────────┘                                                  │
      │                                                       │
      └───────────────────────────────────────────────────────┘
                                                              ↓
                                                     ┌───────────────┐
                                                     │ output[m, k1] │
                                                     └───────────────┘


图示:
输入\输出Tensor    ┌──────────┐
                  │          │
                  └──────────┘

vector计算       ┌──────────┐
                 │          │
                 └──────────┘

条件判断             ◇

数据流向             ───→
```

计算过程分为如下几步，均在Vector上进行：

1.  GatherMask步骤：如果k1, k2不相等，则根据src的shape\[m, k1\]，对输入mask\[m, k2\]通过GatherMask进行reduce计算，使得mask的k轴多余部分被舍去，shape转换为\[m, k1\]；
2.  Cast步骤：将上一步的mask结果cast成half类型；
3.  Compare步骤：使用Compare接口将上一步的mask结果与0进行比较，得到cmpmask结果；
4.  Select步骤：根据cmpmask的结果，选择srcTensor相应位置的值或者scalar值，输出Output。

**图 2**  Select算法框图<a name="fig299811162012"></a>  
<!-- img2text -->
```text
                           ┌─────────────┐
                           │ mask[m, k2] │
                           └──────┬──────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                             │
│                         ┌─────────────────────┐                             │
│                         │      int i = 0      │                             │
│                         └──────────┬──────────┘                             │
│                                    │                                        │
│                                    ▼                                        │
│                               ╱─────────╲                                   │
│                              ╱   i < m   ╲                                  │
│                             ╱─────────────╲                                 │
│                             └──────┬──────┘                                 │
│                                    │                                        │
│                         否 ────────┘                                        │
│                        │                                                    │
│                        ▼                                                    │
│                  ┌──────────────┐                                           │
│                  │ output[m, k1]│                                           │
│                  └──────────────┘                                           │
│                                                                             │
│                                    │                                        │
│                                    │ 是                                     │
│                                    ▼                                        │
│                        ┌─────────────────────────┐                          │
│                        │     Compare(EQ 0)       │                          │
│                        │      cmpmask[(k1)]      │                          │
│                        └──────────┬──────────────┘                          │
│                                   │                                         │
│                                   ▼                                         │
│ scalar ───────┐        ┌─────────────────────────┐                          │
│               ├───────►│ Select(src, scalar,     │──────────────┐           │
│ src[m, k1] ───┘        │      cmpmask, k1)       │              │           │
│                        └─────────────────────────┘              │           │
│                                                                 │           │
│                                                                 ▼           │
│                                                       ┌────────────────┐    │
│                                                       │      i++       │    │
│                                                       └────────┬───────┘    │
│                                                                │            │
│                                                                └──────┐     │
│                                                                       │     │
│                                                                       ▼     │
│                                                                  (回到 i < m)│
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


图示:
输入输出Tensor    ──┌──────────────┐
                  │              │
                  └──────────────┘

vector计算       ──┌──────────────┐
                  │              │
                  └──────────────┘

条件判断         ──╱──────────────╲
                   ╲──────────────╱

数据流向         ──→
```

计算过程在Vector上进行，循环m次，每次对k1个元素进行如下操作：

1.  Compare步骤：使用Compare接口将mask值与0进行比较，得到cmpmask结果；
2.  Select步骤：根据cmpmask的结果，选择srcTensor相应位置的值或者scalar值，输出Output。

## 函数原型<a name="section620mcpsimp"></a>

-   src0为srcTensor（tensor类型），src1为srcScalar（scalar类型）

    ```
    template <typename T, typename U, bool isReuseMask = true>
    __aicore__ inline void Select(const LocalTensor<T>& dst, const LocalTensor<T>& src0, T src1, const LocalTensor<U>& mask, const LocalTensor<uint8_t>& sharedTmpBuffer, const SelectWithBytesMaskShapeInfo& info)
    ```

-   src0为srcScalar（scalar类型），src1为srcTensor（tensor类型）

    ```
    template <typename T, typename U, bool isReuseMask = true>
    __aicore__ inline void Select(const LocalTensor<T>& dst, T src0, const LocalTensor<T>& src1, const LocalTensor<U>& mask, const LocalTensor<uint8_t>& sharedTmpBuffer, const SelectWithBytesMaskShapeInfo& info)
    ```

该接口需要额外的临时空间来存储计算过程中的中间变量。临时空间需要开发者**申请并通过sharedTmpBuffer入参传入**。临时空间大小BufferSize的获取方式如下：通过[GetSelectMaxMinTmpSize](GetSelectMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p125969172719"><a name="p125969172719"></a><a name="p125969172719"></a>操作数的数据类型。</p>
<p id="p1526517121192"><a name="p1526517121192"></a><a name="p1526517121192"></a><span id="ph1928231818522"><a name="ph1928231818522"></a><a name="ph1928231818522"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row197562193260"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p18961115352610"><a name="p18961115352610"></a><a name="p18961115352610"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1287421219275"><a name="p1287421219275"></a><a name="p1287421219275"></a>掩码Tensor mask的数据类型。</p>
<p id="p135053011229"><a name="p135053011229"></a><a name="p135053011229"></a><span id="ph691515191215"><a name="ph691515191215"></a><a name="ph691515191215"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：bool、int8_t、uint8_t、int16_t、uint16_t、int32_t、uint32_t。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseMask</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1118159145815"><a name="p1118159145815"></a><a name="p1118159145815"></a>是否允许修改maskTensor。默认为true。</p>
<p id="p6185945814"><a name="p6185945814"></a><a name="p6185945814"></a>取值为true时，仅在maskTensor尾轴元素个数和srcTensor尾轴元素个数不同的情况下，maskTensor可能会被修改；其余场景，maskTensor不会修改。</p>
<p id="p7186915589"><a name="p7186915589"></a><a name="p7186915589"></a>取值为false时，任意场景下，maskTensor均不会修改，但可能会需要更多的临时空间。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="16.38163816381638%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.261126112611262%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.35723572357236%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p19576531173410"><a name="p19576531173410"></a><a name="p19576531173410"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p16576163119347"><a name="p16576163119347"></a><a name="p16576163119347"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p134985910344"><a name="p134985910344"></a><a name="p134985910344"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row69959512410"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p165761231123417"><a name="p165761231123417"></a><a name="p165761231123417"></a>src0(srcTensor)</p>
<p id="p490520575412"><a name="p490520575412"></a><a name="p490520575412"></a>src1(srcTensor)</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p757693163410"><a name="p757693163410"></a><a name="p757693163410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p17251202810350"><a name="p17251202810350"></a><a name="p17251202810350"></a>源操作数。<strong id="b102138521310"><a name="b102138521310"></a><a name="b102138521310"></a>源操作数Tensor尾轴需32字节对齐。</strong></p>
<p id="p14953193173517"><a name="p14953193173517"></a><a name="p14953193173517"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p10576203116349"><a name="p10576203116349"></a><a name="p10576203116349"></a>src1(srcScalar)</p>
<p id="p13740434650"><a name="p13740434650"></a><a name="p13740434650"></a>src0(srcScalar)</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p1157693123417"><a name="p1157693123417"></a><a name="p1157693123417"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p19262048174314"><a name="p19262048174314"></a><a name="p19262048174314"></a>源操作数。类型为scalar。</p>
</td>
</tr>
<tr id="row1495634115010"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p10974181411356"><a name="p10974181411356"></a><a name="p10974181411356"></a>mask</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p1797491412352"><a name="p1797491412352"></a><a name="p1797491412352"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p178554582519"><a name="p178554582519"></a><a name="p178554582519"></a>掩码Tensor。用于描述如何选择srcTensor和srcScalar之间的值。<strong id="b42134501317"><a name="b42134501317"></a><a name="b42134501317"></a>maskTensor尾轴需32字节对齐且元素个数为16的倍数。</strong></p>
<a name="ul1565216916491"></a><a name="ul1565216916491"></a><ul id="ul1565216916491"><li><strong id="b460384916567"><a name="b460384916567"></a><a name="b460384916567"></a>src0为srcTensor（tensor类型），src1为srcScalar（scalar类型）</strong><p id="p624565125611"><a name="p624565125611"></a><a name="p624565125611"></a>若mask的值为0，选择srcTensor相应的值放入dstLocal，否则选择srcScalar的值放入dstLocal。</p>
</li><li><strong id="b1623350570"><a name="b1623350570"></a><a name="b1623350570"></a>src0为srcScalar（scalar类型），src1为srcTensor（tensor类型）</strong><p id="p184402041195716"><a name="p184402041195716"></a><a name="p184402041195716"></a>若mask的值为0，选择srcScalar的值放入dstLocal，否则选择srcTensor相应的值放入dstLocal。</p>
</li></ul>
</td>
</tr>
<tr id="row769135514428"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p175358462154"><a name="p175358462154"></a><a name="p175358462154"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p10535746191515"><a name="p10535746191515"></a><a name="p10535746191515"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p14183115914443"><a name="p14183115914443"></a><a name="p14183115914443"></a>该API用于计算的临时空间，所需空间大小根据<a href="GetSelectMaxMinTmpSize.md">GetSelectMaxMinTmpSize</a>获取。</p>
</td>
</tr>
<tr id="row1514910082412"><td class="cellrowborder" valign="top" width="16.38163816381638%" headers="mcps1.2.4.1.1 "><p id="p1014913018244"><a name="p1014913018244"></a><a name="p1014913018244"></a>info</p>
</td>
<td class="cellrowborder" valign="top" width="11.261126112611262%" headers="mcps1.2.4.1.2 "><p id="p201491700243"><a name="p201491700243"></a><a name="p201491700243"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.35723572357236%" headers="mcps1.2.4.1.3 "><p id="p1874811478502"><a name="p1874811478502"></a><a name="p1874811478502"></a>描述SrcTensor和maskTensor的shape信息。SelectWithBytesMaskShapeInfo类型，定义如下：</p>
<a name="screen1998174885413"></a><a name="screen1998174885413"></a><pre class="screen" codetype="Cpp" id="screen1998174885413">struct SelectWithBytesMaskShapeInfo {
__aicore__ SelectShapeInfo(){};
uint32_t firstAxis = 0;    
uint32_t srcLastAxis = 0; 
uint32_t maskLastAxis = 0;
};</pre>
<a name="ul741115095118"></a><a name="ul741115095118"></a><ul id="ul741115095118"><li>firstAxis：srcLocal/maskTensor的前轴元素个数。</li><li>srcLastAxis：srcLocal的尾轴元素个数。</li><li>maskLastAxis：maskTensor的尾轴元素个数。</li></ul>
<p id="p733553716329"><a name="p733553716329"></a><a name="p733553716329"></a>注意：</p>
<a name="ul5343143718326"></a><a name="ul5343143718326"></a><ul id="ul5343143718326"><li>需要满足srcTensor和maskTensor的前轴元素个数相同，均为firstAxis。</li><li>需要满足firstAxis * srcLastAxis = srcTensor.GetSize() ；firstAxis * maskLastAxis = maskTensor.GetSize()。</li><li>maskTensor尾轴的元素个数大于等于srcTensor尾轴的元素个数，计算时会丢弃maskTensor多余部分，不参与计算。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section198548421851"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数与目的操作数可以复用。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   maskTensor尾轴元素个数和源操作数尾轴元素个数不同的情况下， maskTensor的数据有可能被接口改写。

## 调用示例<a name="section642mcpsimp"></a>

```
AscendC::SelectWithBytesMaskShapeInfo info;
srcLocal1 = inQueueX1.DeQue<srcType>();
maskLocal = maskQueue.DeQue<maskType>();
AscendC::LocalTensor<uint8_t> tmpBuffer = sharedTmpBuffer.Get<uint8_t>();
dstLocal = outQueue.AllocTensor<srcType>();
AscendC::Select(dstLocal, srcLocal1, scalar, maskLocal, tmpBuffer, info);
outQueue.EnQue<srcType>(dstLocal);
maskQueue.FreeTensor(maskLocal);
inQueueX1.FreeTensor(srcLocal1);
```

结果示例如下：

```
输入数据srcLocal1: 
[-84.6    -24.38    30.97   -30.25    22.28   -92.56    90.44   -58.72  -86.56     5.74     6.754  -86.3    -96.7    -37.38   -81.9     46.9
 -99.4     94.2    -41.78   -60.3    -14.43    78.6      8.93   -65.2    79.94   -46.88     4.516   20.03   -25.56    24.73     0.3223  21.98

 -87.4    -93.9     46.22   -69.9     90.8    -24.17   -96.2    -91.    90.44     9.766   68.25   -57.78   -75.44    -8.86   -91.56    21.6
  76.      82.1    -78.     -23.75    92.     -66.44    75.      94.9   2.62   -90.9     15.945   38.16    50.84    96.94   -59.38    44.22  ]
输入数据scalar: 
[35.6]
输入数据maskLocal: 
[False  True False False  True  True False  True  True False False  True False  True False  True  
 True   False False False  True  True  True  True   True False  True False  True  True  True  True 

 False False  True False  True False  True False  True False  True False  True  True  True False
 True False  True False  True False  True  True   True False False False  True False  True  True
]

输出数据dstLocal: 
[-84.6    35.6    30.97   -30.25   35.6    35.6    90.44   35.6  35.6    5.74    6.754   35.6   -96.7    35.6   -81.9    35.6
  35.6    94.2    -41.78  -60.3    35.6    35.6    35.6    35.6  35.6   -46.88   35.6    20.03   35.6    35.6    35.6    35.6
   
 -87.4   -93.9    35.6    -69.9    35.6   -24.17   35.6   -91.   35.6   9.766  35.6   -57.78   35.6     35.6    35.6    21.6
  35.6    82.1    35.6    -23.75   35.6   -66.44   35.6    35.6  35.6   -90.9    15.945  38.16   35.6    96.94   35.6    35.6  ]
```

