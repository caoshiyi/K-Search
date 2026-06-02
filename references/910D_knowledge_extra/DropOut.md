# DropOut<a name="ZH-CN_TOPIC_0000002523304930"></a>

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

提供根据MaskTensor对SrcTensor（源操作数，输入Tensor）进行过滤的功能，得到DstTensor（目的操作数、输出Tensor）。仅支持输入shape为ND格式。

该过滤功能包括两种模式，**字节模式**和**比特模式**。

-   **字节模式**

    MaskTensor中存储的数值为布尔类型，每个布尔数值代表是否取用SrcTensor对应位置的数值：如果是，则选取SrcTensor中的数值存入DstTensor；否则，对DstTensor中的对应位置赋值为零。DstTensor，SrcTensor和MaskTensor的shape相同。示例如下：

    **SrcTensor=\[1，2，3，4，5，6，7，8，9，10\]**

    **MaskTensor=\[1，0，1，0，1，0，0，1，1，0\]（每个数的数据类型为uint8\_t）**

    **DstTensor=\[1，0，3，0，5，0，0，8，9，0\]**

-   **比特模式**

    MaskTensor的每个bit数值，代表是否取用SrcTensor对应位置的数值：如果是，则选取SrcTensor中的数值存入DstTensor；否则，对DstTensor中的对应位置赋值为零。SrcTensor和DstTensor的shape相同，假设均为\[height ， width\]，MaskTensor的shape为\[height ， \(width / 8\)\]。示例如下：

    **SrcTensor=\[1，2，3，4，5，6，7，8\]**

    **MaskTensor=\[169\]（转换为二进制表示为1010 1001）**

    **DstTensor=\[1，0，3，0，5，0，0，8\]**

    -   特殊情况1：当MaskTensor有效数据非连续存放时，MaskTensor的width轴，为了满足32B对齐，需要填充无效数值，SrcTensor的width轴，需满足32Byte对齐。示例如下：

        SrcTensor=\[1，2，3，4，5，6，7，8，11，12，13，14，15，16，17，18\]

        MaskTensor=\[1，0，1，0，1，0，0，1，X，X，1，0，1，0，1，0，0，1，X，X\]（X为无效数值，假设数据已满足对齐要求，示例数值为二进制形式表示）

        DstTensor=\[1，0，3，0，5，0，0，8，11，0， 13， 0， 15， 0， 0，18\]

    -   特殊情况2：当MaskTensor有效数据连续存放，maskTensor\_size不满足32B对齐时，需要在MaskTensor的尾部补齐32B对齐时，对应SrcTensor的尾部也需要补充无效数据，使得srcTensor\_size满足32B对齐。示例如下：

        SrcTensor=\[1，2，3，4，5，6，7，8，11，12，13，14，15，16，17，18\]

        MaskTensor=\[1，0，1，0，1，0，0，1， 1，  0，  1，  0，  1，  0，  0，  1，X，X，X，X\]（X为无效数值，假设数据已满足对齐要求，示例数值为二进制形式表示）

        DstTensor= \[1，0，3，0，5，0，0，8， 11， 0， 13， 0， 15， 0， 0， 18\]

## 实现原理<a name="section13229175017585"></a>

以float类型，ND格式，shape为\[srcM, srcN\]的SrcTensor，shape为\[maskM, maskN\]的MaskTensor，比特模式场景为例，描述Dropout高阶API内部算法框图，如下图所示。

**图 1**  Dropout算法框图<a name="fig144741562718"></a>  
<!-- img2text -->
```text
┌───────────────────────┐   ┌────────────────────────┐
│ SrcTensor[srcM, srcN] │   │ MaskTensor[maskM, maskN] │
└───────────────────────┘   └────────────────────────┘
             │                         │
             └──────────────┬──────────┘
                            │
        ┌──────────────────────────────────────────┐
        │                                          │
        │        ┌──────────────────────────┐      │
        │        │ GatherMask(MaskTensor)   │      │
        │        └──────────────────────────┘      │
        │                     │                    │
        │                     ↓                    │
        │        ┌──────────────────────────┐      │
        │        │ Select(MaskTensor,       │      │
        │        │        SrcTensor)        │      │
        │        └──────────────────────────┘      │
        │                     │                    │
        │                     ↓                    │
        │        ┌──────────────────────────┐      │
        │        │ Muls(DstTensor, 1 /      │      │
        │        │      keepProb)           │      │
        │        └──────────────────────────┘      │
        │                     │                    │
        └─────────────────────┼────────────────────┘
                              │
                              ↓
                   ┌───────────────────────┐
                   │ DstTensor[srcM, srcN] │
                   └───────────────────────┘


图示:
输入输出Tensor    ── 矩形框
vector计算       ── 矩形框
数据流向         ── → / ↓
```

计算过程分为如下几步，均在Vector上进行：

1.  GatherMask步骤：对输入的MaskTensor做脏数据清理，使得MaskTensor中只保留有效数据；
2.  Select步骤：根据输入的MaskTensor对SrcTensor做数据选择，被选中的数据位置，保留原始数据，对舍弃的数据位置，设置为0；
3.  Muls步骤：将输出数据每个元素除以keepProb。

**图 2**  Dropout算法框图<a name="fig767314154518"></a>  
<!-- img2text -->
```text
┌────────────────────────┐      ┌──────────────────────────┐
│ SrcTensor[srcM, srcN]  │      │ MaskTensor[maskM, maskN] │
└────────────┬───────────┘      └─────────────┬────────────┘
             │                                │
             └───────────────┬────────────────┘
                             │
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                         ┌───────────┐                                        │
│                         │ int i = 0 │                                        │
│                         └─────┬─────┘                                        │
│                               │                                              │
│                               ▼                                              │
│                         ╱───────────╲                                        │
│                        ╱  i < srcM   ╲──────────────否──────────────────┐    │
│                        ╲             ╱                                  │    │
│                         ╲───────────╱                                   │    │
│                               │是                                       │    │
│                               ▼                                         │    │
│             ┌────────────────────────────────────┐                      │    │
│             │ Select(MaskTensor, SrcTensor,      │                      │    │
│             │ srcN)                              │                      │    │
│             └────────────────┬───────────────────┘                      │    │
│                              │                                          │    │
│                              ▼                                          │    │
│             ┌────────────────────────────────────┐                      │    │
│             │ Muls(DstTensor, 1 / keepProb,      │                      │    │
│             │ srcN)                              │                      │    │
│             └────────────────┬───────────────────┘                      │    │
│                              │                                          │    │
│                              ▼                                          │    │
│                        ┌───────────┐                                    │    │
│                        │    i++    │                                    │    │
│                        └─────┬─────┘                                    │    │
│                              │                                          │    │
│                              └───────────────────────回到 i < srcM──────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────┬───────┘
                                                                       │
                                                                       ▼
                                                     ┌────────────────────────┐
                                                     │ DstTensor[srcM, srcN]  │
                                                     └────────────────────────┘


图示:
输入输出Tensor   ────────────────  矩形框
vector计算       ────────────────  圆角矩形框
条件判断         ────────────────  菱形框
数据流向         ────────────────  → 
```

计算过程在Vector上进行，循环srcM次，每次对srcN个元素进行如下操作：

1.  Select步骤：根据输入的MaskTensor对SrcTensor做数据选择，被选中的数据位置，保留原始数据，对舍弃的数据位置，设置为0；
2.  Muls步骤：将输出数据每个元素除以keepProb。

## 函数原型<a name="section620mcpsimp"></a>

```
template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOut(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint8_t>& maskLocal, const float keepProb, const DropOutShapeInfo& info)
```

```
template <typename T, bool isInitBitMode = false, uint32_t dropOutMode = 0>
__aicore__ inline void DropOut(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<uint8_t>& maskLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const float keepProb, const DropOutShapeInfo& info)
```

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
<p id="p14882204972511"><a name="p14882204972511"></a><a name="p14882204972511"></a><span id="ph691515191215"><a name="ph691515191215"></a><a name="ph691515191215"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="row197562193260"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p18961115352610"><a name="p18961115352610"></a><a name="p18961115352610"></a>isInitBitMode</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1287421219275"><a name="p1287421219275"></a><a name="p1287421219275"></a>在比特模式下，是否需要在接口内部初始化（默认false）。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>dropOutMode</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p14512517006"><a name="p14512517006"></a><a name="p14512517006"></a>选择执行何种输入场景：</p>
<p id="p75121417206"><a name="p75121417206"></a><a name="p75121417206"></a>0：默认值，由接口根据输入shape推断运行模式，注意，推断不符合预期的场景，需设置对应模式</p>
<p id="p151211175015"><a name="p151211175015"></a><a name="p151211175015"></a>1：执行字节模式，且maskLocal含有脏数据</p>
<p id="p11512181717019"><a name="p11512181717019"></a><a name="p11512181717019"></a>2：执行字节模式，且maskLocal不含有脏数据</p>
<p id="p251201712020"><a name="p251201712020"></a><a name="p251201712020"></a>3：执行比特模式，且maskLocal不含有脏数据</p>
<p id="p12512151713019"><a name="p12512151713019"></a><a name="p12512151713019"></a>4：执行比特模式，且maskLocal含有脏数据</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="13.4013401340134%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="12.451245124512452%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="74.14741474147414%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row5956546509"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p4428175618426"><a name="p4428175618426"></a><a name="p4428175618426"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p2428856174212"><a name="p2428856174212"></a><a name="p2428856174212"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row18195104073211"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p10429155616425"><a name="p10429155616425"></a><a name="p10429155616425"></a>srcLocal</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p164291756114215"><a name="p164291756114215"></a><a name="p164291756114215"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p2201757171819"><a name="p2201757171819"></a><a name="p2201757171819"></a>源操作数。</p>
<p id="p253763161920"><a name="p253763161920"></a><a name="p253763161920"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1942985674213"><a name="p1942985674213"></a><a name="p1942985674213"></a>srcLocal的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row83038193319"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p203045116333"><a name="p203045116333"></a><a name="p203045116333"></a>maskLocal</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p33049133318"><a name="p33049133318"></a><a name="p33049133318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p1861414963310"><a name="p1861414963310"></a><a name="p1861414963310"></a>存放mask的Tensor，数据类型为uint8_t。</p>
<p id="p1370141481917"><a name="p1370141481917"></a><a name="p1370141481917"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row6919184671816"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p179192469188"><a name="p179192469188"></a><a name="p179192469188"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p209199466187"><a name="p209199466187"></a><a name="p209199466187"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p47801866195"><a name="p47801866195"></a><a name="p47801866195"></a>共享缓冲区，用于存放API内部计算产生的临时数据。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存。Tensor的大小应符合对应tiling的要求，配合tiling一起使用。共享缓冲区大小BufferSize的获取方式请参考<a href="GetDropOutMaxMinTmpSize.md">GetDropOutMaxMinTmpSize</a>。</p>
<p id="p167809613199"><a name="p167809613199"></a><a name="p167809613199"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row448510291335"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p64850295337"><a name="p64850295337"></a><a name="p64850295337"></a>keepProb</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p55671433103318"><a name="p55671433103318"></a><a name="p55671433103318"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p948516298337"><a name="p948516298337"></a><a name="p948516298337"></a>权重系数，数据类型为float，srcLocal中数据被保留的概率，过滤后的结果会除以权重系数，存放至dstLocal中。</p>
<p id="p13711277215"><a name="p13711277215"></a><a name="p13711277215"></a>keepProb∈(0，1)</p>
</td>
</tr>
<tr id="row9747135833415"><td class="cellrowborder" valign="top" width="13.4013401340134%" headers="mcps1.2.4.1.1 "><p id="p3709161712358"><a name="p3709161712358"></a><a name="p3709161712358"></a>info</p>
</td>
<td class="cellrowborder" valign="top" width="12.451245124512452%" headers="mcps1.2.4.1.2 "><p id="p87099174355"><a name="p87099174355"></a><a name="p87099174355"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="74.14741474147414%" headers="mcps1.2.4.1.3 "><p id="p27481649183615"><a name="p27481649183615"></a><a name="p27481649183615"></a>DropOutShapeInfo类型，DropOutShapeInfo结构定义如下：</p>
<a name="screen1998174885413"></a><a name="screen1998174885413"></a><pre class="screen" codetype="Cpp" id="screen1998174885413">struct DropOutShapeInfo {
__aicore__ DropOutShapeInfo(){};
uint32_t firstAxis = 0;   // srcLocal/maskTensor的height轴元素个数
uint32_t srcLastAxis = 0; // srcLocal的width轴元素个数
uint32_t maskLastAxis = 0;// maskTensor的width轴元素个数（如有数据补齐场景，则为带有脏数据的长度，注意，所有模式的元素个数均为对应Tensor类型下的个数，取值需要大于0，如uint8类型Tensor对应Uint8类型元素个数）
};</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section5468191312484"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   srcTensor和dstTensor的Tensor空间可以复用。
-   srcLocal和dstLocal地址对齐要求请见：[通用说明和约束](通用说明和约束.md)。
-   仅支持输入shape为ND格式。
-   maskLocal含有脏数据的场景，要求info.maskLastAxis中有效数值的个数，应为2的整数倍。
-   maskLocal含有脏数据的场景，maskLocal中的数据可能会被修改，脏数据可能会被舍弃。

## 调用示例<a name="section642mcpsimp"></a>

完整的算子样例请参考[DropOut样例](http://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/08_filter/dropout)。

```
AscendC::DropOutShapeInfo info;
float probValue = 0.8;
info.firstAxis = tilingData.firstAxis / tilingData.tileNum;
info.srcLastAxis = tileLength;
info.maskLastAxis = tileLength;
AscendC::DropOut(yLocal, xLocal, maskLocal, sharedTmpBuffer, probValue, info);
```

