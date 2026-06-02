# Sum<a name="ZH-CN_TOPIC_0000002554423605"></a>

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

获取最后一个维度的元素总和。

如果输入是向量，则在向量中对各元素相加；如果输入是矩阵，则沿最后一个维度对每行中元素求和。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

如下图所示，对shape为\(2, 3\)的二维矩阵进行运算，输出结果为\[6, 15\]。

<!-- img2text -->
```text
input
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
└─────┴─────┴─────┘

          按最后一个维度求和
┌──────────────────────────────┐
│                              │
└─────────────────────────────▶│

output
┌─────┐
│  6  │
├─────┤
│ 15  │
└─────┘
```

为计算如上过程，引入一些必备概念：行数称之为**外轴长度（outter）**，每行实际的元素个数称之为**内轴的实际元素个数（n）**，存储n个元素所需的字节长度向上补齐到32整数倍后转换的元素个数称之为**补齐后的内轴元素个数\(inner\)**。本接口要求输入的内轴长度为32字节的整数倍，所以当n占据的字节长度不是32的整数倍时，需要开发者将其向上补齐到32的整数倍。比如，如下的样例中，元素类型为half，每行的实际元素个数n为3，占据字节长度为6字节，不是32字节的整数倍，向上补齐后得到32字节，转换为元素个数为16。故outter = 2，n =3，inner=16。图中的padding代表补齐操作。n和inner的关系如下：**inner = \(n \*sizeof\(T\) + 32 - 1\) / 32 \* 32 / sizeof\(T\)**。

<!-- img2text -->
```text
outter=2

                    inner=16
┌───────┬───────┬───────┬────────────────┐
│   1   │   2   │   3   │    padding     │
├───────┼───────┼───────┼────────────────┤
│   4   │   5   │   6   │    padding     │
└───────┴───────┴───────┴────────────────┘
```

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void Sum(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const SumParams& sumParams)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, int32_t reduceDim = -1, bool isReuseSource = false, bool isBasicBlock = false>
    __aicore__ inline void Sum(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const SumParams& sumParams)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetSumMaxMinTmpSize](GetSumMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

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
<p id="p382544110205"><a name="p382544110205"></a><a name="p382544110205"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p157510160329"><a name="p157510160329"></a><a name="p157510160329"></a>reduceDim</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p8344939163918"><a name="p8344939163918"></a><a name="p8344939163918"></a>用于指定按数据的哪一维度进行求和。本接口按最后一个维度实现，不支持reduceDim参数，传入默认值-1即可。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p89797304326"><a name="p89797304326"></a><a name="p89797304326"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row431312153917"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p53131021173917"><a name="p53131021173917"></a><a name="p53131021173917"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p101501952123710"><a name="p101501952123710"></a><a name="p101501952123710"></a>预留参数，暂不支持。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.38%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p7421248132119"><a name="p7421248132119"></a><a name="p7421248132119"></a>输出值需要outter * sizeof(T)大小的空间进行保存。开发者要根据该大小和框架的对齐要求来为dstTensor分配实际内存空间。</p>
<div class="note" id="note12620157122515"><a name="note12620157122515"></a><a name="note12620157122515"></a><span class="notetitle"> 说明： </span><div class="notebody"><p id="p1962012573256"><a name="p1962012573256"></a><a name="p1962012573256"></a><strong id="b26211557122513"><a name="b26211557122513"></a><a name="b26211557122513"></a>注意：遵循框架对内存开辟的要求（开辟内存的大小满足32Byte对齐），即outter * sizeof(T)不是32Byte对齐时，需要向上进行32Byte对齐。为了对齐而多开辟的内存空间不填值，为一些随机值。</strong></p>
</div></div>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。</p>
<p id="p261714443395"><a name="p261714443395"></a><a name="p261714443395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1148552183919"><a name="p1148552183919"></a><a name="p1148552183919"></a>临时缓存。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于Sum内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetSumMaxMinTmpSize.md">GetSumMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row216545817417"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1949611581317"><a name="p1949611581317"></a><a name="p1949611581317"></a>sumParams</p>
</td>
<td class="cellrowborder" valign="top" width="10.38%" headers="mcps1.2.4.1.2 "><p id="p174961758436"><a name="p174961758436"></a><a name="p174961758436"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p49691232086"><a name="p49691232086"></a><a name="p49691232086"></a>srcTensor的shape信息。SumParams类型，具体定义如下：</p>
<a name="screen641172125916"></a><a name="screen641172125916"></a><pre class="screen" codetype="Cpp" id="screen641172125916">struct SumParams{
    uint32_t outter = 1;    // 表示输入数据的外轴长度
    uint32_t inner;         // 表示输入数据内轴的补齐后元素个数，inner*sizeof(T)必须是32字节的整数倍
    uint32_t n;             // 表示输入数据内轴的实际元素个数
};</pre>
<a name="ul19603810111110"></a><a name="ul19603810111110"></a><ul id="ul19603810111110"><li>sumParams.inner*sizeof(T)必须是32字节的整数倍。</li><li>sumParams.inner是sumParams.n字节数转换后进而进行32的整数倍向上补齐的值，inner = (n *sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)，因此sumParams.n的大小应该满足：1 &lt;= sumParams.n &lt;= sumParams.inner。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   当前仅支持ND格式的输入，不支持其他格式。
-   一维输入的outter值填为1；二维输入按实际情况填写outter和n，inner计算请按如上公式计算，否则功能不正确。
-   srcTensor需要能够容纳内轴对齐后的数据占用空间大小，dstTensor需要能够容纳outter个结果对齐后的数据占用空间大小。
-   对于Sum，其内部使用的底层相加方式和[ReduceSum](ReduceSum.md)以及[WholeReduceSum](WholeReduceSum.md)的内部的相加方式一致，采用二叉树方式，两两相加：

    假设源操作数为128个half类型的数据\[data0,data1,data2...data127\]，一个repeat可以计算完，计算过程如下。

    1.  data0和data1相加得到data00，data2和data3相加得到data01...data124和data125相加得到data62，data126和data127相加得到data63；
    2.  data00和data01相加得到data000，data02和data03相加得到data001...data62和data63相加得到data031；
    3.  以此类推，得到目的操作数为1个half类型的数据\[data\]。

## 调用示例<a name="section642mcpsimp"></a>

完整的算子样例请参考[sum算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/sum)。

```
AscendC::SumParams params;
params.inner = inner;
params.outter = outter;
params.n = n;
T scalar(0);
AscendC::Duplicate<T>(yLocal, scalar, out_inner);
AscendC::Sum(yLocal, xLocal, sharedTmpBuffer, params);
```

结果示例如下，输入元素类型为half，大小为2\*3的二维数据，则outter为2，n为3，sizeof\(T\)为2，inner = \(3 \* 2 + 32 - 1\)/32 \* 32 / 2 = 16。

```
输入数据srcLocal: [[1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0],
                     [4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0]]
输出数据dstLocal: [6 15 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

