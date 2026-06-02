# Mean<a name="ZH-CN_TOPIC_0000002554344707"></a>

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

根据最后一轴的方向对各元素求平均值。

如果输入是向量，则在向量中对各元素相加求平均；如果输入是矩阵，则沿最后一个维度对元素求平均。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

如下图所示，对shape为\(4, 5\)的二维矩阵进行求平均操作，输出结果为\[3， 8， 13， 18\]。

<!-- img2text -->
```text
┌────┬────┬────┬────┬────┐
│ 1  │ 2  │ 3  │ 4  │ 5  │
├────┼────┼────┼────┼────┤
│ 6  │ 7  │ 8  │ 9  │ 10 │
├────┼────┼────┼────┼────┤
│ 11 │ 12 │ 13 │ 14 │ 15 │
├────┼────┼────┼────┼────┤
│ 16 │ 17 │ 18 │ 19 │ 20 │
└────┴────┴────┴────┴────┘

        按最后一个维度求平均值
                 ─────────────→

┌────┐
│ 3  │
├────┤
│ 8  │
├────┤
│ 13 │
├────┤
│ 18 │
└────┘
```

在了解接口具体功能之前，需要了解一些必备概念：数据的行数称之为**外轴长度（outter）**，每行实际的元素个数称之为**内轴的实际元素个数（n）**，内轴实际元素个数n向上32字节对齐后的元素个数称之为**补齐后的内轴元素个数\(inner\)**。本接口要求输入的内轴长度满足32字节对齐，所以当n占据的字节长度不是32字节的整数倍时，需要开发者将其向上补齐到32字节的整数倍。如下样例中，元素类型为float，每行的实际元素个数n为5，占据字节长度为20字节，不是32字节的整数倍，向上补齐后得到32字节，对应的元素个数为8。图中的padding代表补齐操作。n和inner的关系如下：**inner = \(n \*sizeof\(T\) + 32 - 1\) / 32 \* 32 / sizeof\(T\)**。

<!-- img2text -->
```text
                         padding
                    <──────────────>
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│  1  │  2  │  3  │  4  │  5  │     │     │     │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│  6  │  7  │  8  │  9  │ 10  │     │     │     │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 11  │ 12  │ 13  │ 14  │ 15  │     │     │     │
├─────┼─────┼─────┼─────┼─────┼─────┼─────┼─────┤
│ 16  │ 17  │ 18  │ 19  │ 20  │     │     │     │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
outter = 4

<───────────────────────────────────────────────>
      数据类型为float，每行实际元素个数n=5，32字节补齐后元素个数inner=8
```

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false, int32_t reduceDim = -1>
    __aicore__ inline void Mean(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const MeanParams& meanParams)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, typename accType = T, bool isReuseSource = false, bool isBasicBlock = false, int32_t reduceDim = -1>
    __aicore__ inline void Mean(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor, const MeanParams& meanParams)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetMeanMaxMinTmpSize](GetMeanMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

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
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p157510160329"><a name="p157510160329"></a><a name="p157510160329"></a>accType</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p8344939163918"><a name="p8344939163918"></a><a name="p8344939163918"></a>实际参与计算的数据类型，设置的accType精度高于输入T的情况下，在计算之前会将输入转换为accType，使用accType类型计算，计算完成后再转换为原来的数据类型。设置accType值升精度可以防止数据类型溢出。T为half时，您可以将accType设置为float，表示为输入half类型升精度至float进行计算。不支持accType精度低于输入T的情况。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p19898103017419"><a name="p19898103017419"></a><a name="p19898103017419"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row691132444211"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p53131021173917"><a name="p53131021173917"></a><a name="p53131021173917"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p828982819404"><a name="p828982819404"></a><a name="p828982819404"></a>预留参数，暂不支持。</p>
</td>
</tr>
<tr id="row11269152712423"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p168341447114214"><a name="p168341447114214"></a><a name="p168341447114214"></a>reduceDim</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p783484764216"><a name="p783484764216"></a><a name="p783484764216"></a>用于指定按数据的哪一维度进行求和。本接口按最后一个维度实现，不支持reduceDim参数，传入默认值-1即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="17.380000000000003%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.74%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.88%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p17444349398"><a name="p17444349398"></a><a name="p17444349398"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1061213369414"><a name="p1061213369414"></a><a name="p1061213369414"></a>输出值需要outter * sizeof(T)大小的空间进行保存。开发者要根据该大小和框架的对齐要求来为dstTensor分配实际内存空间。</p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p3707144233915"><a name="p3707144233915"></a><a name="p3707144233915"></a>源操作数。</p>
<p id="p261714443395"><a name="p261714443395"></a><a name="p261714443395"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p2048710451721"><a name="p2048710451721"></a><a name="p2048710451721"></a>输入数据shape为outter * inner。开发者需要为其开辟大小为outter * inner * sizeof(T)的空间。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p1148552183919"><a name="p1148552183919"></a><a name="p1148552183919"></a>临时缓存。</p>
<p id="p5983205412394"><a name="p5983205412394"></a><a name="p5983205412394"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于Mean内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetMeanMaxMinTmpSize.md">GetMeanMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row128437174616"><td class="cellrowborder" valign="top" width="17.380000000000003%" headers="mcps1.2.4.1.1 "><p id="p1328514137476"><a name="p1328514137476"></a><a name="p1328514137476"></a>MeanParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.74%" headers="mcps1.2.4.1.2 "><p id="p17337238144716"><a name="p17337238144716"></a><a name="p17337238144716"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.88%" headers="mcps1.2.4.1.3 "><p id="p49691232086"><a name="p49691232086"></a><a name="p49691232086"></a>srcTensor的shape信息。MeanParams类型，具体定义如下：</p>
<a name="screen641172125916"></a><a name="screen641172125916"></a><pre class="screen" codetype="Cpp" id="screen641172125916">struct MeanParams{
    uint32_t outter = 1;    // 表示输入数据的外轴长度
    uint32_t inner;         // 表示输入数据内轴实际元素个数32字节补齐后的元素个数，inner*sizeof(T)必须是32字节的整数倍
    uint32_t n;             // 表示输入数据内轴的实际元素个数
};</pre>
<a name="ul19603810111110"></a><a name="ul19603810111110"></a><ul id="ul19603810111110"><li>MeanParams.inner*sizeof(T)必须是32字节的整数倍。</li><li>MeanParams.inner是MeanParams.n向上32字节对齐后的值，inner = (n *sizeof(T) + 32 - 1) / 32 * 32 / sizeof(T)，因此MeanParams.n的大小应该满足：1 &lt;= MeanParams.n &lt;= MeanParams.inner。</li></ul>
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
-   对于mean，采用的方式为先求和再做除法，其求和时内部使用的底层相加方式与[Sum](Sum.md)、[ReduceSum](ReduceSum.md)以及[WholeReduceSum](WholeReduceSum.md)的内部的相加方式一致，采用二叉树方式，两两相加，可参考[Sum](Sum.md)。

## 调用示例<a name="section642mcpsimp"></a>

完整的算子样例请参考[mean算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/05_reduce/mean)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<T> srcLocal = inQueue.DeQue<T>();
AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
T scalar(0);
AscendC::Duplicate<T>(dstLocal, scalar,
                        (meanParams.outter * sizeof(T) + AscendC::ONE_BLK_SIZE - 1) / AscendC::ONE_BLK_SIZE
                            * AscendC::ONE_BLK_SIZE);
if (tmpSize != 0) {
    pipe->InitBuffer(tmplocalBuf, tmpSize);
    AscendC::LocalTensor<uint8_t> tmplocalTensor = tmplocalBuf.Get<uint8_t>();
    AscendC::Mean<T, accType>(dstLocal, srcLocal, tmplocalTensor, meanParams);
} else {
    AscendC::Mean<T, accType>(dstLocal, srcLocal, meanParams);
}

outQueue.EnQue<T>(dstLocal);
inQueue.FreeTensor(srcLocal);
```

结果示例如下：

输入元素类型为half，大小为2\*3的二维数据，则outter为2，n为3，sizeof\(T\)为2，inner = \(3 \* 2 + 32 - 1\)/32 \* 32 / 2 = 16。

```
输入数据(srcLocal): [[1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0],
                     [4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0]]
输出数据(dstLocal): [2 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
```

