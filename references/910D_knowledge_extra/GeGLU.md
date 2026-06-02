# GeGLU<a name="ZH-CN_TOPIC_0000002554424069"></a>

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

## 功能说明<a name="section785018556590"></a>

GeGLU是采用GELU作为激活函数的GLU变体。具体计算公式如下：

<!-- img2text -->
$$
y = \operatorname{GELU}(x_1) \otimes x_2
$$

其中GELU激活函数的计算公式如下：

<!-- img2text -->
$$\operatorname{GELU}(x)=xP(X\leq x)=x\Phi(x)$$

上述公式中的erf为误差函数：<!-- img2text -->
$$\operatorname{GELU}(x)=xP(X\leq x)=x\Phi(x)=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)$$

误差函数没有解析表达式，按照业界普遍使用的tanh近似表达式：<!-- img2text -->
$$
\operatorname{erf}(x)=\frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^{2}}\,dt
$$

将GELU近似公式代入可得GeGLU表达式为：

<!-- img2text -->
$$\operatorname{GeGLU}(x_0,x_1)=x_0\times \operatorname{GELU}(x_1)=x_0\times \frac{x_1}{2}\left(1+\tanh\left(\sqrt{\frac{2}{\pi}}\left(x_1+0.044715x_1^3\right)\right)\right)$$

其中_a_=-0.0713548162726,  _b_=2.2363860002236e1，x1和x0代表srcTensor1和srcTensor0中的元素。

## 函数原型<a name="section8850255125911"></a>

-   通过sharedTmpBuffer入参传入临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const LocalTensor<uint8_t>& sharedTmpBuffer)
        ```

-   接口框架申请临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void GeGLU(const LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetGeGLUMaxMinTmpSize](GetGeGLUMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

## 参数说明<a name="section1085025505914"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001538537601_row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001538537601_p675519193268"><a name="zh-cn_topic_0000001538537601_p675519193268"></a><a name="zh-cn_topic_0000001538537601_p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001538537601_p375511918267"><a name="zh-cn_topic_0000001538537601_p375511918267"></a><a name="zh-cn_topic_0000001538537601_p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001538537601_row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p47551198266"><a name="zh-cn_topic_0000001538537601_p47551198266"></a><a name="zh-cn_topic_0000001538537601_p47551198266"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p125969172719"><a name="zh-cn_topic_0000001538537601_p125969172719"></a><a name="zh-cn_topic_0000001538537601_p125969172719"></a>操作数的数据类型。</p>
<p id="p38041041613"><a name="p38041041613"></a><a name="p38041041613"></a><span id="ph183108134329"><a name="ph183108134329"></a><a name="ph183108134329"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table1485015517590"></a>
<table><thead align="left"><tr id="row885118552595"><th class="cellrowborder" valign="top" width="19.21%" id="mcps1.2.4.1.1"><p id="p1585195518592"><a name="p1585195518592"></a><a name="p1585195518592"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.26%" id="mcps1.2.4.1.2"><p id="p0851185511597"><a name="p0851185511597"></a><a name="p0851185511597"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.53%" id="mcps1.2.4.1.3"><p id="p1785175516591"><a name="p1785175516591"></a><a name="p1785175516591"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2851125520594"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.2.4.1.1 "><p id="p9851165515593"><a name="p9851165515593"></a><a name="p9851165515593"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.26%" headers="mcps1.2.4.1.2 "><p id="p1785185514591"><a name="p1785185514591"></a><a name="p1785185514591"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>目的操作数。</p>
<p id="p12851115519599"><a name="p12851115519599"></a><a name="p12851115519599"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row6851155510593"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.2.4.1.1 "><p id="p1385135518597"><a name="p1385135518597"></a><a name="p1385135518597"></a>srcTensor0/</p>
<p id="p74435201655"><a name="p74435201655"></a><a name="p74435201655"></a>srcTensor1</p>
</td>
<td class="cellrowborder" valign="top" width="12.26%" headers="mcps1.2.4.1.2 "><p id="p585119553596"><a name="p585119553596"></a><a name="p585119553596"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p963863814519"><a name="p963863814519"></a><a name="p963863814519"></a>源操作数。</p>
<p id="p493465115344"><a name="p493465115344"></a><a name="p493465115344"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p15450144034510"><a name="p15450144034510"></a><a name="p15450144034510"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row204461978565"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.2.4.1.1 "><p id="p175721146111119"><a name="p175721146111119"></a><a name="p175721146111119"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.26%" headers="mcps1.2.4.1.2 "><p id="p222342312915"><a name="p222342312915"></a><a name="p222342312915"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p1020195318118"><a name="p1020195318118"></a><a name="p1020195318118"></a>临时缓存。</p>
<p id="p8208530115"><a name="p8208530115"></a><a name="p8208530115"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p520953181120"><a name="p520953181120"></a><a name="p520953181120"></a>用于GeGLU内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p1520195316110"><a name="p1520195316110"></a><a name="p1520195316110"></a>临时空间大小BufferSize的获取方式请参考<a href="GetGeGLUMaxMinTmpSize.md">GetGeGLUMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row171991119901"><td class="cellrowborder" valign="top" width="19.21%" headers="mcps1.2.4.1.1 "><p id="p6921128129"><a name="p6921128129"></a><a name="p6921128129"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="12.26%" headers="mcps1.2.4.1.2 "><p id="p621617401705"><a name="p621617401705"></a><a name="p621617401705"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.53%" headers="mcps1.2.4.1.3 "><p id="p12521175751115"><a name="p12521175751115"></a><a name="p12521175751115"></a>实际计算数据元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section11852175575912"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   **不支持源操作数与目的操作数地址重叠。**
-   当前仅支持ND格式的输入，不支持其他格式。
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。

## 调用示例<a name="section208521655195918"></a>

完整的算子样例请参考[geglu算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/geglu)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<srcType> dstLocal = outQueue.AllocTensor<srcType>();
AscendC::LocalTensor<srcType> src0Local = inQueue0.DeQue<srcType>();
AscendC::LocalTensor<srcType> src1Local = inQueue1.DeQue<srcType>();
AscendC::LocalTensor<uint8_t> temp;
if ((sizeof(srcType) == sizeof(half)) && (tmpBufSize > 0)) {
    temp = buf.Get<uint8_t>();
}
if ((tmpBufSize > 0) && (calCount > 0)) {
    AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, temp, calCount);
} else if (tmpBufSize > 0) {
    AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, temp);
} else if (calCount > 0) {
    AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local, calCount);
} else {
    AscendC::GeGLU<srcType, false>(dstLocal, src0Local, src1Local);
}
outQueue.EnQue<srcType>(dstLocal);
inQueue0.FreeTensor(src0Local);
inQueue1.FreeTensor(src1Local);
```

结果示例如下：

```
输入数据(srcTensor0): 
[ 1.6025391   3.4765625   3.4316406   3.7539062  -1.3330078   0.72314453
 -3.0078125   0.85498047 -1.3691406   2.6894531  -2.9101562  -3.6992188
 -2.2734375  -2.859375    2.5683594  -1.7802734 ]

输入数据(srcTensor1)
[-0.6015625  1.9589844  1.9257812  3.8769531  0.5878906  2.9179688
 -1.8847656  3.2304688  2.8945312  2.4550781  1.3730469 -1.9248047
  0.7919922 -2.5332031 -2.1425781 -2.9433594]

输出数据(dstLocal): [-0.263916015625000000 6.640625000000000000 6.429687500000000000
14.554687500000000000 -0.565429687500000000 2.107421875000000000 0.168579101562500000
2.759765625000000000 -3.957031250000000000 6.558593750000000000 -3.656250000000000000
0.192993164062500000 -1.415039062500000000 0.039642333984375000 -0.087890625000000000
0.007740020751953125]
```

