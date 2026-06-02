# SwiGLU<a name="ZH-CN_TOPIC_0000002523304006"></a>

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

SwiGLU是采用Swish作为激活函数的GLU变体。具体计算公式如下：

<!-- img2text -->
$$\operatorname{SwiGLU}(x, W, V, b, c)=\operatorname{Swish}(xW+b)\otimes(xV+c)$$

其中Swish激活函数的计算公式如下（β为常量）：

<!-- img2text -->
$$\operatorname{swish}(x)=x\cdot \sigma(\beta x)=\frac{x}{1+e^{-\beta x}}$$

## 函数原型<a name="section8850255125911"></a>

-   通过sharedTmpBuffer入参传入临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const LocalTensor<uint8_t>& sharedTmpBuffer)
        ```

-   接口框架申请临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, const LocalTensor<T>& srcTensor0, const LocalTensor<T>& srcTensor1, const float& scalarValue, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false>
        __aicore__ inline void SwiGLU(LocalTensor<T>& dstTensor, LocalTensor<T>& srcTensor0, LocalTensor<T>& srcTensor1, const float& scalarValue)
        ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持开发者**通过sharedTmpBuffer入参传入**和**接口框架申请**两种方式。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。
-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间；接口框架申请的方式，开发者需要预留临时空间。临时空间大小BufferSize的获取方式如下：通过[GetSwiGLUMaxMinTmpSize](GetSwiGLUMaxMinTmpSize.md)中提供的接口获取需要预留空间范围的大小。

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
<p id="p75831786153"><a name="p75831786153"></a><a name="p75831786153"></a><span id="ph183108134329"><a name="ph183108134329"></a><a name="ph183108134329"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
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
<table><thead align="left"><tr id="row885118552595"><th class="cellrowborder" valign="top" width="19.24%" id="mcps1.2.4.1.1"><p id="p1585195518592"><a name="p1585195518592"></a><a name="p1585195518592"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="12.22%" id="mcps1.2.4.1.2"><p id="p0851185511597"><a name="p0851185511597"></a><a name="p0851185511597"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="68.54%" id="mcps1.2.4.1.3"><p id="p1785175516591"><a name="p1785175516591"></a><a name="p1785175516591"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row2851125520594"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p9851165515593"><a name="p9851165515593"></a><a name="p9851165515593"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="12.22%" headers="mcps1.2.4.1.2 "><p id="p1785185514591"><a name="p1785185514591"></a><a name="p1785185514591"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="68.54%" headers="mcps1.2.4.1.3 "><p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>目的操作数。</p>
<p id="p12851115519599"><a name="p12851115519599"></a><a name="p12851115519599"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row6851155510593"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1385135518597"><a name="p1385135518597"></a><a name="p1385135518597"></a>srcTensor0/srcTensor1</p>
</td>
<td class="cellrowborder" valign="top" width="12.22%" headers="mcps1.2.4.1.2 "><p id="p585119553596"><a name="p585119553596"></a><a name="p585119553596"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.54%" headers="mcps1.2.4.1.3 "><p id="p963863814519"><a name="p963863814519"></a><a name="p963863814519"></a>源操作数。</p>
<p id="p493465115344"><a name="p493465115344"></a><a name="p493465115344"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
<p id="p15450144034510"><a name="p15450144034510"></a><a name="p15450144034510"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row4852185535916"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1244747105613"><a name="p1244747105613"></a><a name="p1244747105613"></a>scalarValue</p>
</td>
<td class="cellrowborder" valign="top" width="12.22%" headers="mcps1.2.4.1.2 "><p id="p44478765615"><a name="p44478765615"></a><a name="p44478765615"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.54%" headers="mcps1.2.4.1.3 "><p id="p5692173212345"><a name="p5692173212345"></a><a name="p5692173212345"></a>激活函数中的β参数。</p>
</td>
</tr>
<tr id="row204461978565"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p1684115202816"><a name="p1684115202816"></a><a name="p1684115202816"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="12.22%" headers="mcps1.2.4.1.2 "><p id="p1684117207819"><a name="p1684117207819"></a><a name="p1684117207819"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.54%" headers="mcps1.2.4.1.3 "><p id="p188174394122"><a name="p188174394122"></a><a name="p188174394122"></a>临时缓存。</p>
<p id="p178181139171217"><a name="p178181139171217"></a><a name="p178181139171217"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1981833951213"><a name="p1981833951213"></a><a name="p1981833951213"></a>用于SwiGLU内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p3818739141216"><a name="p3818739141216"></a><a name="p3818739141216"></a>临时空间大小BufferSize的获取方式请参考<a href="GetSwiGLUMaxMinTmpSize.md">GetSwiGLUMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row171991119901"><td class="cellrowborder" valign="top" width="19.24%" headers="mcps1.2.4.1.1 "><p id="p22166407018"><a name="p22166407018"></a><a name="p22166407018"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="12.22%" headers="mcps1.2.4.1.2 "><p id="p621617401705"><a name="p621617401705"></a><a name="p621617401705"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="68.54%" headers="mcps1.2.4.1.3 "><p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>实际计算数据元素个数。</p>
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

完整的算子样例请参考[swiglu算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/02_activation/swiglu)。

```
#include "kernel_operator.h"

AscendC::LocalTensor<T> dstLocal = outQueue.AllocTensor<T>();
AscendC::LocalTensor<T> src0Local = inQueue0.DeQue<T>();
AscendC::LocalTensor<T> src1Local = inQueue1.DeQue<T>();
AscendC::LocalTensor<uint8_t> tmpLocal;
if (tmpBufSize > 0) {
    tmpLocal = bufQueue.Get<uint8_t>();
}
if ((tmpBufSize > 0) && (calCount > 0)) {
    AscendC::SwiGLU(dstLocal, src0Local, src1Local, betaValue, tmpLocal, calCount);
} else if (tmpBufSize > 0) {
    AscendC::SwiGLU(dstLocal, src0Local, src1Local, betaValue, tmpLocal);
} else if (calCount > 0) {
    AscendC::SwiGLU(dstLocal, src0Local, src1Local, betaValue, calCount);
} else {
    AscendC::SwiGLU(dstLocal, src0Local, src1Local, betaValue);
}
outQueue.EnQue<T>(dstLocal);
inQueue0.FreeTensor(src0Local);
inQueue1.FreeTensor(src1Local);
```

结果示例如下：

```
输入数据(srcTensor0): 
 [ 0.4065 -0.2167 -0.963  -3.895  -0.7275  3.227  -0.522  -2.299  -1.813
 -1.569   3.764   1.407  -1.633   3.908  -0.9927 -2.234   1.545   2.
 -3.06    1.94    0.765  -1.313   3.27    2.055   2.842   2.979   2.732
  2.533   2.03    1.154  -2.363  -2.451 ]
输入数据(srcTensor1)
 [-2.285  -1.502   2.783  -3.72    0.352  -2.615   0.8604  0.612   3.582
 -3.102  -3.86    2.88   -0.2117 -0.592  -0.5586  1.315   0.4087  3.771
  2.69    0.755  -2.154  -1.03   -3.459  -3.125   3.531  -0.657   3.885
  2.807   0.469  -1.434  -3.455  -1.3   ]
输出数据(dstLocal): 
[-0.0858   0.05927 -2.523    0.3425  -0.1504  -0.575   -0.3157  -0.912
 -6.32     0.2095  -0.2998   3.838    0.1545  -0.8237   0.2018  -2.316
  0.3794   7.375   -7.707    0.9966  -0.1713   0.356   -0.345   -0.2703
  9.75    -0.6685  10.4      6.703    0.5854  -0.3186   0.25     0.6826 ]
```

