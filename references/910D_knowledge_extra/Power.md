# Power<a name="ZH-CN_TOPIC_0000002523304860"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table1334714391211"></a>
<table><thead align="left"><tr id="row1334743121213"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p834713321216"><a name="p834713321216"></a><a name="p834713321216"></a><span id="ph834783101215"><a name="ph834783101215"></a><a name="ph834783101215"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p2347234127"><a name="p2347234127"></a><a name="p2347234127"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row113472312122"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p234710320128"><a name="p234710320128"></a><a name="p234710320128"></a><span id="ph103471336127"><a name="ph103471336127"></a><a name="ph103471336127"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p4751940181211"><a name="p4751940181211"></a><a name="p4751940181211"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

实现按元素做幂运算功能，提供3类接口，处理逻辑如下：

<!-- img2text -->
$$
Y = \operatorname{pow}(X, \textit{exponent})
$$

$$
Y_i =
\begin{cases}
X_i^{\textit{exponent}}, & X_i \geq 0 \\
\left| X_i \right|^{\textit{exponent}} \times \cos(\pi \times \textit{exponent}) + \left( \left| X_i \right|^{\textit{exponent}} \times \sin(\pi \times \textit{exponent}) \right) i, & X_i < 0
\end{cases}
$$

<!-- img2text -->
$$
\text{dstTensor}_i =
\begin{cases}
\text{src0Tensor}_i^{\text{src1Tensor}_i} & \text{Power(dstTensor, src0Tensor, src1Tensor)} \\
\text{src0Tensor}_i^{\text{scalar}} & \text{Power(dstTensor, src0Tensor, scalar)} \\
\text{scalar}^{\text{src1Tensor}_i} & \text{Power(dstTensor, scalar, src1Tensor)}
\end{cases}
$$

## 函数原型<a name="section620mcpsimp"></a>

-   Power\(dstTensor, src0Tensor, src1Tensor\)
    -   通过sharedTmpBuffer入参传入临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
            ```

    -   接口框架申请临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor)
            ```

-   Power\(dstTensor, src0Tensor, src1Scalar\)
    -   通过sharedTmpBuffer入参传入临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, const LocalTensor<uint8_t>& sharedTmpBuffer)
            ```

    -   接口框架申请临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const T& src1Scalar)
            ```

-   Power\(dstTensor, src0Scalar, src1Tensor\)
    -   通过sharedTmpBuffer入参传入临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
            ```

    -   接口框架申请临时空间
        -   源操作数Tensor全部/部分参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor, uint32_t calCount)
            ```

        -   源操作数Tensor全部参与计算

            ```
            template <typename T, bool isReuseSource = false, const PowerConfig& config = defaultPowerConfig>
            __aicore__ inline void Power(const LocalTensor<T>& dstTensor, const T& src0Scalar, const LocalTensor<T>& src1Tensor)
            ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetPowerMaxMinTmpSize](GetPowerMaxMinTmpSize.md)中提供的GetPowerMaxMinTmpSize接口获取需要预留空间的范围大小。

## 参数说明<a name="section622mcpsimp"></a>

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
<p id="p5315184745513"><a name="p5315184745513"></a><a name="p5315184745513"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint8_t、int8_t、uint16_t、int16_t、uint32_t、int32_t、half、bfloat16_t、float。</p>
</td>
</tr>
<tr id="zh-cn_topic_0000001538537601_row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row1298025681320"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p179771456161314"><a name="p179771456161314"></a><a name="p179771456161314"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1977105611138"><a name="p1977105611138"></a><a name="p1977105611138"></a>Power计算的相关配置。此参数可选配，PowerConfig类型，具体定义如下方代码所示，其中参数的含义为：</p>
<div class="p" id="p4742172411393"><a name="p4742172411393"></a><a name="p4742172411393"></a>algo：不同的数据类型支持的不同的Power算法。该参数支持的取值如下：<a name="ul827932118545"></a><a name="ul827932118545"></a><ul id="ul827932118545"><li>INTRINSIC：默认值。如果数据类型是整型，INTRINSIC算法使用快速幂算法实现Power计算，支持的数据类型为uint8_t、int8_t、uint16_t、int16_t、uint32_t、int32_t。如果数据类型是浮点数类型，INTRINSIC算法按照公式Power(x, y) = exp(y * ln(x))进行Power计算，支持的数据类型为half、float。</li><li>DOUBLE_FLOAT_TECH：DOUBLE_FLOAT_TECH算法是高精度浮点数算法，将源操作数的精度提升后，按照公式Power(x, y) = exp(y * ln(x))进行Power计算，减少计算过程中的精度损失，支持的数据类型为bfloat16_t、half、float。</li></ul>
</div>
</td>
</tr>
</tbody>
</table>

```
enum class PowerAlgo {
    INTRINSIC = 0,
    DOUBLE_FLOAT_TECH, 
};

struct PowerConfig {
    PowerAlgo algo = PowerAlgo::INTRINSIC;
};
```

**表 2**  接口参数说明

<a name="table148471830151913"></a>
<table><thead align="left"><tr id="row1984733010194"><th class="cellrowborder" valign="top" width="16.470000000000002%" id="mcps1.2.4.1.1"><p id="p2847730181917"><a name="p2847730181917"></a><a name="p2847730181917"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.29%" id="mcps1.2.4.1.2"><p id="p58476303197"><a name="p58476303197"></a><a name="p58476303197"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.24000000000001%" id="mcps1.2.4.1.3"><p id="p10847203021913"><a name="p10847203021913"></a><a name="p10847203021913"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row98477303196"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p15847183018194"><a name="p15847183018194"></a><a name="p15847183018194"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p148471930161917"><a name="p148471930161917"></a><a name="p148471930161917"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p91371226164420"><a name="p91371226164420"></a><a name="p91371226164420"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>src0Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p897119334445"><a name="p897119334445"></a><a name="p897119334445"></a>源操作数。</p>
<p id="p174261936154417"><a name="p174261936154417"></a><a name="p174261936154417"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row1276105695614"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p554386145714"><a name="p554386145714"></a><a name="p554386145714"></a>src1Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p195437617571"><a name="p195437617571"></a><a name="p195437617571"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1948441114419"><a name="p1948441114419"></a><a name="p1948441114419"></a>源操作数。</p>
<p id="p1270404284418"><a name="p1270404284418"></a><a name="p1270404284418"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p1654314619576"><a name="p1654314619576"></a><a name="p1654314619576"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row8982145785720"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p104294335816"><a name="p104294335816"></a><a name="p104294335816"></a>src0Scalar/src1Scalar</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p242993165811"><a name="p242993165811"></a><a name="p242993165811"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p19429173105819"><a name="p19429173105819"></a><a name="p19429173105819"></a>源操作数，类型为Scalar。源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p1250544115597"><a name="p1250544115597"></a><a name="p1250544115597"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p1450514115593"><a name="p1450514115593"></a><a name="p1450514115593"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1657516544449"><a name="p1657516544449"></a><a name="p1657516544449"></a>临时内存空间。</p>
<p id="p39145804415"><a name="p39145804415"></a><a name="p39145804415"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p53269189483"><a name="p53269189483"></a><a name="p53269189483"></a>针对3个power接口，不同输入数据类型情况下，临时空间大小BufferSize的获取方式请参考<a href="GetPowerMaxMinTmpSize.md">GetPowerMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row20445230135819"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p13445830155815"><a name="p13445830155815"></a><a name="p13445830155815"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p19445163011582"><a name="p19445163011582"></a><a name="p19445163011582"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p1344523013588"><a name="p1344523013588"></a><a name="p1344523013588"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例请参考[Power样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/00_math/power)。

```
// dstLocal: 存放计算结果的Tensor
// srcLocalExp: Power计算使用的指数Tensor
// srcLocalBase: Power计算使用的底数Tensor

// 使用srcLocalBase做底数对srcLocalExp中的全部元素做幂运算
AscendC::Power<T, false>(dstLocal, srcLocalBase, srcLocalExp);

// scalarValueBase: Power计算使用的底数
T scalarValueBase = srcLocalBase.GetValue(0);
// 使用同一个底数scalarValueBase对srcLocalExp中的全部元素做幂运算
AscendC::Power<T, false>(dstLocal, scalarValueBase, srcLocalExp);

// scalarValueExp: Power计算使用的指数
T scalarValueExp = srcLocalExp.GetValue(0);
// 使用同一个指数scalarValueExp对srcLocalBase中的全部元素做幂运算
AscendC::Power<T, false>(dstLocal, srcLocalBase, scalarValueExp);

// static constexpr AscendC::PowerConfig config = { AscendC::PowerAlgo::DOUBLE_FLOAT_TECH };
// AscendC::Power<srcType, false, config>(dstLocal, scalarValue, srcLocal2);
```

AscendC::Power<T, false\>\(dstLocal, srcLocalBase, srcLocalExp\) 示例数据如下：

```
输入数据(srcLocalBase): [2 3 4 5 6 7 8 9]
输入数据(srcLocalExp): [4 3 2 1 4 3 2 1]
输出数据(dstLocal): [16 27 16 5 1296 343 64 9]
```

AscendC::Power<T, false\>\(dstLocal, scalarValueBase, srcLocalExp\) 示例数据如下：

```
输入数据(scalarValueBase): 2
输入数据(srcLocalExp): [4 3 2 1 4 3 2 1]
输出数据(dstLocal): [16 8 4 2 16 8 4 2]
```

AscendC::Power<T, false\>\(dstLocal, srcLocalBase, scalarValueExp\) 示例数据如下：

```
输入数据(srcLocalBase): [2 3 4 5 6 7 8 9]
输入数据(scalarValueExp): 4
输出数据(dstLocal): [16 81 256 625 1296 2401 4096 6561]
```

