# Fmod<a name="ZH-CN_TOPIC_0000002554423503"></a>

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

按元素计算两个浮点数a, b相除后的余数。计算公式如下：

<!-- img2text -->
$$
\operatorname{out}_{i} = a_{i} - \operatorname{Trunc}\left(\frac{a_{i}}{b_{i}}\right) \times b_{i}
$$

其中，$\operatorname{Trunc}$ 为向零取整操作。

<!-- img2text -->
$$\operatorname{Fmod}(a, b) = a - \operatorname{Trunc}\left(\frac{a}{b}\right) \times b$$

其中，Trunc为向零取整操作。举例如下：

Fmod\(2.0, 1.5\) = 0.5

Fmod\(-3.0, 1.1\) = -0.8

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false, const FmodConfig& config = DEFAULT_FMOD_CONFIG>
        __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false, const FmodConfig& config = DEFAULT_FMOD_CONFIG>
        __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const LocalTensor<uint8_t>& sharedTmpBuffer)
        ```

-   接口框架申请临时空间
    -   源操作数Tensor全部/部分参与计算

        ```
        template <typename T, bool isReuseSource = false, const FmodConfig& config = DEFAULT_FMOD_CONFIG>
        __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor, const uint32_t calCount)
        ```

    -   源操作数Tensor全部参与计算

        ```
        template <typename T, bool isReuseSource = false, const FmodConfig& config = DEFAULT_FMOD_CONFIG>
        __aicore__ inline void Fmod(const LocalTensor<T>& dstTensor, const LocalTensor<T>& src0Tensor, const LocalTensor<T>& src1Tensor)
        ```

由于该接口的内部实现中涉及精度转换。需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[GetFmodMaxMinTmpSize](GetFmodMaxMinTmpSize.md)中提供的接口获取需要预留空间的大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1743114241217"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.37%" id="mcps1.2.3.1.1"><p id="p17431024628"><a name="p17431024628"></a><a name="p17431024628"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.63%" id="mcps1.2.3.1.2"><p id="p204317241428"><a name="p204317241428"></a><a name="p204317241428"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p14431424724"><a name="p14431424724"></a><a name="p14431424724"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p18431192420215"><a name="p18431192420215"></a><a name="p18431192420215"></a>操作数的数据类型。</p>
<p id="p1688223162815"><a name="p1688223162815"></a><a name="p1688223162815"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row78561818101715"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="zh-cn_topic_0000001538537601_p1682112447268"><a name="zh-cn_topic_0000001538537601_p1682112447268"></a><a name="zh-cn_topic_0000001538537601_p1682112447268"></a>isReuseSource</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001538537601_p98212044172612"><a name="zh-cn_topic_0000001538537601_p98212044172612"></a><a name="zh-cn_topic_0000001538537601_p98212044172612"></a>是否允许修改源操作数。该参数预留，传入默认值false即可。</p>
</td>
</tr>
<tr id="row19674913154"><td class="cellrowborder" valign="top" width="19.37%" headers="mcps1.2.3.1.1 "><p id="p179771456161314"><a name="p179771456161314"></a><a name="p179771456161314"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.63%" headers="mcps1.2.3.1.2 "><p id="p1977105611138"><a name="p1977105611138"></a><a name="p1977105611138"></a>Fmod计算的相关配置。此参数可选配，FmodConfig类型，具体定义如下方代码所示，其中参数的含义为：</p>
<div class="p" id="p1267610594274"><a name="p1267610594274"></a><a name="p1267610594274"></a>algo：指定Fmod的算法。该参数支持的取值如下：<a name="ul827932118545"></a><a name="ul827932118545"></a><ul id="ul827932118545"><li>NORMAL：algo的默认值，使用模拟的普通模式，支持的数据类型为：half、float。</li><li>ITERATION_COMPENSATION：迭代补偿的高精度模式，支持的数据类型为：float。</li></ul>
</div>
<p id="p126764595271"><a name="p126764595271"></a><a name="p126764595271"></a>iterationNum：迭代补偿的高精度模式下的迭代补偿轮次，该参数仅在algo为ITERATION_COMPENSATION模式下生效，轮次范围1至11，默认值为11次。迭代轮次越多，结果精度越高，但性能会相应降低。使用时，可根据两个浮点数的指数位差异来选择迭代轮次，float类型共有8位指数位，src0Tensor和src1Tensor之间的指数位差异不应超过24*iterationNum。</p>
</td>
</tr>
</tbody>
</table>

```
constexpr uint32_t FMOD_ITERATION_NUM_MAX = 11;
enum class FmodAlgo {
    NORMAL = 0,
    ITERATION_COMPENSATION = 1,
};
struct FmodConfig {
    FmodAlgo algo = FmodAlgo::NORMAL;
    uint32_t iterationNum = FMOD_ITERATION_NUM_MAX;
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
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p17341123413535"><a name="p17341123413535"></a><a name="p17341123413535"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row11848103091920"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p58481330191917"><a name="p58481330191917"></a><a name="p58481330191917"></a>src0Tensor、src1Tensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p158485305196"><a name="p158485305196"></a><a name="p158485305196"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p177171853175314"><a name="p177171853175314"></a><a name="p177171853175314"></a>源操作数。</p>
<p id="p163731255155313"><a name="p163731255155313"></a><a name="p163731255155313"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p515144315188"><a name="p515144315188"></a><a name="p515144315188"></a>源操作数的数据类型需要与目的操作数保持一致。</p>
</td>
</tr>
<tr id="row4848123011192"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p1313415271911"><a name="p1313415271911"></a><a name="p1313415271911"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p5133352201914"><a name="p5133352201914"></a><a name="p5133352201914"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p0400131017545"><a name="p0400131017545"></a><a name="p0400131017545"></a>临时空间。</p>
<p id="p11947511105415"><a name="p11947511105415"></a><a name="p11947511105415"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>用于Fmod内部复杂计算时存储中间变量，由开发者提供。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetFmodMaxMinTmpSize.md">GetFmodMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row17767122113114"><td class="cellrowborder" valign="top" width="16.470000000000002%" headers="mcps1.2.4.1.1 "><p id="p15767132115116"><a name="p15767132115116"></a><a name="p15767132115116"></a>calCount</p>
</td>
<td class="cellrowborder" valign="top" width="11.29%" headers="mcps1.2.4.1.2 "><p id="p37673214115"><a name="p37673214115"></a><a name="p37673214115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.24000000000001%" headers="mcps1.2.4.1.3 "><p id="p184961858133"><a name="p184961858133"></a><a name="p184961858133"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   源操作数src0Tensor与src1Tensor的数据长度必须保持一致。
-   **不支持源操作数与目的操作数地址重叠。**
-   不支持sharedTmpBuffer与源操作数和目的操作数地址重叠。
-   对于Ascend 950PR/Ascend 950DT，模板参数config中的algo为ITERATION\_COMPENSATION迭代补偿模式下，操作数的数据类型仅支持float。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例请参考[fmod算子样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/00_math/fmod)。

```
// dstLocal: 存放Fmod计算结果的输出Tensor
// src0Local: 存放Fmod计算除数的输入Tensor
// src1Local: 存放Fmod计算被除数的输入Tensor
// sharedTmpBuffer: 存放Fmod计算过程中临时缓存的Tensor

// 算子输入的数据类型为half, 需要参与计算的元素个数为512
AscendC::Fmod(dstLocal, src0Local, src1Local, sharedTmpBuffer, 512);
```

```
__aicore__ constexpr AscendC::FmodConfig GetConfig() {
    return { .algo = AscendC::FmodAlgo::ITERATION_COMPENSATION, .iterationNum = 11 };
}
static constexpr AscendC::FmodConfig config = GetConfig();
AscendC::Fmod<float, false, config>(dstLocal, src0Local, src1Local, sharedTmpBuffer, 512);
```

结果示例如下：

```
输入数据(src0Local): [ 0.5 -6.3 5.5 ... 11.1 -11.6]
输入数据(src1Local): [ 2.1 3.0 -0.3 ...  5.6   5.9]
输出数据(dstLocal):  [ 0.5 -0.3 0.1 ...  5.5  -5.7]
```

