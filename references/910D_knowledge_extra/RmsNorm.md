# RmsNorm<a name="ZH-CN_TOPIC_0000002523304034"></a>

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

实现对shape大小为\[B，S，H\]的输入数据的RmsNorm归一化，其计算公式如下：

<!-- img2text -->
$$
y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2 + \varepsilon}} \cdot \gamma_i
$$

其中，γ为缩放系数，ε为防除零的权重系数。

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <typename T, bool isBasicBlock = false>
    __aicore__ inline void RmsNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gammaLocal, const LocalTensor<uint8_t>& sharedTmpBuffer, const T epsilon, const RmsNormTiling& tiling)
    ```

-   接口框架申请临时空间

    ```
    template <typename T, bool isBasicBlock = false>
    __aicore__ inline void RmsNorm(const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const LocalTensor<T>& gammaLocal, const T epsilon, const RmsNormTiling& tiling)
    ```

由于该接口的内部实现中涉及复杂的计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为tensor申请空间。临时空间大小BufferSize的获取方式如下：通过[RmsNorm Tiling](RmsNorm-Tiling.md)中提供的GetRmsNormMaxMinTmpSize接口获取所需最大和最小临时空间大小，最小空间可以保证功能正确，最大空间用于提升性能。

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
<p id="p131411813171015"><a name="p131411813171015"></a><a name="p131411813171015"></a><span id="ph1168842372812"><a name="ph1168842372812"></a><a name="ph1168842372812"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：half、float。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p1682112447268"><a name="p1682112447268"></a><a name="p1682112447268"></a>isBasicBlock</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1626481084910"><a name="p1626481084910"></a><a name="p1626481084910"></a>srcTensor和dstTensor的shape信息和Tiling切分策略满足基本块要求的情况下，可以使能该参数用于提升性能，默认不使能。基本块要求srcTensor和dstTensor的shape需要满足如下条件：</p>
<a name="ul1426419106495"></a><a name="ul1426419106495"></a><ul id="ul1426419106495"><li>last轴即H的长度为64的倍数，但小于2048；</li><li>非last轴长度（B*S）为8的倍数。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="17.77%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.66%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.57000000000001%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p462911347151"><a name="p462911347151"></a><a name="p462911347151"></a>dstLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p186293346150"><a name="p186293346150"></a><a name="p186293346150"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p242825624218"><a name="p242825624218"></a><a name="p242825624218"></a>目的操作数。</p>
<p id="p16911647191712"><a name="p16911647191712"></a><a name="p16911647191712"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p9255193274511"><a name="p9255193274511"></a><a name="p9255193274511"></a>dstLocal的shape和源操作数srcLocal需要保持一致。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1762920347151"><a name="p1762920347151"></a><a name="p1762920347151"></a>srcLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p1662903414157"><a name="p1662903414157"></a><a name="p1662903414157"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p082871620172"><a name="p082871620172"></a><a name="p082871620172"></a>源操作数。</p>
<p id="p15450144034510"><a name="p15450144034510"></a><a name="p15450144034510"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p220914264371"><a name="p220914264371"></a><a name="p220914264371"></a>shape为[B, S, H]，尾轴H长度需要满足32字节对齐。</p>
</td>
</tr>
<tr id="row81221355101812"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1412365512189"><a name="p1412365512189"></a><a name="p1412365512189"></a>gammaLocal</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p121236558181"><a name="p121236558181"></a><a name="p121236558181"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p16591756121820"><a name="p16591756121820"></a><a name="p16591756121820"></a>缩放系数。</p>
<p id="p10883142218184"><a name="p10883142218184"></a><a name="p10883142218184"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p2060756121810"><a name="p2060756121810"></a><a name="p2060756121810"></a>shape需要与srcLocal和dstLocal的尾轴H长度相等，即shape为[H]。</p>
</td>
</tr>
<tr id="row20749938181910"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p1575033814199"><a name="p1575033814199"></a><a name="p1575033814199"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p77501738191912"><a name="p77501738191912"></a><a name="p77501738191912"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p1323018409190"><a name="p1323018409190"></a><a name="p1323018409190"></a>临时空间。</p>
<p id="p14203184218188"><a name="p14203184218188"></a><a name="p14203184218188"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_3"><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_3"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_3"><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_3"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_3"><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_3"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p104071111204211"><a name="p104071111204211"></a><a name="p104071111204211"></a>接口内部复杂计算时用于存储中间变量，由开发者提供。</p>
<p id="p2230154015198"><a name="p2230154015198"></a><a name="p2230154015198"></a>临时空间大小BufferSize的获取方式请参考<a href="RmsNorm-Tiling.md">RmsNorm Tiling</a>。</p>
</td>
</tr>
<tr id="row041413392198"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p4630634141515"><a name="p4630634141515"></a><a name="p4630634141515"></a>epsilon</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p263018345154"><a name="p263018345154"></a><a name="p263018345154"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p13630123491515"><a name="p13630123491515"></a><a name="p13630123491515"></a>防除零的权重系数，数据类型需要与srcLocal/dstLocal保持一致。</p>
</td>
</tr>
<tr id="row45915087"><td class="cellrowborder" valign="top" width="17.77%" headers="mcps1.2.4.1.1 "><p id="p95215986"><a name="p95215986"></a><a name="p95215986"></a>tiling</p>
</td>
<td class="cellrowborder" valign="top" width="9.66%" headers="mcps1.2.4.1.2 "><p id="p6551519818"><a name="p6551519818"></a><a name="p6551519818"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.57000000000001%" headers="mcps1.2.4.1.3 "><p id="p1851815387"><a name="p1851815387"></a><a name="p1851815387"></a>RmsNorm计算所需Tiling信息，Tiling信息的获取请参考<a href="RmsNorm-Tiling.md">RmsNorm Tiling</a>。</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   dstLocal和gammaLocal的Tensor空间不允许复用。
-   当前仅支持ND格式的输入，不支持其他格式。
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   当srcLocal的原始shape中H轴非32字节对齐时，开发者需要对原始输入在H轴方向补齐数据到32字节对齐，API的计算结果会覆盖dstLocal中对应srcLocal补齐位置的数据。

## 调用示例<a name="section94691236101419"></a>

完整的调用样例可参考[RmsNorm样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/03_normalization/rmsnorm)。

```
// dstLocal：存放RmsNorm计算结果的Tensor
// srcLocal：参与计算的输入Tensor
// gammaLocal：输入张量，归一化后数据的缩放系数γ
// epsilon：防除0的权重系数ε
// tiling：Tiling数据，从Host侧获取

// // 若尾轴的长度（H）不超过2040且为64的倍数，同时非尾轴长度（B*S）为8的倍数，可设置isBasicBlock = true提升性能
AscendC::RmsNorm<dataType, isBasicBlock>(dstLocal, srcLocal, gammaLocal, epsilon, tiling);
```

