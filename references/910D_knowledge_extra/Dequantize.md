# Dequantize<a name="ZH-CN_TOPIC_0000002554344775"></a>

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

## 功能说明<a name="section1030413110138"></a>

按元素做反量化计算，比如将int32\_t数据类型反量化为half/float等数据类型。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

Dequantize与[AscendDequant](AscendDequant.md)的功能类似，本接口在不同量化场景下的形式更统一，因此推荐使用本接口。

本接口的反量化策略包括PER\_TENSOR，PER\_CHANNEL，PER\_TOKEN，PER\_GROUP四种，反量化系数scale在PER\_TENSOR场景下为标量，其余场景下为矢量，具体计算公式如下：

-   PER\_TENSOR场景 （按张量反量化）：scale和offset的shape为\[1\]。

    <!-- img2text -->
$$dstTensor = (srcTensor + offset) \times scale$$

-   PER\_CHANNEL场景 （按通道反量化）：srcTensor的shape为\[m, n\]，每个channel维度对应一个量化参数，scale和offset的shape为\[1, n\]。

    <!-- img2text -->
$$dstTensor_i = \left( srcTensor_i + offset_i \right) \times scale_i$$

-   PER\_TOKEN场景 （按token反量化）：srcTensor的每组token（token为n方向，共有m组token）中的元素共享一个量化参数，srcTensor的shape为\[m, n\]时，scale和offset的shape为\[m, 1\]。

    <!-- img2text -->
$$dstTensor_i = srcTensor_i \times scale_i + offset_i$$

-   PER\_GROUP场景 （按组反量化）：定义group的计算方向为k方向，srcTensor在k方向上每groupSize个元素共享一组scale和offset。srcTensor的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale和offset的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale和offset的shape为\[m，\(n + groupSize - 1\) / groupSize\]。
    -   k为m方向，即公式中i轴为group的计算方向（kDim=0）：

        <!-- img2text -->
$$
dstTensor(i, j) = \left( srcTensor(i, j) + offset\!\left(\left\lfloor \frac{i}{groupSize} \right\rfloor, j\right) \right) * scale\!\left(\left\lfloor \frac{i}{groupSize} \right\rfloor, j\right)
$$

    -   k为n方向，即公式中j轴为group的计算方向（kDim=1）：

        <!-- img2text -->
$$
\text{dstLocal}_{i,j}=
\begin{cases}
\text{src0Local}_{i,j}+\text{src1Local}_{i,j+\left\lfloor \dfrac{j}{n}\right\rfloor \cdot m}, & 0\le j<n \\
\text{src0Local}_{i,j}\times \text{src1Local}_{i,j+\left\lfloor \dfrac{j}{n}\right\rfloor \cdot m}, & n\le j<2n
\end{cases}
$$

## 函数原型<a name="section196311132152614"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <const DequantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void Dequantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor, const ScaleT& scale, const OffsetT& offset, const LocalTensor<uint8_t>& sharedTmpBuffer, const DequantizeParams& params)
    ```

-   接口框架申请临时空间

    ```
    template <const DequantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void Dequantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor, const ScaleT& scale, const OffsetT& offset, const DequantizeParams& params)
    ```

    由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

    -   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

    -   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

    接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetDequantizeMaxMinTmpSize](GetDequantizeMaxMinTmpSize.md)中提供的GetDequantizeMaxMinTmpSize接口获取需要预留空间的范围大小。

## 参数说明<a name="section1193017460420"></a>

**表 1**  模板参数说明

<a name="table575571914269"></a>
<table><thead align="left"><tr id="row18755131942614"><th class="cellrowborder" valign="top" width="19.39%" id="mcps1.2.3.1.1"><p id="p675519193268"><a name="p675519193268"></a><a name="p675519193268"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.61%" id="mcps1.2.3.1.2"><p id="p375511918267"><a name="p375511918267"></a><a name="p375511918267"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row14755141911264"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p47551198266"><a name="p47551198266"></a><a name="p47551198266"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p13892754135017"><a name="p13892754135017"></a><a name="p13892754135017"></a>用于配置反量化相关信息，DequantizeConfig类型，具体定义如下。</p>
<a name="screen1434141315715"></a><a name="screen1434141315715"></a><pre class="screen" codetype="Cpp" id="screen1434141315715">struct DequantizeConfig {
    DequantizePolicy policy;
    bool hasOffset = false;
    int32_t kDim = 1;
}</pre>
<a name="ul558513317417"></a><a name="ul558513317417"></a><ul id="ul558513317417"><li>policy：用于配置量化策略，枚举类型，具体定义如下。<a name="screen10546253131615"></a><a name="screen10546253131615"></a><pre class="screen" codetype="Cpp" id="screen10546253131615">enum class DequantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP
}</pre>
</li></ul>
<a name="ul1859516965710"></a><a name="ul1859516965710"></a><ul id="ul1859516965710"><li>hasOffset：预留参数，目前仅支持配置为false。</li><li>kDim：group的计算方向，即k方向。仅在PER_GROUP场景有效，支持的取值如下。<a name="ul95959916579"></a><a name="ul95959916579"></a><ul id="ul95959916579"><li>0：k轴是第0轴，即m方向为group的计算方向。</li><li>1：k轴是第1轴，即n方向为group的计算方向。</li></ul>
</li></ul>
</td>
</tr>
<tr id="row69924559439"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p392964134410"><a name="p392964134410"></a><a name="p392964134410"></a>DstT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p2182948165115"><a name="p2182948165115"></a><a name="p2182948165115"></a>目的操作数的数据类型。接口内根据入参dstTensor自动推导数据类型，开发者无需配置该参数，保证dstTensor满足<a href="#table1963437121712">表3 输入输出支持的数据类型组合</a>即可。</p>
</td>
</tr>
<tr id="row6356241194912"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p143561041144915"><a name="p143561041144915"></a><a name="p143561041144915"></a>SrcT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p4299155115268"><a name="p4299155115268"></a><a name="p4299155115268"></a>源操作数的数据类型。接口内根据入参srcTensor自动推导数据类型，开发者无需配置该参数，保证srcTensor满足<a href="#table1963437121712">表3 输入输出支持的数据类型组合</a>即可。</p>
</td>
</tr>
<tr id="row9756719122620"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p330515236446"><a name="p330515236446"></a><a name="p330515236446"></a>ScaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p1616392714412"><a name="p1616392714412"></a><a name="p1616392714412"></a>scale的数据类型。接口内根据入参scale自动推导数据类型，开发者无需配置该参数。ScaleT可以为标量数据类型或LocalTensor类型。</p>
<p id="p779618341861"><a name="p779618341861"></a><a name="p779618341861"></a>注意：</p>
<a name="ul2044020331618"></a><a name="ul2044020331618"></a><ul id="ul2044020331618"><li>对于PER_TENSOR场景，scale为标量，ScaleT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP场景，scale为矢量，ScaleT只能为LocalTensor类型。</li></ul>
</td>
</tr>
<tr id="row168921122316"><td class="cellrowborder" valign="top" width="19.39%" headers="mcps1.2.3.1.1 "><p id="p111267258235"><a name="p111267258235"></a><a name="p111267258235"></a>OffsetT</p>
</td>
<td class="cellrowborder" valign="top" width="80.61%" headers="mcps1.2.3.1.2 "><p id="p98102162315"><a name="p98102162315"></a><a name="p98102162315"></a>offset的数据类型。接口内根据入参offset自动推导数据类型，开发者无需配置该参数。OffsetT可以为标量数据类型或LocalTensor类型。</p>
<p id="p67438582710"><a name="p67438582710"></a><a name="p67438582710"></a>注意：</p>
<a name="ul1052919110813"></a><a name="ul1052919110813"></a><ul id="ul1052919110813"><li>对于PER_TENSOR量化策略，offset为标量，OffsetT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP量化策略，offset为矢量，OffsetT只能为LocalTensor类型。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="15.55%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="9.24%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.21%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p143185337247"><a name="p143185337247"></a><a name="p143185337247"></a>源操作数。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p2836135311718"><a name="p2836135311718"></a><a name="p2836135311718"></a>假设srcTensor的shape为[m, n]，每行数据（即n个输入数据）所占字节数要求<strong id="b733515211256"><a name="b733515211256"></a><a name="b733515211256"></a>32字节对齐</strong>。</p>
</td>
</tr>
<tr id="row617218172310"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p16660132211315"><a name="p16660132211315"></a><a name="p16660132211315"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p156601822153115"><a name="p156601822153115"></a><a name="p156601822153115"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p13434937543"><a name="p13434937543"></a><a name="p13434937543"></a>输入数据反量化时的缩放因子。</p>
</td>
</tr>
<tr id="row194612231282"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p1046132302810"><a name="p1046132302810"></a><a name="p1046132302810"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p11640193317280"><a name="p11640193317280"></a><a name="p11640193317280"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p18304103872815"><a name="p18304103872815"></a><a name="p18304103872815"></a>输入数据反量化时的偏移量。当前为预留参数，可配置为0或空Tensor。</p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetDequantizeMaxMinTmpSize.md">GetDequantizeMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row850382835820"><td class="cellrowborder" valign="top" width="15.55%" headers="mcps1.2.4.1.1 "><p id="p714563310589"><a name="p714563310589"></a><a name="p714563310589"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="9.24%" headers="mcps1.2.4.1.2 "><p id="p210914395588"><a name="p210914395588"></a><a name="p210914395588"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.21%" headers="mcps1.2.4.1.3 "><p id="p3885143933717"><a name="p3885143933717"></a><a name="p3885143933717"></a>反量化接口的参数，DequantizeParams类型，定义如下。</p>
<a name="screen08851139153718"></a><a name="screen08851139153718"></a><pre class="screen" codetype="Cpp" id="screen08851139153718">struct DequantizeParams {
        uint32_t m;
        uint32_t n;
        uint32_t groupSize = 0;
}</pre>
<a name="ul1888563913371"></a><a name="ul1888563913371"></a><ul id="ul1888563913371"><li>m：m方向元素个数。</li><li>n：n方向元素个数。n值对应的数据大小需满足32字节对齐的要求，即shape最后一维为n的输入或输出均需要满足该维度上32字节对齐的要求。</li><li>groupSize：PER_GROUP场景有效，表示groupSize行/列数据共用一个scale。groupSize的取值必须大于0且是32的整倍数。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  输入输出支持的数据类型组合

<a name="table1963437121712"></a>
<table><thead align="left"><tr id="row16963183711175"><th class="cellrowborder" valign="top" width="14.221422142214221%" id="mcps1.2.5.1.1"><p id="p1859924818720"><a name="p1859924818720"></a><a name="p1859924818720"></a>量化策略</p>
</th>
<th class="cellrowborder" valign="top" width="21.35213521352135%" id="mcps1.2.5.1.2"><p id="p12599948576"><a name="p12599948576"></a><a name="p12599948576"></a>dstTensor</p>
</th>
<th class="cellrowborder" valign="top" width="29.112911291129116%" id="mcps1.2.5.1.3"><p id="p15599174817715"><a name="p15599174817715"></a><a name="p15599174817715"></a>srcTensor</p>
</th>
<th class="cellrowborder" valign="top" width="35.31353135313531%" id="mcps1.2.5.1.4"><p id="p2059984817711"><a name="p2059984817711"></a><a name="p2059984817711"></a>scale/offset</p>
</th>
</tr>
</thead>
<tbody><tr id="row575511819151"><td class="cellrowborder" rowspan="4" valign="top" width="14.221422142214221%" headers="mcps1.2.5.1.1 "><p id="p10482155515231"><a name="p10482155515231"></a><a name="p10482155515231"></a>PER_TENSOR</p>
</td>
<td class="cellrowborder" valign="top" width="21.35213521352135%" headers="mcps1.2.5.1.2 "><p id="p629415350189"><a name="p629415350189"></a><a name="p629415350189"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" width="29.112911291129116%" headers="mcps1.2.5.1.3 "><p id="p3294153531811"><a name="p3294153531811"></a><a name="p3294153531811"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="35.31353135313531%" headers="mcps1.2.5.1.4 "><p id="p0294133517184"><a name="p0294133517184"></a><a name="p0294133517184"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row167551382152"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p97757397183"><a name="p97757397183"></a><a name="p97757397183"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p11775183911186"><a name="p11775183911186"></a><a name="p11775183911186"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p7775103914187"><a name="p7775103914187"></a><a name="p7775103914187"></a>float</p>
</td>
</tr>
<tr id="row575513815155"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p11711310172310"><a name="p11711310172310"></a><a name="p11711310172310"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1671121013233"><a name="p1671121013233"></a><a name="p1671121013233"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1771101062315"><a name="p1771101062315"></a><a name="p1771101062315"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row147553811158"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1587213295186"><a name="p1587213295186"></a><a name="p1587213295186"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16872829191819"><a name="p16872829191819"></a><a name="p16872829191819"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p13872129151819"><a name="p13872129151819"></a><a name="p13872129151819"></a>float</p>
</td>
</tr>
<tr id="row10755118191518"><td class="cellrowborder" rowspan="5" valign="top" width="14.221422142214221%" headers="mcps1.2.5.1.1 "><p id="p46611054131718"><a name="p46611054131718"></a><a name="p46611054131718"></a>PER_CHANNEL</p>
</td>
<td class="cellrowborder" valign="top" width="21.35213521352135%" headers="mcps1.2.5.1.2 "><p id="p8964203741720"><a name="p8964203741720"></a><a name="p8964203741720"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="29.112911291129116%" headers="mcps1.2.5.1.3 "><p id="p896463713171"><a name="p896463713171"></a><a name="p896463713171"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="35.31353135313531%" headers="mcps1.2.5.1.4 "><p id="p496411375175"><a name="p496411375175"></a><a name="p496411375175"></a>uint64_t</p>
<p id="p0792388399"><a name="p0792388399"></a><a name="p0792388399"></a>注意：当scale的数据类型是uint64_t时，其中的低32位数据是参与计算的float类型数据，高32位本接口不使用。</p>
</td>
</tr>
<tr id="row87551984158"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p16964143711712"><a name="p16964143711712"></a><a name="p16964143711712"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p125026399197"><a name="p125026399197"></a><a name="p125026399197"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p8964153701717"><a name="p8964153701717"></a><a name="p8964153701717"></a>float</p>
</td>
</tr>
<tr id="row10754587155"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1396463781710"><a name="p1396463781710"></a><a name="p1396463781710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16204019195"><a name="p16204019195"></a><a name="p16204019195"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p10964163715171"><a name="p10964163715171"></a><a name="p10964163715171"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row775415851510"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p7964133731719"><a name="p7964133731719"></a><a name="p7964133731719"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p253912402195"><a name="p253912402195"></a><a name="p253912402195"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p596414373170"><a name="p596414373170"></a><a name="p596414373170"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row2075418819152"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p16178207509"><a name="p16178207509"></a><a name="p16178207509"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p4617202014509"><a name="p4617202014509"></a><a name="p4617202014509"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1161712085012"><a name="p1161712085012"></a><a name="p1161712085012"></a>float</p>
</td>
</tr>
<tr id="row975419831518"><td class="cellrowborder" rowspan="10" valign="top" width="14.221422142214221%" headers="mcps1.2.5.1.1 "><p id="p85998117263"><a name="p85998117263"></a><a name="p85998117263"></a>PER_TOKEN/PER_GROUP</p>
</td>
<td class="cellrowborder" valign="top" width="21.35213521352135%" headers="mcps1.2.5.1.2 "><p id="p1155881512172"><a name="p1155881512172"></a><a name="p1155881512172"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="29.112911291129116%" headers="mcps1.2.5.1.3 "><p id="p1355861561711"><a name="p1355861561711"></a><a name="p1355861561711"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" width="35.31353135313531%" headers="mcps1.2.5.1.4 "><p id="p6558115181719"><a name="p6558115181719"></a><a name="p6558115181719"></a>half</p>
</td>
</tr>
<tr id="row9754128161518"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p156405237171"><a name="p156405237171"></a><a name="p156405237171"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p156401323161718"><a name="p156401323161718"></a><a name="p156401323161718"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p10640923121714"><a name="p10640923121714"></a><a name="p10640923121714"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row975410851517"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p7976329181710"><a name="p7976329181710"></a><a name="p7976329181710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p9976122951720"><a name="p9976122951720"></a><a name="p9976122951720"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p2976102919174"><a name="p2976102919174"></a><a name="p2976102919174"></a>float</p>
</td>
</tr>
<tr id="row57531087154"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p12364143591718"><a name="p12364143591718"></a><a name="p12364143591718"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p83641135181711"><a name="p83641135181711"></a><a name="p83641135181711"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p8364435121710"><a name="p8364435121710"></a><a name="p8364435121710"></a>float</p>
</td>
</tr>
<tr id="row1375310841514"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p11684144041710"><a name="p11684144041710"></a><a name="p11684144041710"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p268444019172"><a name="p268444019172"></a><a name="p268444019172"></a>int32_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p868415409177"><a name="p868415409177"></a><a name="p868415409177"></a>float</p>
</td>
</tr>
<tr id="row67539861513"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p97168142257"><a name="p97168142257"></a><a name="p97168142257"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p117162148256"><a name="p117162148256"></a><a name="p117162148256"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p771621412255"><a name="p771621412255"></a><a name="p771621412255"></a>half</p>
</td>
</tr>
<tr id="row1175313851518"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p079619102514"><a name="p079619102514"></a><a name="p079619102514"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p9791119142516"><a name="p9791119142516"></a><a name="p9791119142516"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p6791319162515"><a name="p6791319162515"></a><a name="p6791319162515"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row575310801514"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p187282226253"><a name="p187282226253"></a><a name="p187282226253"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p97286222251"><a name="p97286222251"></a><a name="p97286222251"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p772842232518"><a name="p772842232518"></a><a name="p772842232518"></a>float</p>
</td>
</tr>
<tr id="row575316831515"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p244762613259"><a name="p244762613259"></a><a name="p244762613259"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p6447122613256"><a name="p6447122613256"></a><a name="p6447122613256"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p4447326152520"><a name="p4447326152520"></a><a name="p4447326152520"></a>float</p>
</td>
</tr>
<tr id="row675217851516"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p0219830112516"><a name="p0219830112516"></a><a name="p0219830112516"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p19219030112513"><a name="p19219030112513"></a><a name="p19219030112513"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p10219133019254"><a name="p10219133019254"></a><a name="p10219133019254"></a>float</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   连续计算方向（即n方向）的数据量要求32字节对齐。

## 调用示例<a name="section5708172172815"></a>

```
// 注意m,n需从外部传入
constexpr static DequantizePolicy policy = DequantizePolicy::PER_TOKEN;  // 可修改为PER_CHANNEL/PER_GROUP;
constexpr static DequantizeConfig config = {policy, false, -1};
DequantizeParams params;
params.m = m;
params.n = n;
params.groupSize = 0;  // 仅PER_GROUP场景下生效;
Dequantize<config>(dstLocal, srcLocal, scaleLocal, offsetLocal, params);  // offsetLocal为预留参数，可配置为空Tensor;
```

