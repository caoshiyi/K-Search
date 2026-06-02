# Quantize<a name="ZH-CN_TOPIC_0000002523344304"></a>

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

按元素做量化计算，将高精度数据转换为低精度数据。本接口的量化策略包括PER\_TENSOR，PER\_CHANNEL，PER\_TOKEN和PER\_GROUP四种，每种量化策略均支持配置舍入模式。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

Quantize与[AscendQuant](AscendQuant.md)的功能类似，Quantize在PER\_TENSOR、PER\_CHANNEL量化场景，扩展了配置舍入模式的功能，因此推荐使用本接口。

-   PER\_TENSOR量化:整个srcTensor对应一个量化参数，scale和offset的shape为\[1\]。

    <!-- img2text -->
$$dstTensor = \operatorname{quantize}(srcTensor, scale, offset, sqrtMode, roundMode, dstType)$$

$$dstTensor_i = \operatorname{round}\left(\left(\frac{srcTensor_i}{scale_i} \cdot \frac{1}{scale_i}\right) + offset_i\right)$$

$$round = \begin{cases}
\operatorname{ceil}, & \text{when roundMode = 0} \\
\operatorname{away\_from\_zero}, & \text{when roundMode = 1} \\
\operatorname{floor}, & \text{when roundMode = 2} \\
\operatorname{half\_up}, & \text{when roundMode = 3} \\
\operatorname{half\_to\_even}, & \text{when roundMode = 4}
\end{cases}$$

$$\text{当 } sqrtMode = \text{true 时。}$$

-   PER\_CHANNEL量化：srcTensor的shape为\[m, n\]，每个channel维度对应一个量化参数，scale和offset的shape为\[1, n\]。

    <!-- img2text -->
$$dstTensor(i, j) = \operatorname{clamp}\left(\operatorname{round}\left(\frac{srcTensor(i, j)}{scale(0, j)}\right) + offset(0, j)\right)$$

-   PER\_TOKEN量化：srcTensor的每组token（token为n方向，共有m组token）中的元素共享一个量化参数，srcTensor的shape为\[m, n\]时，scale和offset的shape为\[m, 1\]。

    <!-- img2text -->
$$
\text{deqScale}_{i}=\text{scale}_{i}\cdot 2^{-\text{offset}_{i}}
$$

-   PER\_GROUP量化：定义group的计算方向为k方向，srcTensor在k方向上每groupSize个元素共享一组scale和offset。srcTensor的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale和offset的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale和offset的shape为\[m，\(n + groupSize - 1\) / groupSize\]。

    根据输出数据类型的不同，PER\_GROUP量化分为两种场景：fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（后续内容中简称为float4场景）和int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（后续内容中简称为b8场景）。

    -   fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（float4场景）
        -   kDim = 0:

            <!-- img2text -->
$$
\text{quantScale} = \text{round}\left(\frac{\text{coef} \times 16}{\text{srcScale}}\right)
$$

$$
\text{quantOffset} = \text{round}\left(\frac{\text{srcOffset} \times \text{coef} \times 16}{\text{srcScale}}\right)
$$

        -   kDim = 1:

            <!-- img2text -->
$$dstLocal[n] = \sum_{i=0}^{srcK-1} src0Local[i \times srcM + n] \times src1Local[i],\ 0 \leq n < srcM$$

    -   int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（b8场景）
        -   kDim = 0：

            <!-- img2text -->
$$
dstLocal[row, col] = src0Local[row, col] \times src1Local[row]
$$

        -   kDim = 1:

            ![](figures/zh-cn_formulaimage_0000002523346658.png)

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void Quantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor, const LocalTensor<uint8_t>& sharedTmpBuffer, const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
    ```

-   接口框架申请临时空间

    ```
    template <const QuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void Quantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor,const ScaleT& scale, const OffsetT& offset, const QuantizeParams& params)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetQuantizeMaxMinTmpSize](GetQuantizeMaxMinTmpSize.md)中提供的接口获取需要预留空间的范围大小。

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table589125485010"></a>
<table><thead align="left"><tr id="row489125425015"><th class="cellrowborder" valign="top" width="19.259999999999998%" id="mcps1.2.3.1.1"><p id="p6891105420508"><a name="p6891105420508"></a><a name="p6891105420508"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="80.74%" id="mcps1.2.3.1.2"><p id="p17891145445014"><a name="p17891145445014"></a><a name="p17891145445014"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1884072618717"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p8480203115716"><a name="p8480203115716"></a><a name="p8480203115716"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p194802031271"><a name="p194802031271"></a><a name="p194802031271"></a>用于配置量化计算相关信息，QuantizeConfig类型，具体定义如下。</p>
<a name="screen174805314714"></a><a name="screen174805314714"></a><pre class="screen" codetype="Cpp" id="screen174805314714">struct QuantizeConfig {
    QuantizePolicy policy;
    bool hasOffset;
    RoundMode roundMode = RoundMode::CAST_RINT;
    int32_t kDim = 1;
 }</pre>
<a name="ul114808313710"></a><a name="ul114808313710"></a><ul id="ul114808313710"><li>policy：用于配置量化策略，枚举类型，具体定义如下。<a name="screen5480183111713"></a><a name="screen5480183111713"></a><pre class="screen" codetype="Cpp" id="screen5480183111713">enum class QuantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP
}</pre>
</li><li>hasOffset：用于配置offset是否参与计算。<a name="ul16493154020500"></a><a name="ul16493154020500"></a><ul id="ul16493154020500"><li>true：表示offset参与计算。</li><li>false：表示offset不参与计算。</li></ul>
</li><li>roundMode：量化过程中，数据由高精度数据类型转换为低精度数据类型的舍入模式，支持的取值有：CAST_RINT、CAST_ROUND、CAST_FLOOR、CAST_CEIL、CAST_TRUNC、CAST_HYBRID，各个舍入模式的详细介绍请参考<a href="Cast.md#table235404962912">精度转换规则</a>。不同数据类型的量化支持不同的舍入模式，当量化过程中使用了不支持的舍入模式时，将回退到默认的舍入模式；例如，bfloat16_t数据类型量化为hifloat8_t数据类型时，如果配置的roundMode为不支持的CAST_RINT，实际执行量化时将回退到默认的roundMode（CAST_ROUND）。不同数据类型支持的舍入模式请见<a href="#table158181847102411">表3</a>。</li><li>kDim：group的计算方向，即k方向。仅在PER__GROUP场景有效，支持的取值如下：<a name="ul11517466516"></a><a name="ul11517466516"></a><ul id="ul11517466516"><li>0：k轴是第0轴，即m方向为group的计算方向。</li><li>1：k轴是第1轴，即n方向为group的计算方向。</li></ul>
</li></ul>
</td>
</tr>
<tr id="row1589195475011"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p188911054155015"><a name="p188911054155015"></a><a name="p188911054155015"></a>DstT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2182948165115"><a name="p2182948165115"></a><a name="p2182948165115"></a>目的操作数的数据类型。接口内根据入参dstTensor自动推导数据类型，开发者无需配置该参数，保证dstTensor满足<a href="#table158181847102411">表3 输入输出支持的数据类型组合</a>即可。</p>
</td>
</tr>
<tr id="row1891165475020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p68921754135010"><a name="p68921754135010"></a><a name="p68921754135010"></a>SrcT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p5182154805113"><a name="p5182154805113"></a><a name="p5182154805113"></a>源操作数的数据类型。接口内根据入参srcTensor自动推导数据类型，开发者无需配置该参数，保证srcTensor满足<a href="#table158181847102411">表3 输入输出支持的数据类型组合</a>即可。</p>
</td>
</tr>
<tr id="row789235465020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p5892135418509"><a name="p5892135418509"></a><a name="p5892135418509"></a>ScaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2892165465019"><a name="p2892165465019"></a><a name="p2892165465019"></a>缩放因子scale的数据类型。接口内根据入参scale自动推导数据类型，开发者无需配置该参数。ScaleT可以为标量数据类型或LocalTensor类型。</p>
<p id="p779618341861"><a name="p779618341861"></a><a name="p779618341861"></a>注意：</p>
<a name="ul2044020331618"></a><a name="ul2044020331618"></a><ul id="ul2044020331618"><li>对于PER_TENSOR量化策略，scale为标量，ScaleT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP量化策略，scale为矢量，ScaleT只能为LocalTensor类型。</li></ul>
</td>
</tr>
<tr id="row1380111415524"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p648512011526"><a name="p648512011526"></a><a name="p648512011526"></a>OffsetT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p66481524175515"><a name="p66481524175515"></a><a name="p66481524175515"></a>offset的数据类型。接口内根据入参offset自动推导数据类型，开发者无需配置该参数。OffsetT可以为标量数据类型或LocalTensor类型。</p>
<p id="p67438582710"><a name="p67438582710"></a><a name="p67438582710"></a>注意：</p>
<a name="ul1052919110813"></a><a name="ul1052919110813"></a><ul id="ul1052919110813"><li>对于PER_TENSOR量化策略，offset为标量，OffsetT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP量化策略，offset可以是标量或者矢量，OffsetT可以为标量数据类型，也可以为LocalTensor类型。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table44731299481"></a>
<table><thead align="left"><tr id="row247482914489"><th class="cellrowborder" valign="top" width="16.45%" id="mcps1.2.4.1.1"><p id="p147413295483"><a name="p147413295483"></a><a name="p147413295483"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.3%" id="mcps1.2.4.1.2"><p id="p1147432994819"><a name="p1147432994819"></a><a name="p1147432994819"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.25%" id="mcps1.2.4.1.3"><p id="p74749297483"><a name="p74749297483"></a><a name="p74749297483"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row12474329104814"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p1047411294482"><a name="p1047411294482"></a><a name="p1047411294482"></a>dstTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p047412984813"><a name="p047412984813"></a><a name="p047412984813"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p3989161814016"><a name="p3989161814016"></a><a name="p3989161814016"></a>目的操作数。</p>
<p id="p1747492917489"><a name="p1747492917489"></a><a name="p1747492917489"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row18474729124817"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p54741029164810"><a name="p54741029164810"></a><a name="p54741029164810"></a>srcTensor</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p144741829194814"><a name="p144741829194814"></a><a name="p144741829194814"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p6914123244017"><a name="p6914123244017"></a><a name="p6914123244017"></a>源操作数。</p>
<p id="p1493334184019"><a name="p1493334184019"></a><a name="p1493334184019"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
</td>
</tr>
<tr id="row94292371051"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p19542163819516"><a name="p19542163819516"></a><a name="p19542163819516"></a>scale</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p1154323812514"><a name="p1154323812514"></a><a name="p1154323812514"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p1754383817513"><a name="p1754383817513"></a><a name="p1754383817513"></a>输入数据量化时的缩放因子。</p>
</td>
</tr>
<tr id="row2891175273813"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p118911521386"><a name="p118911521386"></a><a name="p118911521386"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p78911552113812"><a name="p78911552113812"></a><a name="p78911552113812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p1237485874015"><a name="p1237485874015"></a><a name="p1237485874015"></a>输入数据量化时的偏移量。</p>
<p id="p122061191946"><a name="p122061191946"></a><a name="p122061191946"></a><span id="ph14206896416"><a name="ph14206896416"></a><a name="ph14206896416"></a>Ascend 950PR/Ascend 950DT</span>，对于PER_GROUP量化的float4场景，offset不生效。</p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetQuantizeMaxMinTmpSize.md">GetQuantizeMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row439021517516"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p75741927751"><a name="p75741927751"></a><a name="p75741927751"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p157413277517"><a name="p157413277517"></a><a name="p157413277517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p45741827355"><a name="p45741827355"></a><a name="p45741827355"></a>量化接口的参数，QuantizeParams类型，具体定义如下。</p>
<a name="screen1157412271953"></a><a name="screen1157412271953"></a><pre class="screen" codetype="Cpp" id="screen1157412271953">struct QuantizeParams {
        uint32_t m;
        uint32_t n;
        uint32_t groupSize = 0;
}</pre>
<a name="ul145741427855"></a><a name="ul145741427855"></a><ul id="ul145741427855"><li>m：m方向元素个数。</li><li>n：n方向元素个数。n值对应的数据大小需满足32字节对齐的要求，即shape最后一维为n的输入或输出均需要满足该维度上32字节对齐的要求。</li><li>groupSize：PER_GROUP场景有效，表示groupSize行/列数据共用一个scale/offset。groupSize的取值必须大于0且是32的整倍数。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 3**  输入输出支持的数据类型组合

<a name="table158181847102411"></a>
<table><thead align="left"><tr id="row381964718248"><th class="cellrowborder" valign="top" width="21.617838216178384%" id="mcps1.2.5.1.1"><p id="p1944002416383"><a name="p1944002416383"></a><a name="p1944002416383"></a>SrcT</p>
</th>
<th class="cellrowborder" valign="top" width="21.27787221277872%" id="mcps1.2.5.1.2"><p id="p2044012242382"><a name="p2044012242382"></a><a name="p2044012242382"></a>ScaleT/OffsetT</p>
</th>
<th class="cellrowborder" valign="top" width="32.42675732426757%" id="mcps1.2.5.1.3"><p id="p04401024173816"><a name="p04401024173816"></a><a name="p04401024173816"></a>DstT</p>
</th>
<th class="cellrowborder" valign="top" width="24.67753224677532%" id="mcps1.2.5.1.4"><p id="p134407245389"><a name="p134407245389"></a><a name="p134407245389"></a>roundMode</p>
</th>
</tr>
</thead>
<tbody><tr id="row13538751414"><td class="cellrowborder" valign="top" width="21.617838216178384%" headers="mcps1.2.5.1.1 "><p id="p203953543811"><a name="p203953543811"></a><a name="p203953543811"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="21.27787221277872%" headers="mcps1.2.5.1.2 "><p id="p839183517382"><a name="p839183517382"></a><a name="p839183517382"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="32.42675732426757%" headers="mcps1.2.5.1.3 "><p id="p624910549395"><a name="p624910549395"></a><a name="p624910549395"></a>fp8_e5m2_t/fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.67753224677532%" headers="mcps1.2.5.1.4 "><a name="ul1515143312404"></a><a name="ul1515143312404"></a><ul id="ul1515143312404"><li>CAST_RINT(默认)</li></ul>
</td>
</tr>
<tr id="row253818519117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p939193533813"><a name="p939193533813"></a><a name="p939193533813"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p123983553813"><a name="p123983553813"></a><a name="p123983553813"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row25387512117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1539235103811"><a name="p1539235103811"></a><a name="p1539235103811"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1640123519383"><a name="p1640123519383"></a><a name="p1640123519383"></a>float</p>
</td>
</tr>
<tr id="row165381459117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p440113523812"><a name="p440113523812"></a><a name="p440113523812"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p184033514384"><a name="p184033514384"></a><a name="p184033514384"></a>float</p>
</td>
</tr>
<tr id="row29020581708"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p140123583818"><a name="p140123583818"></a><a name="p140123583818"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p9404358382"><a name="p9404358382"></a><a name="p9404358382"></a>float</p>
</td>
</tr>
<tr id="row99025582010"><td class="cellrowborder" valign="top" width="21.617838216178384%" headers="mcps1.2.5.1.1 "><p id="p440183520389"><a name="p440183520389"></a><a name="p440183520389"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="21.27787221277872%" headers="mcps1.2.5.1.2 "><p id="p94053503812"><a name="p94053503812"></a><a name="p94053503812"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="32.42675732426757%" headers="mcps1.2.5.1.3 "><p id="p122231037134017"><a name="p122231037134017"></a><a name="p122231037134017"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.67753224677532%" headers="mcps1.2.5.1.4 "><a name="ul737514118406"></a><a name="ul737514118406"></a><ul id="ul737514118406"><li>CAST_ROUND(默认)</li><li>CAST_HYBRID</li></ul>
</td>
</tr>
<tr id="row557719531308"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1740173513812"><a name="p1740173513812"></a><a name="p1740173513812"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p540435163815"><a name="p540435163815"></a><a name="p540435163815"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row165781538013"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p124083515389"><a name="p124083515389"></a><a name="p124083515389"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1240133503814"><a name="p1240133503814"></a><a name="p1240133503814"></a>float</p>
</td>
</tr>
<tr id="row1881954718248"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1040735143819"><a name="p1040735143819"></a><a name="p1040735143819"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1940153515387"><a name="p1940153515387"></a><a name="p1940153515387"></a>float</p>
</td>
</tr>
<tr id="row2819164722415"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p740113510388"><a name="p740113510388"></a><a name="p740113510388"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p134033510386"><a name="p134033510386"></a><a name="p134033510386"></a>float</p>
</td>
</tr>
<tr id="row2819184711242"><td class="cellrowborder" valign="top" width="21.617838216178384%" headers="mcps1.2.5.1.1 "><p id="p6401354387"><a name="p6401354387"></a><a name="p6401354387"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="21.27787221277872%" headers="mcps1.2.5.1.2 "><p id="p84023519381"><a name="p84023519381"></a><a name="p84023519381"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="32.42675732426757%" headers="mcps1.2.5.1.3 "><p id="p472714514011"><a name="p472714514011"></a><a name="p472714514011"></a>int8_t</p>
</td>
<td class="cellrowborder" rowspan="10" valign="top" width="24.67753224677532%" headers="mcps1.2.5.1.4 "><a name="ul7172450134014"></a><a name="ul7172450134014"></a><ul id="ul7172450134014"><li>CAST_RINT(默认)</li><li>CAST_ROUND</li><li>CAST_FLOOR</li><li>CAST_CEIL</li><li>CAST_TRUNC</li></ul>
</td>
</tr>
<tr id="row1481954718242"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p114093519387"><a name="p114093519387"></a><a name="p114093519387"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p140035103817"><a name="p140035103817"></a><a name="p140035103817"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row9819747132418"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p3401235173816"><a name="p3401235173816"></a><a name="p3401235173816"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1040535143814"><a name="p1040535143814"></a><a name="p1040535143814"></a>float</p>
</td>
</tr>
<tr id="row1181904702416"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1400352385"><a name="p1400352385"></a><a name="p1400352385"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p154143514383"><a name="p154143514383"></a><a name="p154143514383"></a>float</p>
</td>
</tr>
<tr id="row12819174742414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1141835103812"><a name="p1141835103812"></a><a name="p1141835103812"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1241113519389"><a name="p1241113519389"></a><a name="p1241113519389"></a>float</p>
</td>
</tr>
<tr id="row12820134710244"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1741635193810"><a name="p1741635193810"></a><a name="p1741635193810"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p17413352383"><a name="p17413352383"></a><a name="p17413352383"></a>half</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" headers="mcps1.2.5.1.3 "><p id="p62028547402"><a name="p62028547402"></a><a name="p62028547402"></a>fp4x2_e1m2_t/fp4x2_e2m1_t</p>
<p id="p17202165474018"><a name="p17202165474018"></a><a name="p17202165474018"></a>（当前均只支持PER_GROUP场景）</p>
</td>
</tr>
<tr id="row1382034714246"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p482204614412"><a name="p482204614412"></a><a name="p482204614412"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p38221146134411"><a name="p38221146134411"></a><a name="p38221146134411"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row672512165446"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p6726516124415"><a name="p6726516124415"></a><a name="p6726516124415"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1672621624415"><a name="p1672621624415"></a><a name="p1672621624415"></a>float</p>
</td>
</tr>
<tr id="row16820147142414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1275174974414"><a name="p1275174974414"></a><a name="p1275174974414"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p641735193812"><a name="p641735193812"></a><a name="p641735193812"></a>float</p>
</td>
</tr>
<tr id="row6820144717249"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1741103553812"><a name="p1741103553812"></a><a name="p1741103553812"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p104113593818"><a name="p104113593818"></a><a name="p104113593818"></a>float</p>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section38228281712"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   **不支持源操作数与目的操作数地址重叠。**
-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   输入输出操作数参与计算的数据长度要求32字节对齐。
-   连续计算方向（即n方向）的数据量要求32字节对齐。
-   PER\_GROUP量化的float4场景不支持offset，该场景下模板参数config中的hasOffset参数必须配置为false。

## 调用示例<a name="section642mcpsimp"></a>

```
// 注意m,n需从外部传入
constexpr static QuantizeConfig config = {QuantizePolicy::PER_TOKEN, hasOffset, RoundMode::CAST_ROUND, -1};
QuantizeParams params;
params.m = m;
params.n = n;
params.groupSize = 0; // 仅PER_GROUP场景有效
Quantize<config>(dstLocal, srcLocal, scale, offset, params);
```

