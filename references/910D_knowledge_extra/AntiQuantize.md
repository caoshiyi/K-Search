# AntiQuantize<a name="ZH-CN_TOPIC_0000002523343754"></a>

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

按元素做伪量化计算，比如将int8\_t数据类型伪量化为half数据类型。**本接口最多支持输入为二维数据，不支持更高维度的输入。**

AntiQuantize与[AscendAntiQuant](AscendAntiQuant.md)的功能类似，本接口在不同量化场景下的形式更统一，因此推荐使用本接口。

本接口的伪量化策略包括PER\_TENSOR，PER\_CHANNEL，PER\_TOKEN，PER\_GROUP四种，反量化系数scale、offset在PER\_TENSOP场景下为标量，其余场景下为矢量，计算公式如下：

-   PER\_TENSOR场景 （按张量量化）：scale和offset的shape为\[1\]。

    <!-- img2text -->
$$dstTensor[i,j]=scale\cdot(srcTensor[i,j]-offset),\ 0\le i<m,\ 0\le j<n$$

-   PER\_CHANNEL场景（按通道量化）：srcTensor的shape为\[m, n\]，每个channel维度对应一个量化参数，scale和offset的shape为\[1, n\]。

    <!-- img2text -->
$$
dstTensor(i,j)=\operatorname{cast}\left(\operatorname{Round}\left(\frac{srcTensor(i,j)}{scale(0,j)}\right)+offset(0,j)\right)
$$

-   PER\_TOKEN场景 （按token量化）：srcTensor的每组token（token为n方向，共有m组token）中的元素共享一个量化参数，srcTensor的shape为\[m, n\]时，scale和offset的shape为\[m, 1\]。

    <!-- img2text -->
$$
output_{i,j} =
\begin{cases}
\operatorname{deqScale}_{i,0} \times (srcTensor_{i,j} - \operatorname{offset}_{i,0}), & \text{if } \operatorname{srcType} = \text{int8\_t} \\
\operatorname{deqScale}_{i,0} \times srcTensor_{i,j}, & \text{if } \operatorname{srcType} \in \{\text{hifloat8\_t}, \text{fp8\_e5m2\_t}, \text{fp8\_e4m3fn\_t}, \text{fp4x2\_e2m1\_t}, \text{fp4x2\_e1m2\_t}\}
\end{cases}
$$

-   PER\_GROUP场景 （按组量化）：定义group的计算方向为k方向，srcTensor在k方向上每groupSize个元素共享一组scale和offset。srcTensor的shape为\[m, n\]时，如果kDim=0，表示k是m方向，scale和offset的shape为\[\(m + groupSize - 1\) / groupSize, n\]；如果kDim=1，表示k是n方向，scale和offset的shape为\[m，\(n + groupSize - 1\) / groupSize\]。

    根据输入数据类型的不同，PER\_GROUP分为两种场景：fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（后续内容中简称为float4场景）和int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（后续内容中简称为b8场景）。

    -   fp4x2\_e2m1\_t/fp4x2\_e1m2\_t场景（float4场景）
        -   k为m方向，即公式中i轴为group的计算方向（kDim=0）：

            <!-- img2text -->
$$
\text{scale}_{i,j}=\text{scale}_{\left\lfloor \frac{i}{\text{groupSize}} \right\rfloor,j}
$$

        -   k为n方向，即公式中j轴为group的计算方向（kDim=1）：

            <!-- img2text -->
$$
Scale_i = \max_j \left( \left| x_{ij} \right| \right) / 448.0
$$

    -   int8\_t/hifloat8\_t/fp8\_e5m2\_t/fp8\_e4m3fn\_t场景（b8场景）
        -   k为m方向，即公式中i轴为group的计算方向（kDim=0）：

            <!-- img2text -->
$$
\mathrm{dst}[j][i] = \sum_{k = 0}^{K - 1} A[i][k] \times B[k][j]
$$

        -   k为n方向，即公式中j轴为group的计算方向（kDim=1）：

            <!-- img2text -->
$$
\left\{
\begin{aligned}
\text{若 } K > 1,\quad \operatorname{Offset}(i,j,k) &= i \times (N \times H \times W \times C_0) + j \times (H \times W \times C_0) + k \times C_0 \\
\text{若 } K = 1,\quad \operatorname{Offset}(i,j,k) &= i \times (N \times H \times W \times C_0) + k \times (W \times C_0) + j \times C_0
\end{aligned}
\right.
$$

## 函数原型<a name="section620mcpsimp"></a>

-   通过sharedTmpBuffer入参传入临时空间

    ```
    template <const AntiQuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void AntiQuantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor, const ScaleT& scale, const OffsetT& offset, const LocalTensor<uint8_t>& sharedTmpBuffer, const AntiQuantizeParams& params)
    ```

-   接口框架申请临时空间

    ```
    template <const AntiQuantizeConfig& config, typename DstT, typename SrcT, typename ScaleT, typename OffsetT>
    __aicore__ inline void AntiQuantize(const LocalTensor<DstT>& dstTensor, const LocalTensor<SrcT>& srcTensor, const ScaleT& scale, const OffsetT& offset, const AntiQuantizeParams& params)
    ```

由于该接口的内部实现中涉及复杂的数学计算，需要额外的临时空间来存储计算过程中的中间变量。临时空间支持**接口框架申请**和开发者**通过sharedTmpBuffer入参传入**两种方式。

-   接口框架申请临时空间，开发者无需申请，但是需要预留临时空间的大小。

-   通过sharedTmpBuffer入参传入，使用该tensor作为临时空间进行处理，接口框架不再申请。该方式开发者可以自行管理sharedTmpBuffer内存空间，并在接口调用完成后，复用该部分内存，内存不会反复申请释放，灵活性较高，内存利用率也较高。

接口框架申请的方式，开发者需要预留临时空间；通过sharedTmpBuffer传入的情况，开发者需要为sharedTmpBuffer申请空间。临时空间大小BufferSize的获取方式如下：通过[GetAntiQuantizeMaxMinTmpSize](GetAntiQuantizeMaxMinTmpSize.md)中提供的接口获取需要预留空间的范围大小。

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
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p194802031271"><a name="p194802031271"></a><a name="p194802031271"></a>用于配置伪量化相关信息，AntiQuantizeConfig类型，具体定义如下。</p>
<a name="screen174805314714"></a><a name="screen174805314714"></a><pre class="screen" codetype="Cpp" id="screen174805314714">struct AntiQuantizeConfig {
    AntiQuantizePolicy policy;
    bool hasOffset;
    int32_t kDim = 1;
 }</pre>
<a name="ul114808313710"></a><a name="ul114808313710"></a><ul id="ul114808313710"><li>policy：用于配置量化策略，枚举类型，具体定义如下。<a name="screen5480183111713"></a><a name="screen5480183111713"></a><pre class="screen" codetype="Cpp" id="screen5480183111713">enum class AntiQuantizePolicy : int32_t {
    PER_TENSOR,
    PER_CHANNEL,
    PER_TOKEN,
    PER_GROUP
}</pre>
</li><li>hasOffset：用于配置offset是否参与计算。<a name="ul44801311714"></a><a name="ul44801311714"></a><ul id="ul44801311714"><li>true：表示offset参与计算。</li><li>false：表示offset不参与计算。</li></ul>
</li><li>kDim：group的计算方向，即k方向。仅在PER_GROUP场景有效，支持的取值如下。<a name="ul54803312073"></a><a name="ul54803312073"></a><ul id="ul54803312073"><li>0：k轴是第0轴，即m方向为group的计算方向。</li><li>1：k轴是第1轴，即n方向为group的计算方向。</li></ul>
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
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p4299155115268"><a name="p4299155115268"></a><a name="p4299155115268"></a>源操作数的数据类型。接口内根据入参srcTensor自动推导数据类型，开发者无需配置该参数，保证srcTensor满足<a href="#table158181847102411">表3 输入输出支持的数据类型组合</a>即可。</p>
</td>
</tr>
<tr id="row789235465020"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p5892135418509"><a name="p5892135418509"></a><a name="p5892135418509"></a>ScaleT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p2892165465019"><a name="p2892165465019"></a><a name="p2892165465019"></a>scale的数据类型。接口内根据入参scale自动推导数据类型，开发者无需配置该参数。ScaleT可以为标量数据类型或LocalTensor类型。</p>
<p id="p779618341861"><a name="p779618341861"></a><a name="p779618341861"></a>注意：</p>
<a name="ul2044020331618"></a><a name="ul2044020331618"></a><ul id="ul2044020331618"><li>对于PER_TENSOR场景，scale为标量，ScaleT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP场景，scale为矢量，ScaleT只能为LocalTensor类型。</li></ul>
</td>
</tr>
<tr id="row1380111415524"><td class="cellrowborder" valign="top" width="19.259999999999998%" headers="mcps1.2.3.1.1 "><p id="p648512011526"><a name="p648512011526"></a><a name="p648512011526"></a>OffsetT</p>
</td>
<td class="cellrowborder" valign="top" width="80.74%" headers="mcps1.2.3.1.2 "><p id="p98102162315"><a name="p98102162315"></a><a name="p98102162315"></a>offset的数据类型。接口内根据入参offset自动推导数据类型，开发者无需配置该参数。OffsetT可以为标量数据类型或LocalTensor类型。</p>
<p id="p67438582710"><a name="p67438582710"></a><a name="p67438582710"></a>注意：</p>
<a name="ul1052919110813"></a><a name="ul1052919110813"></a><ul id="ul1052919110813"><li>对于PER_TENSOR量化策略，offset为标量，OffsetT只能为标量数据类型。</li><li>对于PER_CHANNEL、PER_TOKEN、PER_GROUP量化策略，offset为矢量，OffsetT只能为LocalTensor类型。</li></ul>
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
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p1754383817513"><a name="p1754383817513"></a><a name="p1754383817513"></a>输入数据伪量化时的缩放因子。</p>
</td>
</tr>
<tr id="row2891175273813"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p118911521386"><a name="p118911521386"></a><a name="p118911521386"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p78911552113812"><a name="p78911552113812"></a><a name="p78911552113812"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p1237485874015"><a name="p1237485874015"></a><a name="p1237485874015"></a>输入数据伪量化时的偏移量。</p>
<p id="p122061191946"><a name="p122061191946"></a><a name="p122061191946"></a><span id="ph14206896416"><a name="ph14206896416"></a><a name="ph14206896416"></a>Ascend 950PR/Ascend 950DT</span>，对于PER_GROUP量化的float4场景，offset不生效。</p>
</td>
</tr>
<tr id="row1747412296483"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p74741029204817"><a name="p74741029204817"></a><a name="p74741029204817"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p1747452954810"><a name="p1747452954810"></a><a name="p1747452954810"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p191160465422"><a name="p191160465422"></a><a name="p191160465422"></a>临时缓存。</p>
<p id="p979635010404"><a name="p979635010404"></a><a name="p979635010404"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_2"><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_2"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_2"><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_2"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_2"><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_2"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p5881016172817"><a name="p5881016172817"></a><a name="p5881016172817"></a>临时空间大小BufferSize的获取方式请参考<a href="GetAntiQuantizeMaxMinTmpSize.md">GetAntiQuantizeMaxMinTmpSize</a>。</p>
</td>
</tr>
<tr id="row439021517516"><td class="cellrowborder" valign="top" width="16.45%" headers="mcps1.2.4.1.1 "><p id="p75741927751"><a name="p75741927751"></a><a name="p75741927751"></a>params</p>
</td>
<td class="cellrowborder" valign="top" width="11.3%" headers="mcps1.2.4.1.2 "><p id="p157413277517"><a name="p157413277517"></a><a name="p157413277517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.25%" headers="mcps1.2.4.1.3 "><p id="p45741827355"><a name="p45741827355"></a><a name="p45741827355"></a>量化接口的参数，AntiQuantizeParams类型，具体定义如下。</p>
<a name="screen1157412271953"></a><a name="screen1157412271953"></a><pre class="screen" codetype="Cpp" id="screen1157412271953">struct AntiQuantizeParams {
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
<table><thead align="left"><tr id="row381964718248"><th class="cellrowborder" valign="top" width="19.708029197080293%" id="mcps1.2.5.1.1"><p id="p9111168018"><a name="p9111168018"></a><a name="p9111168018"></a>量化策略</p>
</th>
<th class="cellrowborder" valign="top" width="24.90750924907509%" id="mcps1.2.5.1.2"><p id="p1681934711240"><a name="p1681934711240"></a><a name="p1681934711240"></a><strong id="b16266153142818"><a name="b16266153142818"></a><a name="b16266153142818"></a>SrcT</strong></p>
</th>
<th class="cellrowborder" valign="top" width="21.95780421957804%" id="mcps1.2.5.1.3"><p id="p4819184792415"><a name="p4819184792415"></a><a name="p4819184792415"></a><strong id="b158285152286"><a name="b158285152286"></a><a name="b158285152286"></a>ScaleT</strong>/<strong id="b204229214284"><a name="b204229214284"></a><a name="b204229214284"></a>OffsetT</strong></p>
</th>
<th class="cellrowborder" valign="top" width="33.42665733426657%" id="mcps1.2.5.1.4"><p id="p48194471241"><a name="p48194471241"></a><a name="p48194471241"></a><strong id="b0801695281"><a name="b0801695281"></a><a name="b0801695281"></a>DstT</strong></p>
</th>
</tr>
</thead>
<tbody><tr id="row13538751414"><td class="cellrowborder" rowspan="8" valign="top" width="19.708029197080293%" headers="mcps1.2.5.1.1 "><p id="p10892134839"><a name="p10892134839"></a><a name="p10892134839"></a>PER_TENSOR/PER_CHANNEL</p>
</td>
<td class="cellrowborder" valign="top" width="24.90750924907509%" headers="mcps1.2.5.1.2 "><p id="p13180101423"><a name="p13180101423"></a><a name="p13180101423"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" width="21.95780421957804%" headers="mcps1.2.5.1.3 "><p id="p1053818513118"><a name="p1053818513118"></a><a name="p1053818513118"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="33.42665733426657%" headers="mcps1.2.5.1.4 "><p id="p253865311"><a name="p253865311"></a><a name="p253865311"></a>half</p>
</td>
</tr>
<tr id="row253818519117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p463510137216"><a name="p463510137216"></a><a name="p463510137216"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p16538051817"><a name="p16538051817"></a><a name="p16538051817"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1153815519114"><a name="p1153815519114"></a><a name="p1153815519114"></a>half</p>
</td>
</tr>
<tr id="row25387512117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p04231517127"><a name="p04231517127"></a><a name="p04231517127"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1053835914"><a name="p1053835914"></a><a name="p1053835914"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p75385511111"><a name="p75385511111"></a><a name="p75385511111"></a>half</p>
</td>
</tr>
<tr id="row165381459117"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p203623211625"><a name="p203623211625"></a><a name="p203623211625"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p753914512118"><a name="p753914512118"></a><a name="p753914512118"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p185391751217"><a name="p185391751217"></a><a name="p185391751217"></a>half</p>
</td>
</tr>
<tr id="row29020581708"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p49021581207"><a name="p49021581207"></a><a name="p49021581207"></a>fp8_e4m3fn_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1490214584019"><a name="p1490214584019"></a><a name="p1490214584019"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p690216581601"><a name="p690216581601"></a><a name="p690216581601"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row99025582010"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1890212582018"><a name="p1890212582018"></a><a name="p1890212582018"></a>fp8_e5m2_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p17902125818018"><a name="p17902125818018"></a><a name="p17902125818018"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p3902165819013"><a name="p3902165819013"></a><a name="p3902165819013"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row557719531308"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p10578155315012"><a name="p10578155315012"></a><a name="p10578155315012"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p115788532005"><a name="p115788532005"></a><a name="p115788532005"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p135781353805"><a name="p135781353805"></a><a name="p135781353805"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row165781538013"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p9235102417213"><a name="p9235102417213"></a><a name="p9235102417213"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p85789531009"><a name="p85789531009"></a><a name="p85789531009"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1857815311010"><a name="p1857815311010"></a><a name="p1857815311010"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row1881954718248"><td class="cellrowborder" rowspan="17" valign="top" width="19.708029197080293%" headers="mcps1.2.5.1.1 "><p id="p11121261807"><a name="p11121261807"></a><a name="p11121261807"></a>PER_TOKEN/PER_GROUP</p>
</td>
<td class="cellrowborder" rowspan="5" valign="top" width="24.90750924907509%" headers="mcps1.2.5.1.2 "><p id="p6329646165012"><a name="p6329646165012"></a><a name="p6329646165012"></a>int8_t</p>
</td>
<td class="cellrowborder" valign="top" width="21.95780421957804%" headers="mcps1.2.5.1.3 "><p id="p1881984712245"><a name="p1881984712245"></a><a name="p1881984712245"></a>half</p>
</td>
<td class="cellrowborder" valign="top" width="33.42665733426657%" headers="mcps1.2.5.1.4 "><p id="p15414132419508"><a name="p15414132419508"></a><a name="p15414132419508"></a>half</p>
</td>
</tr>
<tr id="row2819164722415"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p2819104711244"><a name="p2819104711244"></a><a name="p2819104711244"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p341462445013"><a name="p341462445013"></a><a name="p341462445013"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row2819184711242"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p178191447122415"><a name="p178191447122415"></a><a name="p178191447122415"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p13414324155014"><a name="p13414324155014"></a><a name="p13414324155014"></a>float</p>
</td>
</tr>
<tr id="row1481954718242"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p98190478243"><a name="p98190478243"></a><a name="p98190478243"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p64142247502"><a name="p64142247502"></a><a name="p64142247502"></a>half</p>
</td>
</tr>
<tr id="row9819747132418"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p148191847132412"><a name="p148191847132412"></a><a name="p148191847132412"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p64141624175010"><a name="p64141624175010"></a><a name="p64141624175010"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row1181904702416"><td class="cellrowborder" rowspan="5" valign="top" headers="mcps1.2.5.1.1 "><p id="p1978211567503"><a name="p1978211567503"></a><a name="p1978211567503"></a>hifloat8_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p2974122902715"><a name="p2974122902715"></a><a name="p2974122902715"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1414112412505"><a name="p1414112412505"></a><a name="p1414112412505"></a>half</p>
</td>
</tr>
<tr id="row12819174742414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p12974329192720"><a name="p12974329192720"></a><a name="p12974329192720"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1241502411507"><a name="p1241502411507"></a><a name="p1241502411507"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row12820134710244"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p19974132972710"><a name="p19974132972710"></a><a name="p19974132972710"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p84151624115010"><a name="p84151624115010"></a><a name="p84151624115010"></a>float</p>
</td>
</tr>
<tr id="row1382034714246"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p119742029112718"><a name="p119742029112718"></a><a name="p119742029112718"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p1941512410504"><a name="p1941512410504"></a><a name="p1941512410504"></a>half</p>
</td>
</tr>
<tr id="row16820147142414"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p2097432917271"><a name="p2097432917271"></a><a name="p2097432917271"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p14415192420507"><a name="p14415192420507"></a><a name="p14415192420507"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row6820144717249"><td class="cellrowborder" rowspan="5" valign="top" headers="mcps1.2.5.1.1 "><p id="p1666914411278"><a name="p1666914411278"></a><a name="p1666914411278"></a>fp8_e5m2_t/fp8_e4m3fn_t</p>
<p id="p20886140149"><a name="p20886140149"></a><a name="p20886140149"></a></p>
<p id="p108861471410"><a name="p108861471410"></a><a name="p108861471410"></a></p>
<p id="p12886642148"><a name="p12886642148"></a><a name="p12886642148"></a></p>
<p id="p488613421414"><a name="p488613421414"></a><a name="p488613421414"></a></p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p18669164419276"><a name="p18669164419276"></a><a name="p18669164419276"></a>half</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p144151924185019"><a name="p144151924185019"></a><a name="p144151924185019"></a>half</p>
</td>
</tr>
<tr id="row38201547172420"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p0669154418274"><a name="p0669154418274"></a><a name="p0669154418274"></a>bfloat16_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p241562417509"><a name="p241562417509"></a><a name="p241562417509"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row138201347102416"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p46699449270"><a name="p46699449270"></a><a name="p46699449270"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p10415112416508"><a name="p10415112416508"></a><a name="p10415112416508"></a>float</p>
</td>
</tr>
<tr id="row1882010471247"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p1669184415276"><a name="p1669184415276"></a><a name="p1669184415276"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p941592465012"><a name="p941592465012"></a><a name="p941592465012"></a>half</p>
</td>
</tr>
<tr id="row188202475248"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p176691449279"><a name="p176691449279"></a><a name="p176691449279"></a>float</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.2 "><p id="p241614246509"><a name="p241614246509"></a><a name="p241614246509"></a>bfloat16_t</p>
</td>
</tr>
<tr id="row9149111019142"><td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.5.1.1 "><p id="p935094292810"><a name="p935094292810"></a><a name="p935094292810"></a>fp4x2_e1m2_t/fp4x2_e2m1_t</p>
<p id="p18231111621415"><a name="p18231111621415"></a><a name="p18231111621415"></a>（当前均只支持PER_GROUP场景）</p>
</td>
<td class="cellrowborder" rowspan="2" valign="top" headers="mcps1.2.5.1.2 "><p id="p109091731111414"><a name="p109091731111414"></a><a name="p109091731111414"></a>fp8_e8m0_t</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.5.1.3 "><p id="p1314911011145"><a name="p1314911011145"></a><a name="p1314911011145"></a>half</p>
</td>
</tr>
<tr id="row1972181219144"><td class="cellrowborder" valign="top" headers="mcps1.2.5.1.1 "><p id="p37211512111416"><a name="p37211512111416"></a><a name="p37211512111416"></a>bfloat16_t</p>
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
constexpr static AntiQuantizePolicy policy = AntiQuantizePolicy::PER_TOKEN;
constexpr static AntiQuantizeConfig config = {policy, hasOffset, -1};
AntiQuantizeParams params;
params.m = m;
params.n = n;
params.groupSize = 0; // 仅PER_GROUP场景有效
AntiQuantize<config>(dstLocal, srcLocal, scale, offset, params);
```

