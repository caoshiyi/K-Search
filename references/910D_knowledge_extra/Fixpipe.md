# Fixpipe<a name="ZH-CN_TOPIC_0000002523303664"></a>

## 产品支持情况<a name="section1550532418810"></a>

<a name="table38301303189"></a>
<table><thead align="left"><tr id="row20831180131817"><th class="cellrowborder" valign="top" width="57.99999999999999%" id="mcps1.1.3.1.1"><p id="p1883113061818"><a name="p1883113061818"></a><a name="p1883113061818"></a><span id="ph20833205312295"><a name="ph20833205312295"></a><a name="ph20833205312295"></a>产品</span></p>
</th>
<th class="cellrowborder" align="center" valign="top" width="42%" id="mcps1.1.3.1.2"><p id="p783113012187"><a name="p783113012187"></a><a name="p783113012187"></a>是否支持</p>
</th>
</tr>
</thead>
<tbody><tr id="row1272474920205"><td class="cellrowborder" valign="top" width="57.99999999999999%" headers="mcps1.1.3.1.1 "><p id="p17301775812"><a name="p17301775812"></a><a name="p17301775812"></a><span id="ph2272194216543"><a name="ph2272194216543"></a><a name="ph2272194216543"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p173073381243"><a name="p173073381243"></a><a name="p173073381243"></a>支持包含FixpipeParamsV220/FixpipeParamsC310参数的接口。</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

矩阵计算完成后，对结果进行处理，例如对计算结果进行量化操作，并把数据从CO1搬迁到Global Memory中。

## 函数原型<a name="section620mcpsimp"></a>

-   传入FixpipeParamsV220
    -   通路CO1-\>GM，不使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsV220& intriParams)
        ```

    -   通路CO1-\>GM，使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsV220& intriParams)
        ```

-   传入FixpipeParamsM300
    -   通路CO1-\>UB，不使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsM300& intriParams)
        ```

    -   通路CO1-\>UB，使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams)
        ```

    -   通路CO1-\>GM，不使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsM300& intriParams)
        ```

    -   通路CO1-\>GM，使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR, typename S = uint64_t, typename Std::enable_if<Std::is_same<PrimT<S>, uint64_t>::value, bool>::type = true>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<S>& cbufWorkspace, const FixpipeParamsM300& intriParams)
        ```

-   传入FixpipeParamsC310
    -   通路CO1-\>L1 Buffer，不使能tensor量化功能

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams)
        ```

    -   通路CO1-\>L1 Buffer，使能tensor量化功能

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsC310<config.format>& intriParams)
        ```

    -   通路CO1-\>UB，不使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams)
        ```

    -   通路CO1-\>UB，使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const LocalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsC310<config.format>& intriParams)
        ```

    -   通路CO1-\>GM，不使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const FixpipeParamsC310<config.format>& intriParams)
        ```

    -   通路CO1-\>GM，使能tensor量化功能：

        ```
        template <typename T, typename U, const FixpipeConfig& config = CFG_ROW_MAJOR>
        __aicore__ inline void Fixpipe(const GlobalTensor<T>& dst, const LocalTensor<U>& src, const LocalTensor<uint64_t>& cbufWorkspace, const FixpipeParamsC310<config.format>& intriParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="zh-cn_topic_0000001429830437_row118356578583"><th class="cellrowborder" valign="top" width="14.08%" id="mcps1.2.3.1.1"><p id="zh-cn_topic_0000001429830437_p48354572582"><a name="zh-cn_topic_0000001429830437_p48354572582"></a><a name="zh-cn_topic_0000001429830437_p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="85.92%" id="mcps1.2.3.1.2"><p id="zh-cn_topic_0000001429830437_p583535795817"><a name="zh-cn_topic_0000001429830437_p583535795817"></a><a name="zh-cn_topic_0000001429830437_p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="zh-cn_topic_0000001429830437_row1835857145817"><td class="cellrowborder" valign="top" width="14.08%" headers="mcps1.2.3.1.1 "><p id="p511145143017"><a name="p511145143017"></a><a name="p511145143017"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="85.92%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000001429830437_p168351657155818"><a name="zh-cn_topic_0000001429830437_p168351657155818"></a><a name="zh-cn_topic_0000001429830437_p168351657155818"></a>目的操作数数据类型。</p>
</td>
</tr>
<tr id="row1648615377"><td class="cellrowborder" valign="top" width="14.08%" headers="mcps1.2.3.1.1 "><p id="p1212015191874"><a name="p1212015191874"></a><a name="p1212015191874"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="85.92%" headers="mcps1.2.3.1.2 "><p id="p1912061914715"><a name="p1912061914715"></a><a name="p1912061914715"></a>源操作数数据类型。</p>
</td>
</tr>
<tr id="row937273310304"><td class="cellrowborder" valign="top" width="14.08%" headers="mcps1.2.3.1.1 "><p id="p19405113493017"><a name="p19405113493017"></a><a name="p19405113493017"></a>config</p>
</td>
<td class="cellrowborder" valign="top" width="85.92%" headers="mcps1.2.3.1.2 "><p id="p1740523443011"><a name="p1740523443011"></a><a name="p1740523443011"></a>Fixpipe相关配置参数，类型为FixpipeConfig。取值如下：</p>
<a name="ul17842203693817"></a><a name="ul17842203693817"></a><ul id="ul17842203693817"><li><strong id="b640553483017"><a name="b640553483017"></a><a name="b640553483017"></a>CFG_ROW_MAJOR（默认取值）</strong>：使能NZ2ND，输出数据格式为ND格式。</li><li>CFG_NZ: 不使能NZ2ND，输出数据格式为NZ格式。</li><li>CFG_COLUMN_MAJOR：使能NZ2DN，输出数据格式为DN格式。</li></ul>
<a name="screen4405834193014"></a><a name="screen4405834193014"></a><pre class="screen" codetype="Cpp" id="screen4405834193014">struct FixpipeConfig {
    CO2Layout format;
    bool isToUB; // 用于用户指定目的地址的位置是否是UB 
};
enum class CO2Layout : uint8_t {
    NZ = 0, // 输出数据格式仍为NZ格式。
    ROW_MAJOR, // 使能NZ2ND，输出数据格式为ND格式。
    COLUMN_MAJOR, // 使能NZ2DN，输出数据格式为DN格式。
};
constexpr FixpipeConfig CFG_NZ = {CO2Layout::NZ};
constexpr FixpipeConfig CFG_ROW_MAJOR = {CO2Layout::ROW_MAJOR};
constexpr FixpipeConfig CFG_COLUMN_MAJOR = {CO2Layout::COLUMN_MAJOR};</pre>
</td>
</tr>
<tr id="row162252057304"><td class="cellrowborder" valign="top" width="14.08%" headers="mcps1.2.3.1.1 "><p id="p182256513019"><a name="p182256513019"></a><a name="p182256513019"></a>S</p>
</td>
<td class="cellrowborder" valign="top" width="85.92%" headers="mcps1.2.3.1.2 "><p id="p973319214339"><a name="p973319214339"></a><a name="p973319214339"></a>cbufWorkspace的数据类型。</p>
<a name="ul7112101853218"></a><a name="ul7112101853218"></a><ul id="ul7112101853218"><li>当目的操作数、源操作数 、cbufWorkspace使用基础数据类型时，模板参数S必须为uint64_t类型，否则编译失败。</li><li>当目的操作数、源操作数 、cbufWorkspace使用<a href="TensorTrait.md">TensorTrait</a>类型时，模板参数S的LiteType必须为uint64_t类型，否则编译失败。</li></ul>
<p id="p187025511315"><a name="p187025511315"></a><a name="p187025511315"></a>模板参数S后一个模板参数仅用于上述数据类型检查，用户无需关注。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="table8955841508"></a>
<table><thead align="left"><tr id="row15956194105014"><th class="cellrowborder" valign="top" width="14.18141814181418%" id="mcps1.2.4.1.1"><p id="p7956144195014"><a name="p7956144195014"></a><a name="p7956144195014"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.291029102910292%" id="mcps1.2.4.1.2"><p id="p1295624145013"><a name="p1295624145013"></a><a name="p1295624145013"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="75.52755275527552%" id="mcps1.2.4.1.3"><p id="p16956144145011"><a name="p16956144145011"></a><a name="p16956144145011"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row27001444121011"><td class="cellrowborder" valign="top" width="14.18141814181418%" headers="mcps1.2.4.1.1 "><p id="p1259095171013"><a name="p1259095171013"></a><a name="p1259095171013"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.291029102910292%" headers="mcps1.2.4.1.2 "><p id="p5590551181016"><a name="p5590551181016"></a><a name="p5590551181016"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="75.52755275527552%" headers="mcps1.2.4.1.3 "><p id="p255711621817"><a name="p255711621817"></a><a name="p255711621817"></a>目的操作数，类型为<a href="LocalTensor.md">LocalTensor</a>或<a href="GlobalTensor.md">GlobalTensor</a>。</p>
<a name="ul16146134605518"></a><a name="ul16146134605518"></a><ul id="ul16146134605518"><li>针对LocalTensor：<p id="p1463883873214"><a name="p1463883873214"></a><a name="p1463883873214"></a><span id="ph1263893863210"><a name="ph1263893863210"></a><a name="ph1263893863210"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、half、bfloat16_t、float、half、int32_t。</p>
</li><li>针对GlobalTensor：<p id="p71581752135517"><a name="p71581752135517"></a><a name="p71581752135517"></a><span id="ph14920205581619"><a name="ph14920205581619"></a><a name="ph14920205581619"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t、uint8_t、hifloat8_t、fp8_e4m3fn_t、half、bfloat16_t、int32_t、float。</p>
<p id="p15158175275514"><a name="p15158175275514"></a><a name="p15158175275514"></a>数据格式为NZ或ND<span id="ph19679103564512"><a name="ph19679103564512"></a><a name="ph19679103564512"></a>或DN</span>格式。经过Fixpipe处理，在量化操作之后，会将矩阵计算中多申请的数据删除。</p>
</li></ul>
</td>
</tr>
<tr id="row4956154125018"><td class="cellrowborder" valign="top" width="14.18141814181418%" headers="mcps1.2.4.1.1 "><p id="p142871414131614"><a name="p142871414131614"></a><a name="p142871414131614"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10.291029102910292%" headers="mcps1.2.4.1.2 "><p id="p628711148165"><a name="p628711148165"></a><a name="p628711148165"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.52755275527552%" headers="mcps1.2.4.1.3 "><p id="p49136785815"><a name="p49136785815"></a><a name="p49136785815"></a>源操作数，支持的TPosition为CO1，为Mmad接口计算的结果，类型为LocalTensor数据结构的定义请参考<a href="LocalTensor.md">LocalTensor</a>。支持的数据类型为float/int32_t，支持的TPosition为CO1，数据格式为NZ格式。起始地址需要满足64B对齐。</p>
</td>
</tr>
<tr id="row0800184819273"><td class="cellrowborder" valign="top" width="14.18141814181418%" headers="mcps1.2.4.1.1 "><p id="p4800104842720"><a name="p4800104842720"></a><a name="p4800104842720"></a>intriParams</p>
</td>
<td class="cellrowborder" valign="top" width="10.291029102910292%" headers="mcps1.2.4.1.2 "><p id="p1780015485273"><a name="p1780015485273"></a><a name="p1780015485273"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.52755275527552%" headers="mcps1.2.4.1.3 "><p id="p451365612118"><a name="p451365612118"></a><a name="p451365612118"></a>Fixpipe搬运参数，具体定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_fixpipe.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
<p id="p1800164862718"><a name="p1800164862718"></a><a name="p1800164862718"></a>参数说明请参考<a href="Fixpipe.md#table16798200165314">表3</a>。</p>
</td>
</tr>
<tr id="row13475143221812"><td class="cellrowborder" valign="top" width="14.18141814181418%" headers="mcps1.2.4.1.1 "><p id="p104759322185"><a name="p104759322185"></a><a name="p104759322185"></a>cbufWorkspace</p>
</td>
<td class="cellrowborder" valign="top" width="10.291029102910292%" headers="mcps1.2.4.1.2 "><p id="p347516327184"><a name="p347516327184"></a><a name="p347516327184"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="75.52755275527552%" headers="mcps1.2.4.1.3 "><p id="p146208155523"><a name="p146208155523"></a><a name="p146208155523"></a>量化参数，类型为LocalTensor&lt;uint64_t&gt;，支持的TPosition为A1。仅当quantPre为VDEQF16/VQF322B8_PRE/VREQ8时支持，quantPre介绍请参考FixpipeParamsV220/FixpipeParamsM300结构体中quantPre部分。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Fixpipe搬运参数（FixpipeParamsV220/FixpipeParamsM300）结构体说明

<a name="table16798200165314"></a>
<table><thead align="left"><tr id="row1979815015536"><th class="cellrowborder" valign="top" width="13.459999999999999%" id="mcps1.2.4.1.1"><p id="p1979819019535"><a name="p1979819019535"></a><a name="p1979819019535"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.05%" id="mcps1.2.4.1.2"><p id="p1179812011538"><a name="p1179812011538"></a><a name="p1179812011538"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.49000000000001%" id="mcps1.2.4.1.3"><p id="p187980014533"><a name="p187980014533"></a><a name="p187980014533"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row779817065319"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p8798160155314"><a name="p8798160155314"></a><a name="p8798160155314"></a>nSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p167983025319"><a name="p167983025319"></a><a name="p167983025319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p2384183185914"><a name="p2384183185914"></a><a name="p2384183185914"></a>源NZ矩阵在N方向上的大小。</p>
<a name="ul1614820539210"></a><a name="ul1614820539210"></a><ul id="ul1614820539210"><li>不使能NZ2ND功能<p id="p3913117638"><a name="p3913117638"></a><a name="p3913117638"></a>若使能channelSplit功能，nSize必须为8的倍数，取值范围：nSize∈[1, 4095]。</p>
<p id="p1350283818336"><a name="p1350283818336"></a><a name="p1350283818336"></a>若不使能channelSplit功能，nSize必须为16的倍数，取值范围：nSize∈[1, 4095]。</p>
</li></ul>
<a name="ul1263216231934"></a><a name="ul1263216231934"></a><ul id="ul1263216231934"><li>使能NZ2ND功能<p id="p1117443113208"><a name="p1117443113208"></a><a name="p1117443113208"></a>nSize取值范围 ∈[1, 4095]。</p>
</li></ul>
</td>
</tr>
<tr id="row167989075313"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p1279815010538"><a name="p1279815010538"></a><a name="p1279815010538"></a>mSize</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1379818010537"><a name="p1379818010537"></a><a name="p1379818010537"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p182986458"><a name="p182986458"></a><a name="p182986458"></a>源NZ矩阵在M方向上的大小。</p>
<a name="ul497913120510"></a><a name="ul497913120510"></a><ul id="ul497913120510"><li>不使能NZ2ND功能<p id="p1420619273418"><a name="p1420619273418"></a><a name="p1420619273418"></a>取值范围：mSize∈[1, 65535]。</p>
</li></ul>
<a name="ul428310181147"></a><a name="ul428310181147"></a><ul id="ul428310181147"><li>使能NZ2ND功能<p id="p97981300531"><a name="p97981300531"></a><a name="p97981300531"></a>取值范围：mSize∈[1, 8192]。</p>
</li></ul>
</td>
</tr>
<tr id="row5798501532"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p107981201535"><a name="p107981201535"></a><a name="p107981201535"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p187991001538"><a name="p187991001538"></a><a name="p187991001538"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p16221439111013"><a name="p16221439111013"></a><a name="p16221439111013"></a>源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：srcStride∈[0, 65535]， 单位：C0_Size(16*sizeof(T)，T为src的数据类型)。</p>
</td>
</tr>
<tr id="row779911045318"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p97991104537"><a name="p97991104537"></a><a name="p97991104537"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p107991005319"><a name="p107991005319"></a><a name="p107991005319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><a name="ul4462818706"></a><a name="ul4462818706"></a><ul id="ul4462818706"><li>不使能NZ2ND功能<p id="p187991003533"><a name="p187991003533"></a><a name="p187991003533"></a>目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0， 单位：datablock(32Bytes)。</p>
</li></ul>
<a name="ul1078723912015"></a><a name="ul1078723912015"></a><ul id="ul1078723912015"><li>使能NZ2ND功能<p id="p1706171910"><a name="p1706171910"></a><a name="p1706171910"></a>目的ND矩阵每一行中的元素个数，取值不为0 ，单位：element。</p>
</li></ul>
</td>
</tr>
<tr id="row20800305532"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p2527010114710"><a name="p2527010114710"></a><a name="p2527010114710"></a>quantPre</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p12800104535"><a name="p12800104535"></a><a name="p12800104535"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p7427841113712"><a name="p7427841113712"></a><a name="p7427841113712"></a>QuantMode_t是一个枚举类型，用于控制量化模式，默认值为QuantMode_t::NoQuant，即不使能量化功能。QuantMode_t取值如下：</p>
<a name="ul17664135412379"></a><a name="ul17664135412379"></a><ul id="ul17664135412379"><li>NoQuant，不使能量化功能</li><li>F322F16，float量化成half，量化结果支持INF_NAN模式</li><li>F322BF16，float量化成bfloat16_t，量化结果支持INF_NAN模式</li><li>DEQF16，int32_t量化成half, scalar量化，量化结果不支持INF_NAN模式</li><li>VDEQF16， int32_t量化成half，tensor量化，量化结果不支持INF_NAN模式</li><li>QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化</li><li>VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化</li><li>REQ8，int32_t量化成uint8_t/int8_t，scalar量化</li><li>VREQ8，int32_t量化成uint8_t/int8_t，tensor量化</li></ul>
</td>
</tr>
<tr id="row129021142124519"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p1902942144514"><a name="p1902942144514"></a><a name="p1902942144514"></a>deqScalar</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p16902124254518"><a name="p16902124254518"></a><a name="p16902124254518"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p1491353317116"><a name="p1491353317116"></a><a name="p1491353317116"></a>scalar量化参数，表示单个scale值，quantPre量化模式为scalar量化时需要设置该参数。支持的数据类型为uint64_t。</p>
</td>
</tr>
<tr id="row174857514214"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p34854544214"><a name="p34854544214"></a><a name="p34854544214"></a>ndNum</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p154854584210"><a name="p154854584210"></a><a name="p154854584210"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p651038432"><a name="p651038432"></a><a name="p651038432"></a>源NZ矩阵的数目，也就是传输ND矩阵的数目，取值范围：ndNum∈[1, 65535]</p>
</td>
</tr>
<tr id="row112913160433"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p51291916164317"><a name="p51291916164317"></a><a name="p51291916164317"></a>srcNdStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p141296168430"><a name="p141296168430"></a><a name="p141296168430"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p11129616194312"><a name="p11129616194312"></a><a name="p11129616194312"></a>不同NZ矩阵起始地址之间的间隔，<span>取值范围：srcNdStride∈</span>[1, 512]，单位：数据块（16 * C0_Size）。当ndNum配置为1时，srcNdStride配置为0即可，不生效。</p>
</td>
</tr>
<tr id="row2525193614320"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p115251236174312"><a name="p115251236174312"></a><a name="p115251236174312"></a>dstNdStride</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p11525836164317"><a name="p11525836164317"></a><a name="p11525836164317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p163611952104320"><a name="p163611952104320"></a><a name="p163611952104320"></a>目的相邻ND矩阵起始地址之间的偏移，取值范围：dstNdstride∈[1, 65535]，单位：element。当ndNum配置为1时，dstNdStride配置为0即可，不生效。</p>
</td>
</tr>
<tr id="row18762171304418"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p176210135442"><a name="p176210135442"></a><a name="p176210135442"></a>reluEn</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p376221314410"><a name="p376221314410"></a><a name="p376221314410"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p7762171320445"><a name="p7762171320445"></a><a name="p7762171320445"></a>是否使能relu的开关，false：不使能relu功能；true：使能relu功能。</p>
</td>
</tr>
<tr id="row4306125615443"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p23074567441"><a name="p23074567441"></a><a name="p23074567441"></a>unitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p1930735614442"><a name="p1930735614442"></a><a name="p1930735614442"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p9307115654415"><a name="p9307115654415"></a><a name="p9307115654415"></a>unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下：</p>
<p id="p1225131744220"><a name="p1225131744220"></a><a name="p1225131744220"></a>0：保留值；</p>
<p id="p3836113514213"><a name="p3836113514213"></a><a name="p3836113514213"></a>2：使能unitFlag，硬件执行完指令之后，不会设置寄存器；</p>
<p id="p53431612144318"><a name="p53431612144318"></a><a name="p53431612144318"></a>3：使能unitFlag，硬件执行完指令之后，会将unitFlag关闭。</p>
<p id="p14988589213"><a name="p14988589213"></a><a name="p14988589213"></a>使能该功能时，Fixpipe指令的unitFlag设置为3即可。</p>
</td>
</tr>
<tr id="row189241520171518"><td class="cellrowborder" valign="top" width="13.459999999999999%" headers="mcps1.2.4.1.1 "><p id="p1092412021519"><a name="p1092412021519"></a><a name="p1092412021519"></a>isChannelSplit</p>
</td>
<td class="cellrowborder" valign="top" width="10.05%" headers="mcps1.2.4.1.2 "><p id="p4924920191517"><a name="p4924920191517"></a><a name="p4924920191517"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.49000000000001%" headers="mcps1.2.4.1.3 "><p id="p15924020141512"><a name="p15924020141512"></a><a name="p15924020141512"></a>是否使能通道拆分的功能。默认为false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能ChannelSplit和NZ2ND功能。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  FixpipeParamsC310结构体参数说明

<a name="table134331412112119"></a>
<table><thead align="left"><tr id="row174331512172117"><th class="cellrowborder" valign="top" width="14.19%" id="mcps1.2.4.1.1"><p id="p17433191210216"><a name="p17433191210216"></a><a name="p17433191210216"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="9.16%" id="mcps1.2.4.1.2"><p id="p843315126213"><a name="p843315126213"></a><a name="p843315126213"></a>数据类型</p>
</th>
<th class="cellrowborder" valign="top" width="76.64999999999999%" id="mcps1.2.4.1.3"><p id="p1443301202112"><a name="p1443301202112"></a><a name="p1443301202112"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row44331012102118"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p124332120217"><a name="p124332120217"></a><a name="p124332120217"></a>nSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p1433131212219"><a name="p1433131212219"></a><a name="p1433131212219"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p18598141656"><a name="p18598141656"></a><a name="p18598141656"></a>源NZ矩阵在N方向上的大小。</p>
<a name="ul743441210218"></a><a name="ul743441210218"></a><ul id="ul743441210218"><li>不使能NZ2ND功能<p id="p3895115235716"><a name="p3895115235716"></a><a name="p3895115235716"></a>若使能channelSplit功能，nSize必须为8的倍数，取值范围：nSize∈[1, 4095]。</p>
<p id="p187966552579"><a name="p187966552579"></a><a name="p187966552579"></a>若不使能channelSplit功能，nSize必须为16的倍数，取值范围：nSize∈[1, 4095]。</p>
</li></ul>
<a name="ul18434111217212"></a><a name="ul18434111217212"></a><ul id="ul18434111217212"><li>使能NZ2ND功能<p id="p194346120215"><a name="p194346120215"></a><a name="p194346120215"></a>nSize取值范围 ∈[1, 4095]。</p>
<a name="ul1942794917104"></a><a name="ul1942794917104"></a><ul id="ul1942794917104"><li>通路CO1-&gt;UB/L1，nSize*sizeof(T)必须为32的倍数。</li></ul>
</li></ul>
</td>
</tr>
<tr id="row154341212102118"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p94347129214"><a name="p94347129214"></a><a name="p94347129214"></a>mSize</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p443471215211"><a name="p443471215211"></a><a name="p443471215211"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p15724163212516"><a name="p15724163212516"></a><a name="p15724163212516"></a>源NZ矩阵在M方向上的大小。</p>
<p id="p2434512112119"><a name="p2434512112119"></a><a name="p2434512112119"></a>取值范围：mSize∈[1, 65535]。</p>
<a name="ul1825551617180"></a><a name="ul1825551617180"></a><ul id="ul1825551617180"><li>使能NZ2DN功能，通路CO1-&gt;UB/L1，mSize*sizeof(T)必须为32的倍数</li></ul>
</td>
</tr>
<tr id="row743491272111"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p1434171214215"><a name="p1434171214215"></a><a name="p1434171214215"></a>srcStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p2434141216218"><a name="p2434141216218"></a><a name="p2434141216218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p643413129218"><a name="p643413129218"></a><a name="p643413129218"></a>源NZ矩阵中相邻Z排布的起始地址偏移，取值范围：srcStride∈[0, 65535]， 单位：C0_Size(16*sizeof(T), T为src的数据类型)。</p>
</td>
</tr>
<tr id="row16434912112110"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p1743431262112"><a name="p1743431262112"></a><a name="p1743431262112"></a>dstStride</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p043411129218"><a name="p043411129218"></a><a name="p043411129218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><a name="ul12435212192115"></a><a name="ul12435212192115"></a><ul id="ul12435212192115"><li>不使能NZ2ND功能<p id="p1943571282110"><a name="p1943571282110"></a><a name="p1943571282110"></a>目的NZ矩阵中相邻Z排布的起始地址偏移，取值不为0， 单位：element。</p>
</li></ul>
<a name="ul943541262117"></a><a name="ul943541262117"></a><ul id="ul943541262117"><li>使能NZ2ND/NZ2DN功能<p id="p10435191216219"><a name="p10435191216219"></a><a name="p10435191216219"></a>目的ND矩阵每一行中的元素个数，取值不为0 ，单位：element。</p>
</li></ul>
</td>
</tr>
<tr id="row443551202120"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p15435171212215"><a name="p15435171212215"></a><a name="p15435171212215"></a>quantPre</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p4435101210213"><a name="p4435101210213"></a><a name="p4435101210213"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p174358123212"><a name="p174358123212"></a><a name="p174358123212"></a>QuantMode_t是一个枚举类型，用于控制量化模式，默认值为QuantMode_t::NoQuant，即不使能量化功能。QuantMode_t取值如下：</p>
<a name="ul978141520339"></a><a name="ul978141520339"></a><ul id="ul978141520339"><li>NoQuant，不使能量化功能</li><li>F322F16，float量化成half，量化结果支持INF_NAN模式</li><li>F322BF16，float量化成bfloat16_t，量化结果支持INF_NAN模式</li><li>DEQF16，int32_t量化成half, scalar量化，量化结果不支持INF_NAN模式</li><li>VDEQF16，int32_t量化成half，tensor量化，量化结果不支持INF_NAN模式</li><li>QF322B8_PRE，float量化成uint8_t/int8_t，scalar量化</li><li>VQF322B8_PRE，float量化成uint8_t/int8_t，tensor量化</li><li>REQ8，int32_t量化成uint8_t/int8_t，scalar量化</li><li>VREQ8，int32_t量化成uint8_t/int8_t，tensor量化</li><li>QF322FP8_PRE，float量化成fp8_e4m3fn_t，scalar量化</li><li>VQF322FP8_PRE，float量化成fp8_e4m3fn_t，tensor量化</li><li>QF322HIF8_PRE，float量化成hifloat8_t(Half to Away Round)，scalar量化</li><li>VQF322HIF8_PRE，float量化成hifloat8_t(Half to Away Round)，tensor量化</li><li>QF322HIF8_PRE_HYBRID，float量化成hifloat8_t(Hybrid Round)，scalar量化</li><li>VQF322HIF8_PRE_HYBRID，float量化成hifloat8_t(Hybrid Round)，tensor量化</li><li>QS322BF16_PRE，int32_t量化成bfloat16_t，scalar量化</li><li>VQS322BF16_PRE，int32_t量化成bfloat16_t，tensor量化</li><li>QF322F16_PRE，float量化成half，scalar量化</li><li>VQF322F16_PRE，float量化成half，tensor量化</li><li>QF322BF16_PRE，float量化成bfloat16_t，scalar量化</li><li>VQF322BF16_PRE，float量化成bfloat16_t，tensor量化</li><li>QF322F32_PRE，float量化成float，scalar量化，该量化模式精度无法达到双万分之一，可以达到双千分之一。如果有双万分之一的精度要求，建议使用<a href="AscendDequant.md">AscendDeQuant</a>高阶API。</li><li>VQF322F32_PRE，float量化成float，tensor量化，该量化模式精度无法达到双万分之一，可以达到双千分之一。如果有双万分之一的精度要求，建议使用<a href="AscendDequant.md">AscendDeQuant</a>高阶API。</li></ul>
</td>
</tr>
<tr id="row0497111218213"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p849721252112"><a name="p849721252112"></a><a name="p849721252112"></a>deqScalar</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p849718124216"><a name="p849718124216"></a><a name="p849718124216"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p114976129219"><a name="p114976129219"></a><a name="p114976129219"></a>scalar量化参数，表示单个scale值，quantPre量化模式为scalar量化时需要设置该参数。支持的数据类型为uint64_t。</p>
</td>
</tr>
<tr id="row1157218571820"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p284010161911"><a name="p284010161911"></a><a name="p284010161911"></a>reluEn</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p208401416391"><a name="p208401416391"></a><a name="p208401416391"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p3840121610920"><a name="p3840121610920"></a><a name="p3840121610920"></a>是否使能relu的开关，false：不使能relu功能；true：使能relu功能。</p>
</td>
</tr>
<tr id="row167211423478"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p1553065184715"><a name="p1553065184715"></a><a name="p1553065184715"></a>unitFlag</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p45307515476"><a name="p45307515476"></a><a name="p45307515476"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p8530857474"><a name="p8530857474"></a><a name="p8530857474"></a>unitFlag是一种Mmad指令和Fixpipe指令细粒度的并行，使能该功能后，硬件每计算完一个分形，计算结果就会被搬出，该功能不适用于在L0C Buffer累加的场景。取值说明如下：</p>
<p id="p132227570444"><a name="p132227570444"></a><a name="p132227570444"></a>0：保留值；</p>
<p id="p2222135704413"><a name="p2222135704413"></a><a name="p2222135704413"></a>2：使能unitFlag，硬件执行完指令之后，不会设置寄存器；</p>
<p id="p4222145784415"><a name="p4222145784415"></a><a name="p4222145784415"></a>3：使能unitFlag，硬件执行完指令之后，会将unitFlag关闭。</p>
<p id="p4222657114418"><a name="p4222657114418"></a><a name="p4222657114418"></a>使能该功能时，Fixpipe指令的unitFlag设置为3即可。</p>
</td>
</tr>
<tr id="row6563440185617"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p356384012560"><a name="p356384012560"></a><a name="p356384012560"></a>TransformParams</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p1156354035611"><a name="p1156354035611"></a><a name="p1156354035611"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p87341352154"><a name="p87341352154"></a><a name="p87341352154"></a>TransformParams 结构体是一个基于模板参数的类型选择器，用于在编译时根据不同的CO2Layout布局类型，自动选择对应的参数类型。</p>
<a name="screen17278377518"></a><a name="screen17278377518"></a><pre class="screen" codetype="Cpp" id="screen17278377518">template &lt;CO2Layout format&gt;
struct TransformParams {};
template &lt;&gt;
struct TransformParams&lt;CO2Layout::NZ&gt; {
    __aicore__ inline TransformParams(){};
    using PARAMS = uint8_t;
};
template &lt;&gt;
struct TransformParams&lt;CO2Layout::ROW_MAJOR&gt; {
    __aicore__ inline TransformParams(){};
    using PARAMS = Nz2NdParams;
};
template &lt;&gt;
struct TransformParams&lt;CO2Layout::COLUMN_MAJOR&gt; {
    __aicore__ inline TransformParams(){};
    using PARAMS = Nz2DnParams;
};</pre>
<p id="p10826134221615"><a name="p10826134221615"></a><a name="p10826134221615"></a>CO2Layout 布局类型如下：</p>
<a name="ul16939237203213"></a><a name="ul16939237203213"></a><ul id="ul16939237203213"><li>ROW_MAJOR<p id="p6946145093113"><a name="p6946145093113"></a><a name="p6946145093113"></a>当启用该模式时， 该指令被定义为从L0C到目标位置的数据移动，并附带NZ2ND转换。</p>
<a name="screen6946165012319"></a><a name="screen6946165012319"></a><pre class="screen" codetype="Cpp" id="screen6946165012319">struct Nz2NdParams {
    uint16_t ndNum = 1;
    uint16_t srcNdStride = 0;
    uint32_t dstNdStride = 0; 
};</pre>
<p id="p1094615508312"><a name="p1094615508312"></a><a name="p1094615508312"></a>ndNum: 源NZ矩阵的数目，也就是传输ND矩阵的数目，取值范围：ndNum∈[1, 65535]。</p>
<p id="p1494618506318"><a name="p1494618506318"></a><a name="p1494618506318"></a>srcNdStride: 不同NZ矩阵起始地址之间的间隔，取值范围：srcNdStride∈[0, 65535]，单位: C0_SIZE。当ndNum配置为1时，srcNdStride配置为0即可，不生效。</p>
<p id="p794619502312"><a name="p794619502312"></a><a name="p794619502312"></a>dstNdStride: 目的相邻ND矩阵起始地址之间的偏移，取值范围：dstNdstride∈[1, 2^32 -1]，单位：element。当ndNum配置为1时，dstNdStride配置为0即可，不生效。</p>
</li><li>COLUMN_MAJOR<p id="p083421611326"><a name="p083421611326"></a><a name="p083421611326"></a>当启用该模式时， 该指令被定义为从L0C到目标位置的数据移动，并附带NZ2DN转换。</p>
<a name="screen1883410164328"></a><a name="screen1883410164328"></a><pre class="screen" codetype="Cpp" id="screen1883410164328">struct Nz2DnParams {
    uint16_t dnNum = 1;
    uint16_t srcNzMatrixStride = 0;
    uint32_t dstDnMatrixStride = 0;
    uint16_t srcNzC0Stride = 0; 
};</pre>
<p id="p168341216193217"><a name="p168341216193217"></a><a name="p168341216193217"></a>dnNum: 传输DN矩阵的数目，取值范围：dnNum∈[1, 65535]。</p>
<p id="p10834141618328"><a name="p10834141618328"></a><a name="p10834141618328"></a>srcNzMatrixStride: 不同源NZ矩阵的偏移（头与头），单位: C0_SIZE。</p>
<p id="p2834101611324"><a name="p2834101611324"></a><a name="p2834101611324"></a>dstDnMatrixStride: 目的相邻DN矩阵起始地址间的偏移，取值范围：dstDnMatrixdstride∈[1, 2^32 -1]，单位：element。</p>
<p id="p1834171610321"><a name="p1834171610321"></a><a name="p1834171610321"></a>srcNzC0Stride: 源矩阵NZ分形中相邻行的地址偏移（头与头）,  单位：C0_SIZE。</p>
</li></ul>
<a name="ul1930331712359"></a><a name="ul1930331712359"></a><ul id="ul1930331712359"><li>NZ<p id="p83941448125315"><a name="p83941448125315"></a><a name="p83941448125315"></a>当启用该模式时，为普通搬运DMA模式，表示从L0C到目标位置的正常数据移动。</p>
</li></ul>
</td>
</tr>
<tr id="row164971312182115"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p1151315501314"><a name="p1151315501314"></a><a name="p1151315501314"></a>dualDstCtrl</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p9512957138"><a name="p9512957138"></a><a name="p9512957138"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p14367102421518"><a name="p14367102421518"></a><a name="p14367102421518"></a>双目标模式控制。当启用双目标模式控制时，L0C中的M×N矩阵将被分成两半，并同时写入两个子块（SUB BLOCK）的UB中，其中前半部分写入SUB BLOCK0，后半部分写入SUB BLOCK1。</p>
<p id="p16367824191512"><a name="p16367824191512"></a><a name="p16367824191512"></a>2'b00：单目标模式，将整个矩阵写入通过subBlockId参数配置的目标UB。</p>
<p id="p11367162441517"><a name="p11367162441517"></a><a name="p11367162441517"></a>2'b01：双目标模式，按M维度拆分，M / 2 * N写入每个UB, M必须为2的倍数。</p>
<p id="p1471018164172"><a name="p1471018164172"></a><a name="p1471018164172"></a>2'b10：双目标模式，按N维度拆分，M * N / 2写入每个UB, N须为2的倍数。</p>
<p id="p18367142418157"><a name="p18367142418157"></a><a name="p18367142418157"></a>2'b11：保留。</p>
<p id="p9129195017257"><a name="p9129195017257"></a><a name="p9129195017257"></a>dualDstCtrl仅支持在普通搬运模式或NZ2ND搬运场景下使用，不支持随路功能场景。</p>
</td>
</tr>
<tr id="row1549713127216"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p6511175161315"><a name="p6511175161315"></a><a name="p6511175161315"></a>subBlockId</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p551015181316"><a name="p551015181316"></a><a name="p551015181316"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p351014518132"><a name="p351014518132"></a><a name="p351014518132"></a>在启用单目标模式时指示目标UB的编号。</p>
</td>
</tr>
<tr id="row1958498181413"><td class="cellrowborder" valign="top" width="14.19%" headers="mcps1.2.4.1.1 "><p id="p7989991144"><a name="p7989991144"></a><a name="p7989991144"></a>isChannelSplit</p>
</td>
<td class="cellrowborder" valign="top" width="9.16%" headers="mcps1.2.4.1.2 "><p id="p149894921418"><a name="p149894921418"></a><a name="p149894921418"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="76.64999999999999%" headers="mcps1.2.4.1.3 "><p id="p129892914147"><a name="p129892914147"></a><a name="p129892914147"></a>是否使能通道拆分的功能。默认为false，不使能该功能。仅在src和dst都为float时才能使能通道拆分，且不能同时使能ChannelSplit和NZ2ND功能。</p>
</td>
</tr>
</tbody>
</table>

不使能NZ2ND的情况下，参数设置示例（通过Fixpipe接口搬运并去除dummy数据）和解释说明如下：

当M方向上的数据元素个数不是16的倍数时，搬入时会额外读取dummy数据，并在写入目标位置后丢弃这些dummy数据。矩阵块被定义为连续的16\*16的数据块，数据块的个数为M/16向上取整， 矩阵块的长度为M\*16\*sizeof\(T\)，T是数据类型。

单搬运模式：

-   nSize = 48，表示源NZ矩阵中待搬运矩阵（图中蓝色区域）在N方向上的大小为48个元素。
-   mSize = 24，表示源NZ矩阵中待搬运矩阵在M方向上的大小为24个元素。
-   srcStride = 64，表示源NZ矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 40，表示目的NZ矩阵中相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为40 \* 32B。

**图 1**  不使能NZ2ND参数的单搬运模式设置示意图、<a name="fig128961542184620"></a>  
<!-- img2text -->
```text
srcLocal
      ←────────────── nSize = 48 ──────────────→
      ┌────────┬────────┬────────┐
      │ 16 ×16 │ 16 ×16 │ 16 ×16 │
      │   ↘    │   ↘    │   ↘    │
↑     ├────────┼────────┼────────┤
│     │        │        │        │
│     │ 16 ×16 │ 16 ×16 │ 16 ×16 │
│     │        │        │        │
│     ├────────┼────────┼────────┤
│     │        │        │        │
│     │        │        │        │
│     │        │        │        │
│     ├────────┼────────┼────────┤
│     │ 16 ×16 │ 16 ×16 │ 16 ×16 │
│     │   ↙    │   ↙    │   ↙    │
↓     └────────┴────────┴────────┘
srcStride = 64
      ↕
    mSize = 24

      ←── C0 ──→

                ↘
                  ↘  dummy data
                    ↘
                      ↘

                         不使能NZ2ND
                              ──→

                                      dstLocal
                                       ←─ C0 ─→
                                      ┌────────┐
                           ↑          │        │
                           │          │        │
                           │          │        │
                           │          ├────────┤
                           │          │        │
                           │          │        │
                           │          ├────────┤
                           │          │        │
                           │          │        │
                           │          ├────────┤
                           │          │        │
                           │          │        │
                           ↓          └────────┘
                      dstStride = 40

                                         ↑
                                         │ mSize = 24
                                         ↓
                                          ||
```

说明:
- 左侧 `srcLocal` 为 3 列排列，顶部标注 `nSize = 48`，每列宽度为 `16`，对应 3 个 `16 × 16`。
- 左侧垂直方向标注 `srcStride = 64`，其中有效搬运高度标注为 `mSize = 24`。
- 底部第一列宽度标注为 `C0`。
- 中间 `dummy data` 表示未使能 `NZ2ND` 时产生的填充/无效数据。
- 右侧 `dstLocal` 为单列纵向排布，顶部宽度标注 `C0`，垂直方向标注 `dstStride = 40`。
- 右侧有效区域高度标注为 `mSize = 24`，旁边 `||` 表示该高度范围。
- 图中斜箭头表示从左侧多个 `16 × 16` 区块搬运映射到右侧纵向 `dstLocal` 中；由于为多对一斜向映射，未逐条精确复现。

双目标控制模式：

在普通搬运模式模式下启用双目标模式如下图所示，分为按M维度拆分和按N维度拆分，按M维度拆分M必须为2的倍数，按N维度拆分N必须为2的倍数：

N方向切分：

-   nSize = 32，表示源NZ矩阵中待搬运矩阵在N方向上的大小为32个元素。
-   mSize = 48，表示源NZ矩阵中待搬运矩阵在M方向上的大小48个元素。
-   srcStride = 64，表示源NZ矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个块Z排布矩阵的起始地址与第二个Z排布矩阵的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 64，表示目的NZ矩阵中相邻Z排布的起始地址偏移，即下图中UB0中Z排布的起始地址与UB1中Z排布的起始地址之间的间隔为64 \* C0\_Size。

M方向切分：

-   nSize = 32，表示源NZ矩阵中待搬运矩阵在N方向上的大小为32个元素。
-   mSize = 24，表示源NZ矩阵中待搬运矩阵在M方向上的大小为48个元素。
-   srcStride = 64，表示源NZ矩阵中待搬运矩阵相邻Z排布的起始地址偏移，即下图中第一个块Z排布矩阵的起始地址与第二个Z排布矩阵的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 40，表示目的NZ矩阵中相邻Z排布的起始地址偏移，即下图中UB0（或UB1）中第一个Z排布的起始地址与第二个Z排布的起始地址之间的间隔为40 \*  C0\_Size。

**图 2**  不使能NZ2ND参数和NZ2DN参数的双目标搬运模式设置示意图<a name="fig6561154491913"></a>  
<!-- img2text -->
```
SRC                              DUAL DST split N :                     DUAL DST split M :
srcLocal                         dstLocal      c0        c0             dstLocal      c0        c0
                                   ┌───────┐  ┌───────┐                  ┌───────┐  ┌───────┐
        nSize = 32                 │ 16×16 │  │ 16×16 │                  │       │  │       │
    <────────────────>             │       │  │       │                  │ 16×16 │  │       │
    ┌──────────────────────┐       ├───────┤  ├───────┤                  │       │  │       │
    │      │       │       │       │ 16×16 │  │ 16×16 │                  ├───────┤  ├───────┤
    │ 16×16│ 16×16 │       │       │       │  │       │                  │       │  │       │
    │      │       │       │       ├───────┤  ├───────┤                  │       │  │       │
    ├──────┼───────┤       │       │ 16×16 │  │ 16×16 │                  │       │  │       │
    │ 16×16│ 16×16 │       │       │       │  │       │                  ├───────┤  ├───────┤
    │      │       │       │       ├───────┤  ├───────┤                  │ 16×16 │  │       │
    ├──────┼───────┤       │       │ 16×16 │  │ 16×16 │                  │       │  │       │
    │ 16×16│ 16×16 │       │       │       │  │       │                  ├───────┤  ├───────┤
    │      │       │       │       ├───────┤  ├───────┤                  │       │  │ 16×16 │
    ├──────┼───────┤       │       │ 16×16 │  │ 16×16 │                  │       │  │       │
    │ 16×16│ 16×16 │       │       │       │  │       │                  │       │  │       │
    │      │       │       │       └───────┘  └───────┘                  ├───────┤  ├───────┤
    └──────────────────────┘          UB0        UB1                     │       │  │       │
      ↑                 ↑               ↑          ↑                      │       │  │       │
      └────── C0 ───────┘               └── C0 ───┘                      │       │  │       │
                                                                           └───────┘  └───────┘
                                                                             UB0        UB1

↑ srcStride = 64                  ↑ dstStride = 64                        ↑ dstStride = 40
↑ mSize = 48                      ↑ mSize = 48                            ↑ size = 24


SRC 内部搬运方向/对应:
  左列:  ↘ 从上到下
  右列:  ↙ 从上到下

DUAL DST split N 内部搬运方向/对应:
  UB0: ↙ 从顶部到低部，底部有水平 →
  UB1: ↙ 从顶部到低部，底部有水平 →

DUAL DST split M 内部搬运方向/对应:
  UB0: 从上部 16×16 区域 ↘ 到下部中间区域，再水平 →
  UB1: 从上部区域 ↙ 到下部中间区域，再水平 →
```

使能NZ2ND的情况下，参数设置示例和解释说明如下：

-   ndNum = 2，表示源NZ矩阵的数目为2。图中蓝色区域为NZ矩阵1，紫色区域为NZ矩阵2。

-   nSize = 32，表示源NZ矩阵（图中蓝色区域）在N方向上的大小为32个元素。
-   mSize = 48，表示源NZ矩阵在M方向上的大小为48个元素。
-   srcStride = 64，表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 64，表示目的ND矩阵每一行中的元素个数为64。
-   srcNdStride = 16,  表示不同NZ矩阵起始地址之间的间隔为16 \*  16 \* C0\_Size。
-   dstNdStride  = 4096，表示目的相邻ND矩阵起始地址之间的偏移为4096个元素。

**图 3**  使能NZ2ND参数设置示意图<a name="fig79783143556"></a>  
<!-- img2text -->
```
srcLocal                                               使能NZ2ND                                              dstLocal
                                                        ─────────→

nSize = 32                                                                                                  nSize = 32
<──────────────>                                                                                           <──────────────>

srcStride = 64
↑
│   mSize = 48
│   <──────>
│   ┌──────┬──────┬──────┬──────┬──────┬──────┐
│   │      │      │      │      │      │      │
│   │      │      │      │      │      │      │
│   ├──────┼──────┼──────┼──────┼──────┼──────┤
│   │      │      │      │      │      │      │
│   │  ↗   │  ↗   │      │      │      │      │
│   ├──────┼──────┼──────┼──────┼──────┼──────┤
│   │      │      │      │      │      │      │
│   │  ↙   │  ↙   │      │      │      │      │
│   ├──────┼──────┼──────┼──────┼──────┼──────┤
│   │      │      │      │      │      │      │
│   └──────┴──────┴──────┴──────┴──────┴──────┘
│   <──>
│   C0
│
└──────────────────────────────────────────────
              srcNdStride = 16


                                                dstNdStride = 64 * dstStride
                                                ↑
                                                │
                                                │      ┌──────────────┐
                                                │      │  第一个ND矩阵 │
                                                │      │              │
                                                │      │              │
                                                │      └──────────────┘
                                                │
                                                │                              mSize = 48
                                                │                              ↑
                                                │                              │
                                                │                              ↓
                                                │
                                                │      ┌──────────────┐
                                                │      │  第二个ND矩阵 │
                                                │      │              │
                                                │      │              │
                                                │      └──────────────┘
                                                │
                                                └──────────────────────────────→
                                                   <──────────────────────>
                                                         dstStride = 64
```

说明:
- 图中左侧为 `srcLocal` 的 NZ 数据布局，右侧为 `dstLocal` 的 ND 数据布局。
- 左侧标注包含：`nSize = 32`、`mSize = 48`、`srcStride = 64`、`C0`、`srcNdStride = 16`。
- 中间操作标注为：`使能NZ2ND`。
- 右侧标注包含：`nSize = 32`、`mSize = 48`、`dstStride = 64`、`dstNdStride = 64 * dstStride`。
- 右侧两个矩阵文字分别为：`第一个ND矩阵`、`第二个ND矩阵`。
- 图中左侧前两个列块表示一个 NZ 矩阵区域，右侧后两个列块表示另一个 NZ 矩阵区域；右图上下两个矩阵分别对应两个 ND 矩阵。

单搬入模式

-   ndNum = 2，表示源NZ矩阵的数目为2。图中蓝色区域为NZ矩阵1，紫色区域为NZ矩阵2。

-   nSize = 32，表示源NZ矩阵（图中蓝色区域）在N方向上的大小为32个元素。
-   mSize = 48，表示源NZ矩阵在M方向上的大小为48个元素。
-   srcStride = 64，表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图中第一个蓝色Z排布的起始地址与第二个蓝色Z排布的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 64，表示目的ND矩阵每一行中的元素个数为64。
-   srcNdStride = 256,  表示不同NZ矩阵起始地址之间的间隔为256，单位为C0\_Size。
-   dstNdStride  = 4096，表示目的相邻ND矩阵起始地址之间的偏移为4096个元素。

**图 4**  使能NZ2ND参数的单搬入模式设置示意图<a name="fig11991024112516"></a>  
<!-- img2text -->
```text
srcLocal                                           使能NZ2ND                                           dstLocal
                                                    ─────→

nSize = 32                                                                                         nSize = 32
<──────────────>                                                                                   <──────────────>

┌──────┬──────┬──────┬──────┬──────┬──────┐                                                    ┌────────────────────┐
│      │      │      │      │      │      │                                                    │    第一个ND矩阵     │
├──────┼──────┼──────┼──────┼──────┼──────┤                                                    │                    │
│      │      │      │      │      │      │                                                    │                    │
├──────┼──────┼──────┼──────┼──────┼──────┤                                                    │                    │
│      │      │      │      │      │      │                                                    └────────────────────┘
├──────┼──────┼──────┼──────┼──────┼──────┤
│      │      │      │      │      │      │
└──────┴──────┴──────┴──────┴──────┴──────┘
  <──────────────>
      C0

↑
│ srcStride = 64
│ nSize = 48
↓

<────────────────────────────>
      srcNdStride = 256


                                                                                                  ↑
                                                                                                  │ mSize = 48
                                                                                                  ↓

                                                                                                  ↑
                                                                                                  │ dstNdStride = 64 * dstStride
                                                                                                  │
                                                                                                  │
                                                                                                  │
                                                                                                  │
                                                                                                  │
                                                                                                  │
                                                                                                  │
                                                                                                  ↓
                                                                                            ┌────────────────────┐
                                                                                            │    第二个ND矩阵     │
                                                                                            │                    │
                                                                                            │                    │
                                                                                            └────────────────────┘

                                                                                            <────────────────────>
                                                                                                  dstStride = 64
```

双搬入模式

-   ndNum = 2，表示源NZ矩阵的数目为2。图中红框区域为矩阵1，蓝框区域为矩阵2。

-   nSize = 32，表示源NZ矩阵（图中红框区域或蓝框区域）在N方向上的大小为32个元素。
-   mSize = 48，表示源NZ矩阵在M方向上的大小为48个元素。
-   srcStride = 64，表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图红框区域中左侧浅色Z排布矩阵的起始地址与右侧深色Z排布矩阵的起始地址之间的间隔为64 \* C0\_Size。
-   dstStride = 64，表示目的ND矩阵每一行中的元素个数为64。
-   ndNum = 2, 表示源NZ矩阵的数目。
-   srcNdStride = 240 ,  表示不同NZ矩阵起始地址之间的间隔为240 \* C0\_Size。
-   dstNdStride  = 4096，表示目的相邻ND矩阵起始地址之间的偏移为4096个元素。

    **图 5**  使能NZ2ND参数双搬入模式设置示意图<a name="fig8810182815117"></a>  
    <!-- img2text -->
```
┌──────────────────────────────────────── SRC ────────────────────────────────────────┐
│ srcLocal                                                                             │
│                                                                                      │
│         nSize = 32                                                                   │
│      ◄────────────────►                                                              │
│      ┌──────────┬──────────┬──────────┬──────────┐                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      └──────────┴──────────┴──────────┴──────────┘                                   │
│                                                                                      │
│      ↑                                                                               │
│      │ srcStride = 64                                                                │
│      ↓                                                                               │
│                                                                                      │
│        ↑                                                                             │
│        │ mSize = 48                                                                  │
│        ↓                                                                             │
│                                                                                      │
│      ◄──────────────────────────────►                                                │
│           srcNdStride = 240                                                          │
│                                                                                      │
│      选中区域(左上 2×3 块):                                                          │
│      ┌──────────┬──────────┐                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      ├──────────┼──────────┤                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      ├──────────┼──────────┤                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      └──────────┴──────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘


DUAL DST split N :

┌────────────────────────────────────────── UB0 ───────────────────────────────────────┐
│ nSize = 16                                                                           │
│ ◄────────►                                                                           │
│                                                                                      │
│ ↑                                                                                    │
│ │ mSize = 64*dstStride                                                               │
│ ↓                                                                                    │
│                                                                                      │
│ ┌──────────┐                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ └──────────┘                                                                         │
│                                                                                      │
│                                                                                      │
│ ┌──────────┐                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ └──────────┘                                                                         │
│                                                                                      │
│ ─────────────────────────────→                                                       │
│ ╲                                                                                   │
│  ╲                                                                                  │
│   ╲────────────────────────→                                                         │
│                                                                                      │
│ ◄────────────────────────────►                                                       │
│         dstStride = 64                                                               │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────── UB1 ───────────────────────────────────────┐
│ nSize = 16                                                                           │
│ ◄────────►                                                                           │
│                                                                                      │
│ ↑                                                                                    │
│ │ mSize = 48                                                                         │
│ ↓                                                                                    │
│                                                                                      │
│ ┌──────────┐                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ └──────────┘                                                                         │
│                                                                                      │
│                                                                                      │
│ ┌──────────┐                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ ├──────────┤                                                                         │
│ │          │                                                                         │
│ └──────────┘                                                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘


┌──────────────────────────────────────── SRC ────────────────────────────────────────┐
│ srcLocal                                                                             │
│                                                                                      │
│         nSize = 32                                                                   │
│      ◄────────────────►                                                              │
│      ┌──────────┬──────────┬──────────┬──────────┐                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      ├──────────┼──────────┼──────────┼──────────┤                                   │
│      │ 16 × 16  │ 16 × 16  │          │          │                                   │
│      └──────────┴──────────┴──────────┴──────────┘                                   │
│                                                                                      │
│      ↑                                                                               │
│      │ srcStride = 64                                                                │
│      ↓                                                                               │
│                                                                                      │
│        ↑                                                                             │
│        │ mSize = 48                                                                  │
│        ↓                                                                             │
│                                                                                      │
│      ◄──────────────────────────────►                                                │
│           srcNdStride = 240                                                          │
│                                                                                      │
│      选中区域(左上 2×3 块):                                                          │
│      ┌──────────┬──────────┐                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      ├──────────┼──────────┤                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      ├──────────┼──────────┤                                                         │
│      │ 16 × 16  │ 16 × 16  │                                                         │
│      └──────────┴──────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────────────────┘


DUAL DST split M :

┌────────────────────────────────────────── UB0 ───────────────────────────────────────┐
│ nSize = 16                                                                           │
│ ◄────────►                                                                           │
│                                                                                      │
│ ↑                                                                                    │
│ │ mSize = 64*dstStride                                                               │
│ ↓                                                                                    │
│                                                                                      │
│ ┌──────────┬──────────┐                                                              │
│ │          │          │                                                              │
│ ├──────────┼──────────┤                                                              │
│ │          │          │                                                              │
│ └──────────┴──────────┘                                                              │
│                                                                                      │
│                                                                                      │
│ ┌──────────┬──────────┐                                                              │
│ │          │          │                                                              │
│ ├──────────┼──────────┤                                                              │
│ │          │          │                                                              │
│ └──────────┴──────────┘                                                              │
│                                                                                      │
│ ─────────────────────────────→                                                       │
│ ╲                                                                                   │
│  ╲                                                                                  │
│   ╲────────────────────────→                                                         │
│                                                                                      │
│ ◄────────────────────────────►                                                       │
│         dstStride = 64                                                               │
└──────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────── UB1 ───────────────────────────────────────┐
│ nSize = 16                                                                           │
│ ◄────────►                                                                           │
│                                                                                      │
│ ↑                                                                                    │
│ │ mSize = 24                                                                         │
│ ↓                                                                                    │
│                                                                                      │
│ ┌──────────┬──────────┐                                                              │
│ │          │          │                                                              │
│ ├──────────┼──────────┤                                                              │
│ │          │          │                                                              │
│ └──────────┴──────────┘                                                              │
│                                                                                      │
│                                                                                      │
│ ┌──────────┬──────────┐                                                              │
│ │          │          │                                                              │
│ ├──────────┼──────────┤                                                              │
│ │          │          │                                                              │
│ └──────────┴──────────┘                                                              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

说明:
- 图中共有两组示意：`DUAL DST split N` 与 `DUAL DST split M`。
- 左侧 `SRC` 两图相同：`srcLocal` 中选取一个 `nSize = 32`、`mSize = 48`、`srcStride = 64` 的区域，单元块大小均标为 `16 × 16`，底部标注 `srcNdStride = 240`。
- `DUAL DST split N`：目的端按 `N` 方向拆分，`UB0` 与 `UB1` 的宽度均为 `nSize = 16`；`UB1` 的高度标为 `mSize = 48`；`UB0` 底部标 `dstStride = 64`，左侧标 `mSize = 64*dstStride`。
- `DUAL DST split M`：目的端按 `M` 方向拆分，`UB0` 与 `UB1` 的宽度均为 `nSize = 16`；`UB1` 左侧标 `mSize = 24`；`UB0` 底部标 `dstStride = 64`，左侧标 `mSize = 64*dstStride`。
- 图中彩色块表示源矩阵不同子块被拆分后分别搬运到 `UB0`、`UB1`；斜箭头表示从源选中区域到目的缓冲区的搬运对应关系。

使能NZ2DN的情况下，参数设置示例和解释说明如下：

单搬运模式：

-   dnNum = 2，表示源NZ矩阵的数目为2。图中蓝色区域为NZ矩阵1，紫色区域为NZ矩阵2。

-   nSize = 32，表示源NZ矩阵（图中蓝色区域）在N方向上的大小为32个元素。
-   mSize = 48，表示源NZ矩阵在M方向上的大小为48个元素。
-   srcStride = 80，表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图中相邻两块蓝色Z排布的起始地址之间的间隔为80 \* C0\_Size。
-   dstStride = 80，表示目的DN矩阵每一行中的元素个数为80。
-   ndNum = 2, 表示源NZ矩阵的数目。
-   srcNzMatrixStride = 240，表示不同源NZ矩阵的偏移，即下图中第一个蓝色Z排布的起始地址和第二个紫色Z排布的起始地址之间的间隔为240 \* C0\_Size。
-   srcNzC0Stride = 1：表示源矩阵NZ分形相邻行的地址偏移，。
-   dstDnMatrixStide：表示相邻DN矩阵起始地址间的偏移为48 \* 80 =3840个元素。
-   。

**图 6**  使能NZ2DN单搬运模式示意图1<a name="fig19772110476"></a>  
<!-- img2text -->
```text
srcLocal                                              →→→                                              dstLocal
      <────────────── nSize = 32 ──────────────>                                                   <────────────── mSize = 48 ──────────────>

      srcNzC0Stride = 1
             ↓
┌──────┬──────┬──────┬──────┬──────┬──────┐                                               ┌────────────────────────────────────────────┐
│      │      │      │      │      │      │                                               │               第一个DN矩阵                 │
├──────┼──────┼──────┼──────┼──────┼──────┤                                               │                                            │
│      │      │      │      │      │      │                                               │                                            │
├──────┼──────┼──────┼──────┼──────┼──────┤                                               │                                            │
│      │      │      │      │      │      │                                               │                                            │
├──────┼──────┼──────┼──────┼──────┼──────┤                                               └────────────────────────────────────────────┘
│      │      │      │      │      │      │
├──────┼──────┼──────┼──────┼──────┼──────┤
│      │      │      │      │      │      │                                               ┌────────────────────────────────────────────┐
└──────┴──────┴──────┴──────┴──────┴──────┘                                               │               第二个DN矩阵                 │
  ↑                                                                                       │                                            │
  │ mSize = 48                                                                             │                                            │
  │                                                                                       │                                            │
  │                                                                                       └────────────────────────────────────────────┘
  └──────────── srcStride = 80

  C0

<──────────────────────────────────────────────>
         srcNzMatrixStride = 80 * 3 = 240

                                                                                           ↑
                                                                                           │ nSize = 32
                                                                                           │
                                                                                           │
                                                                                           └──────────── dstStride = 80

                                                                                           ↑
                                                                                           │ dstDnMatrixStride = 48 * dstStride
                                                                                           │
                                                                                           │
                                                                                           └──────────── = 48 * 80

                                          使能NZ2DN

srcLocal 中的两个 NZ 矩阵：
┌──────────────┬──────────────┐
│  NZ矩阵1     │   NZ矩阵2    │
└──────────────┴──────────────┘

对应关系：
- NZ矩阵1 → 第一个DN矩阵
- NZ矩阵2 → 第二个DN矩阵
```

说明:
- 图中左侧为 `srcLocal`，右侧为 `dstLocal`。
- `dnNum = 2`，表示源 NZ 矩阵数目为 2。
- 原图用颜色区分两个 NZ 矩阵及其对应的两个 DN 矩阵：上方一组对应第一个DN矩阵，下方一组对应第二个DN矩阵。
- 左图参数包含：`nSize = 32`、`mSize = 48`、`srcStride = 80`、`srcNzC0Stride = 1`、`srcNzMatrixStride = 80 * 3 = 240`、`C0`。
- 右图参数包含：`mSize = 48`、`nSize = 32`、`dstStride = 80`、`dstDnMatrixStride = 48 * dstStride = 48 * 80`。
- 原图中矩阵内部及跨矩阵存在斜向搬运箭头，表示从 NZ 布局到 DN 布局的地址映射与搬运方向；因斜向多区域映射较复杂，未在 ASCII 中逐线复现。

单搬运模式：

-   dnNum = 2，表示源NZ矩阵的数目为2。图中蓝色区域为NZ矩阵1，红色区域为NZ矩阵2。

-   nSize = 24，表示源NZ矩阵（图中蓝色区域）在N方向上的大小为24个元素。
-   mSize = 24，表示源NZ矩阵在M方向上的大小为24个元素。
-   srcStride = 80，表示源NZ矩阵中相邻Z排布的起始地址偏移，即下图中相邻两块蓝色Z排布的起始地址之间的间隔为80 \* C0\_Size。
-   dstStride = 60，表示目的DN矩阵每一行中的元素个数为80。
-   ndNum = 2, 表示源NZ矩阵的数目。
-   srcNzMatrixStride = 240，表示不同源NZ矩阵的偏移，即下图中第一个蓝色Z排布的起始地址和第二个紫色Z排布的起始地址之间的间隔为240 \* C0\_Size。
-   srcNzC0Stride = 2：表示源矩阵NZ分形相邻行的地址偏移。
-   dstDnMatrixStide：表示相邻DN矩阵起始地址间的偏移为48 \* 60 = 2880个元素。

**图 7**  使能NZ2DN单搬运模式示意图2<a name="fig769114585716"></a>  
<!-- img2text -->
```
srcLocal
                     <──────── mSize = 24 ────────>
                ┌───────────────────────────────────────┐
                │               │               │       │
                │  第一个NZ矩阵  │               │       │
                │  (Z排布)       │               │       │
                │  ┌─────────┐   │               │       │
srcNzC0Stride=2 │  │         │   │               │       │
        ↑       │  │         │   │               │       │
        │       │  │         │   │               │       │
        │       │  │         │   │               │       │
        │       │  │         │   │               │       │
        │       │  │         │   │               │       │
        ↓       │  └─────────┘   │               │       │
                │                │               │       │
                │<─ srcStride = 24 ─>            │       │
                │                │               │       │
                ├────────────────┼───────────────┼───────┤
                │                │               │       │
                │                │  第二个NZ矩阵 │       │
                │                │  (Z排布)      │       │
                │                │  ┌─────────┐  │       │
                │                │  │         │  │       │
                │                │  │         │  │       │
                │                │  │         │  │       │
                │                │  │         │  │       │
                │                │  │         │  │       │
                │                │  └─────────┘  │       │
                │                │               │       │
                └────────────────┴───────────────┴───────┘
                ↑
                │
                │ srcNzMatrixStride = 80 * 3 = 240
                │
                └─ srcStride = 80

                                ───────▶

                           使能NZ2DN

dstLocal
                     <──────── mSize = 24 ────────>
                ┌────────────────────────────────────────────┐
                │  第一个DN矩阵                               │
                │  ┌────────────────────────┐                │
                │  │                        │────────────────────────→
                │  │                        │                │
                │  └────────────────────────┘                │
                │                                            │
                │<── dstStride = 48 dstSize = 24 ──>         │
                ├────────────────────────────────────────────┤
                │                                            │
                │  第二个DN矩阵                               │
                │  ┌────────────────────────┐                │
                │  │                        │                │
                │  └────────────────────────┘                │
                │                                            │
                │                                            │
                │                                            │
                │                                            │
                │                                            │
                │                                            │
                └────────────────────────────────────────────┘
                ↑                                            ↑
                │                                            │
                │                                            └── dstStride = 60
                │
                └── dstDnMatrixStride = 48 * 60 = 2880
```

说明:
- 左侧为 srcLocal 中两个 NZ 矩阵，右侧为 dstLocal 中两个 DN 矩阵。
- 图中保留的文字标注包括：srcLocal、dstLocal、mSize = 24、srcNzC0Stride = 2、srcStride = 24、srcStride = 80、srcNzMatrixStride = 80 * 3 = 240、使能NZ2DN、dstStride = 48 dstSize = 24、dstDnMatrixStride = 48 * 60 = 2880、dstStride = 60、第一个DN矩阵、第二个DN矩阵。
- 左图中两个 NZ 矩阵分别对应右图中的第一个DN矩阵和第二个DN矩阵。右图中的斜线与顶部长横线表示搬运后的地址映射关系。

## 约束说明<a name="section205941531307"></a>

-   ndNum=0 表示不执行，此指令将不被执行并报warning。
-   对于量化输入为float32数据类型的说明如下：
    -   标准的IEEE 754 float32格式为：1bit符号位，8bits指数位，23bits尾数位；当前AI处理器支持的float32格式为：1bit符号位，8bits指数位，10bits尾数位。
    -   如果用户提供的是标准的IEEE 754 float32输入，API内部会处理成处理器支持的float32格式进行计算，此时如果golden数据生成过程中使用的是标准的IEEE 754 float32数据，则可能引入精度不匹配问题，需要修正golden数据的生成，将量化参数的23bits尾数位的低13bits数据位清零再参与量化计算。

## 调用示例<a name="section93513321362"></a>

-   示例一：通路CO1-\>GM，不使能tensor量化功能接口。输入A矩阵和B矩阵的数据类型为half，输出C矩阵为half，默认配置使能Nz2Nd的格式转换，使能F322F16量化将mmad计算出的结果由float量化成half。

    ```
    AscendC::GlobalTensor<half> cGM;
    AscendC::LocalTensor<half> c1Local = outQueueCO1.DeQue<half>();
    uint16_t m = 32, n = 16, k = 32; // m, n为输出的行和列
    uint16_t B32_B16_SIZE = 16 * 16;
    uint8_t n_block = 16;
    AscendC::FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = m;
    fixpipeParams.dstStride = n;
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 2;
    fixpipeParams.dstNdStride = m * n;
    fixpipeParams.quantPre = QuantMode_t::F322F16; // 使能F322F16量化将mmad计算出的结果由float量化成half
    AscendC::Fixpipe(cGM, c1Local, fixpipeParams);
    outQueueCO1.FreeTensor(c1Local);
    
    ```

    示例结果：

    ```
    输入数据A矩阵:
     [[6. 3. 9. 4. 5. 3. 9. 7. 3. 6. 2. 7. 3. 8. 8. 1. 8. 8. 5. 6. 6. 8. 2. 2.
      3. 6. 4. 8. 9. 6. 6. 1.]
     [2. 5. 7. 2. 4. 2. 5. 2. 4. 6. 4. 8. 5. 7. 1. 4. 3. 1. 8. 6. 4. 6. 9. 1.
      8. 2. 9. 5. 3. 7. 7. 8.]
     [5. 8. 2. 1. 4. 5. 7. 7. 4. 6. 8. 5. 6. 5. 4. 2. 5. 4. 7. 9. 5. 4. 7. 4.
      2. 2. 1. 7. 8. 4. 6. 6.]
     [8. 2. 4. 7. 6. 9. 7. 7. 4. 5. 6. 7. 6. 6. 5. 3. 7. 6. 7. 4. 5. 4. 1. 9.
      6. 7. 8. 9. 4. 9. 5. 5.]
     [4. 9. 4. 2. 7. 8. 3. 4. 1. 5. 3. 8. 8. 5. 5. 8. 3. 8. 5. 3. 9. 4. 5. 4.
      2. 4. 3. 8. 9. 8. 4. 3.]
     [1. 3. 8. 3. 1. 9. 9. 5. 5. 6. 3. 2. 3. 4. 3. 3. 5. 9. 6. 7. 1. 3. 4. 2.
      8. 5. 9. 1. 9. 5. 8. 9.]
     [3. 3. 1. 3. 5. 2. 7. 8. 8. 9. 6. 9. 3. 6. 5. 5. 2. 3. 2. 3. 5. 1. 6. 1.
      7. 8. 7. 2. 2. 7. 8. 1.]
     [4. 4. 6. 4. 6. 5. 1. 2. 7. 8. 3. 2. 9. 9. 7. 7. 7. 1. 2. 7. 2. 1. 5. 2.
      1. 3. 2. 1. 3. 3. 2. 9.]
     [4. 6. 3. 5. 8. 4. 1. 1. 2. 5. 8. 8. 8. 3. 9. 6. 5. 6. 7. 9. 2. 1. 9. 3.
      2. 5. 4. 1. 7. 5. 3. 9.]
     [7. 2. 3. 4. 9. 5. 6. 3. 4. 5. 4. 7. 4. 1. 9. 4. 2. 1. 7. 4. 9. 2. 4. 5.
      4. 5. 8. 7. 2. 2. 8. 3.]
     [5. 7. 6. 2. 9. 4. 7. 1. 8. 6. 2. 1. 6. 5. 5. 6. 3. 8. 1. 5. 2. 1. 8. 3.
      1. 9. 3. 3. 5. 2. 2. 5.]
     [4. 7. 5. 9. 9. 6. 7. 3. 1. 9. 2. 6. 5. 2. 6. 7. 1. 7. 6. 9. 3. 7. 6. 1.
      3. 9. 2. 4. 1. 9. 4. 8.]
     [2. 4. 3. 1. 1. 2. 2. 7. 2. 3. 7. 9. 8. 8. 3. 4. 1. 2. 9. 2. 9. 4. 4. 8.
      5. 7. 7. 3. 9. 9. 5. 3.]
     [3. 1. 1. 6. 1. 8. 3. 3. 6. 3. 4. 4. 3. 8. 2. 1. 1. 1. 6. 5. 8. 8. 5. 8.
      5. 1. 2. 2. 1. 3. 7. 4.]
     [4. 2. 8. 4. 4. 1. 9. 6. 9. 9. 5. 4. 3. 1. 3. 8. 1. 2. 8. 2. 5. 8. 9. 3.
      2. 5. 9. 7. 7. 4. 2. 1.]
     [2. 6. 7. 1. 3. 9. 9. 9. 6. 4. 5. 8. 1. 3. 7. 3. 8. 7. 3. 4. 8. 6. 9. 6.
      8. 9. 4. 4. 7. 6. 1. 4.]
     [2. 8. 2. 1. 2. 6. 2. 8. 5. 9. 9. 8. 6. 4. 4. 1. 4. 1. 4. 4. 4. 7. 5. 9.
      9. 8. 9. 1. 8. 4. 7. 3.]
     [3. 6. 2. 5. 1. 2. 9. 2. 6. 7. 4. 5. 9. 6. 5. 9. 7. 9. 5. 5. 6. 7. 4. 7.
      7. 6. 3. 6. 5. 2. 8. 3.]
     [1. 7. 3. 2. 4. 8. 1. 7. 3. 4. 1. 6. 1. 4. 4. 1. 6. 7. 9. 3. 9. 2. 2. 2.
      2. 8. 1. 1. 6. 3. 6. 1.]
     [4. 3. 9. 5. 2. 2. 1. 8. 5. 8. 9. 2. 4. 3. 2. 1. 8. 6. 6. 2. 9. 2. 9. 3.
      9. 5. 3. 7. 9. 7. 6. 2.]
     [9. 4. 8. 1. 3. 7. 9. 5. 2. 4. 9. 9. 6. 9. 6. 4. 6. 3. 3. 9. 6. 8. 1. 5.
      5. 1. 6. 5. 1. 9. 3. 9.]
     [2. 5. 2. 1. 8. 9. 9. 8. 1. 6. 1. 1. 9. 8. 3. 5. 6. 4. 2. 1. 3. 7. 8. 9.
      6. 6. 1. 9. 1. 7. 6. 8.]
     [4. 7. 6. 6. 2. 2. 1. 8. 7. 1. 1. 2. 1. 1. 9. 8. 9. 4. 9. 5. 7. 8. 9. 9.
      5. 1. 6. 8. 9. 6. 7. 5.]
     [1. 1. 6. 9. 9. 3. 7. 6. 5. 6. 5. 1. 5. 5. 3. 7. 6. 7. 4. 8. 8. 2. 2. 5.
      7. 8. 8. 2. 9. 1. 5. 1.]
     [5. 4. 6. 8. 8. 3. 7. 7. 5. 7. 8. 7. 4. 8. 2. 9. 4. 8. 1. 3. 8. 5. 3. 7.
      3. 7. 1. 9. 1. 5. 4. 7.]
     [6. 3. 1. 2. 8. 3. 2. 6. 8. 2. 8. 4. 1. 9. 4. 7. 5. 1. 7. 5. 5. 1. 1. 1.
      2. 8. 1. 7. 9. 8. 5. 4.]
     [2. 8. 5. 1. 3. 4. 9. 8. 6. 9. 6. 2. 4. 2. 2. 7. 8. 2. 1. 3. 7. 1. 4. 6.
      4. 6. 3. 3. 1. 6. 8. 3.]
     [5. 1. 5. 5. 9. 7. 9. 2. 1. 4. 7. 8. 1. 9. 8. 1. 2. 4. 3. 9. 9. 6. 7. 9.
      1. 5. 1. 9. 2. 5. 6. 9.]
     [1. 9. 9. 6. 5. 7. 9. 5. 4. 1. 2. 8. 3. 8. 1. 9. 6. 1. 7. 9. 3. 2. 2. 4.
      7. 9. 9. 4. 7. 1. 5. 8.]
     [3. 2. 2. 5. 9. 3. 6. 9. 2. 4. 4. 8. 4. 2. 6. 1. 2. 8. 8. 8. 9. 7. 7. 1.
      9. 6. 5. 8. 3. 3. 3. 4.]
     [9. 1. 6. 1. 3. 7. 8. 1. 2. 6. 5. 9. 4. 4. 7. 2. 3. 9. 8. 7. 8. 2. 6. 4.
      5. 6. 5. 4. 9. 6. 1. 9.]
     [4. 3. 2. 7. 8. 1. 7. 2. 9. 7. 7. 4. 2. 8. 2. 5. 6. 9. 5. 1. 3. 9. 8. 2.
      4. 8. 4. 7. 4. 1. 3. 7.]]
    输入数据B矩阵: 
    [[3. 5. 9. 6. 2. 9. 3. 6. 5. 9. 5. 5. 3. 8. 5. 2.]
     [5. 1. 5. 7. 5. 4. 2. 2. 4. 8. 1. 1. 3. 3. 7. 2.]
     [6. 7. 4. 6. 1. 4. 8. 3. 9. 2. 2. 3. 4. 6. 5. 3.]
     [4. 8. 2. 6. 4. 8. 6. 7. 3. 8. 6. 7. 3. 8. 1. 1.]
     [6. 7. 8. 6. 1. 9. 9. 3. 9. 9. 2. 1. 3. 3. 3. 3.]
     [7. 2. 4. 7. 5. 8. 9. 2. 1. 7. 9. 6. 8. 7. 1. 3.]
     [3. 3. 9. 2. 3. 9. 4. 1. 8. 2. 5. 1. 2. 6. 5. 5.]
     [6. 4. 8. 8. 7. 5. 9. 6. 7. 6. 8. 8. 2. 6. 1. 2.]
     [4. 2. 3. 8. 6. 1. 1. 1. 7. 9. 5. 2. 2. 5. 7. 6.]
     [4. 5. 9. 5. 6. 8. 1. 2. 1. 9. 2. 7. 8. 6. 6. 1.]
     [4. 8. 6. 6. 3. 1. 7. 8. 7. 3. 2. 9. 8. 6. 9. 8.]
     [3. 2. 5. 5. 7. 9. 7. 7. 4. 8. 3. 5. 2. 7. 1. 2.]
     [3. 8. 2. 8. 9. 5. 1. 5. 7. 4. 1. 3. 4. 1. 4. 6.]
     [9. 5. 2. 2. 4. 6. 3. 3. 7. 1. 9. 6. 8. 6. 4. 7.]
     [2. 3. 8. 1. 5. 9. 8. 4. 5. 4. 6. 5. 4. 5. 3. 2.]
     [3. 5. 4. 2. 1. 2. 9. 2. 3. 8. 9. 8. 8. 1. 2. 7.]
     [1. 4. 5. 1. 3. 8. 2. 5. 9. 9. 5. 5. 5. 6. 4. 2.]
     [7. 6. 7. 7. 6. 9. 1. 3. 8. 1. 9. 8. 8. 5. 1. 6.]
     [5. 3. 8. 9. 8. 2. 6. 6. 1. 3. 2. 1. 2. 9. 3. 9.]
     [1. 1. 4. 9. 8. 6. 6. 5. 6. 8. 4. 2. 2. 7. 2. 1.]
     [8. 1. 3. 5. 8. 7. 5. 7. 4. 6. 7. 4. 8. 2. 2. 3.]
     [5. 8. 6. 8. 1. 8. 6. 8. 3. 9. 1. 1. 3. 8. 3. 2.]
     [7. 7. 5. 1. 5. 4. 6. 1. 1. 6. 8. 8. 1. 7. 7. 2.]
     [1. 7. 7. 7. 7. 6. 1. 7. 3. 3. 8. 9. 3. 8. 9. 8.]
     [4. 9. 5. 6. 9. 6. 8. 9. 1. 1. 6. 5. 1. 4. 3. 5.]
     [4. 1. 8. 9. 6. 5. 5. 7. 8. 9. 8. 2. 7. 5. 5. 3.]
     [9. 8. 4. 9. 5. 4. 7. 5. 7. 6. 9. 8. 5. 7. 2. 9.]
     [6. 6. 5. 1. 4. 5. 9. 6. 7. 5. 5. 2. 3. 7. 6. 5.]
     [5. 2. 5. 7. 9. 2. 2. 3. 2. 3. 1. 4. 6. 5. 3. 1.]
     [5. 1. 9. 3. 2. 4. 1. 6. 7. 7. 4. 9. 8. 8. 6. 1.]
     [3. 7. 5. 6. 7. 8. 2. 2. 8. 7. 6. 1. 3. 5. 3. 2.]
     [7. 6. 7. 8. 6. 5. 2. 2. 8. 2. 2. 6. 6. 4. 9. 6.]]
    输出数据C矩阵: 
    [[ 807.  767. 1007.  925.  853. 1079.  837.  782.  977.  960.  838.  746.
       767. 1013.  642.  594.]
     [ 778.  775.  850.  874.  801.  853.  767.  682.  808.  852.  719.  709.
       651.  891.  663.  635.]
     [ 734.  705.  927.  901.  865.  906.  742.  687.  840.  892.  725.  718.
       692.  911.  702.  601.]
     [ 877.  895. 1099. 1070.  954. 1136.  926.  912. 1028. 1057.  983.  930.
       859. 1119.  760.  768.]
     [ 818.  722.  931.  904.  857.  969.  809.  724.  846.  948.  812.  786.
       811.  885.  644.  619.]
     [ 780.  750.  907.  964.  865.  905.  738.  638.  861.  808.  816.  759.
       735.  913.  627.  640.]
     [ 697.  671.  865.  810.  780.  863.  729.  656.  803.  892.  798.  734.
       664.  819.  593.  561.]
     [ 619.  633.  716.  734.  667.  767.  612.  515.  749.  794.  641.  652.
       650.  705.  596.  518.]
     [ 716.  738.  908.  907.  838.  902.  767.  684.  829.  907.  726.  787.
       728.  872.  671.  609.]
     [ 692.  710.  876.  838.  779.  926.  812.  692.  791.  894.  767.  660.
       629.  844.  588.  597.]
     [ 671.  639.  812.  787.  684.  815.  637.  511.  806.  819.  714.  627.
       652.  734.  628.  546.]
     [ 779.  764. 1011.  962.  806. 1042.  845.  728.  883. 1027.  794.  762.
       764.  949.  667.  576.]
     [ 750.  690.  856.  907.  875.  801.  716.  772.  771.  803.  760.  772.
       724.  865.  633.  656.]
     [ 598.  605.  649.  731.  678.  741.  591.  593.  577.  694.  662.  591.
       536.  750.  508.  508.]
     [ 754.  750.  902.  869.  746.  815.  807.  669.  780.  912.  750.  719.
       658.  905.  658.  633.]
     [ 844.  758. 1037.  971.  920. 1038.  903.  800.  920.  983.  937.  863.
       791. 1011.  726.  648.]
     [ 754.  782.  935. 1018.  936.  909.  770.  795.  799.  947.  796.  811.
       726.  937.  708.  644.]
     [ 744.  828.  940.  936.  914. 1014.  753.  760.  893.  946.  874.  777.
       768.  920.  699.  706.]
     [ 615.  467.  719.  754.  714.  750.  601.  560.  637.  739.  650.  544.
       598.  699.  434.  437.]
     [ 785.  791.  906.  889.  868.  866.  766.  768.  836.  871.  787.  814.
       738.  920.  693.  592.]
     [ 814.  822. 1006.  963.  831. 1062.  868.  826.  991.  950.  834.  853.
       809. 1021.  745.  700.]
     [ 782.  812.  957.  847.  800.  998.  773.  688.  882.  890.  854.  770.
       730.  889.  721.  642.]
     [ 792.  815.  966.  947.  895.  942.  858.  786.  859.  995.  884.  827.
       701. 1006.  711.  657.]
     [ 758.  791.  878.  960.  861.  938.  818.  735.  889.  906.  861.  763.
       751.  869.  588.  649.]
     [ 830.  853.  990.  936.  817. 1044.  862.  796.  990.  994.  902.  865.
       834.  953.  744.  698.]
     [ 679.  586.  833.  792.  716.  754.  713.  653.  816.  856.  708.  654.
       698.  802.  608.  566.]
     [ 636.  642.  844.  775.  723.  821.  652.  600.  809.  864.  743.  693.
       671.  763.  652.  546.]
     [ 804.  789.  987.  887.  824. 1084.  868.  766.  933.  924.  859.  786.
       762. 1002.  735.  639.]
     [ 813.  765.  906. 1016.  889.  947.  902.  735.  933.  949.  870.  738.
       737.  943.  664.  708.]
     [ 790.  769.  946.  935.  877.  996.  899.  798.  840.  903.  807.  718.
       651.  919.  579.  605.]
     [ 803.  725. 1003.  949.  900. 1002.  792.  749.  860.  863.  818.  812.
       790.  972.  686.  657.]
     [ 787.  813.  910.  873.  751.  927.  751.  688.  874.  914.  795.  733.
       721.  903.  697.  664.]]
    ```

-   示例二：通路CO1-\>GM，使能tensor量化功能接口。输入A矩阵和B矩阵的数据类型为int8，输出C矩阵为half，默认配置使能Nz2Nd的格式转换，使能tensor量化（VDEQF16）将mmad计算出的结果由int32 量化成half。

    ```
    AscendC::GlobalTensor<half> cGM;
    AscendC::LocalTensor<dstCO1_T> c1Local = outQueueCO1.DeQue<dstCO1_T>();
    AscendC::LocalTensor<uint64_t> deqTensorLocal = deqQueue.DeQue<uint64_t>();
    uint16_t m = 32, n = 16, k = 32;
    uint16_t B32_B16_SIZE = 16 * 16;
    uint8_t n_block = 16;
    AscendC::FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    fixpipeParams.srcStride = m;
    fixpipeParams.dstStride = n;
    fixpipeParams.ndNum = 1;
    fixpipeParams.srcNdStride = 4;
    fixpipeParams.dstNdStride = m * n;
    fixpipeParams.quantPre = QuantMode_t::VDEQF16;
    AscendC::Fixpipe(cGM, c1Local, deqTensorLocal, fixpipeParams); // CO1到GM可以进行NZ到ND的转换
    outQueueCO1.FreeTensor(c1Local);
    deqQueue.FreeTensor(deqTensorLocal);
    
    ```

    示例结果：

    ```
    输入数据A矩阵:  
    [[6 3 9 4 5 3 9 7 3 6 2 7 3 8 8 1 8 8 5 6 6 8 2 2 3 6 4 8 9 6 6 1]
     [2 5 7 2 4 2 5 2 4 6 4 8 5 7 1 4 3 1 8 6 4 6 9 1 8 2 9 5 3 7 7 8]
     [5 8 2 1 4 5 7 7 4 6 8 5 6 5 4 2 5 4 7 9 5 4 7 4 2 2 1 7 8 4 6 6]
     [8 2 4 7 6 9 7 7 4 5 6 7 6 6 5 3 7 6 7 4 5 4 1 9 6 7 8 9 4 9 5 5]
     [4 9 4 2 7 8 3 4 1 5 3 8 8 5 5 8 3 8 5 3 9 4 5 4 2 4 3 8 9 8 4 3]
     [1 3 8 3 1 9 9 5 5 6 3 2 3 4 3 3 5 9 6 7 1 3 4 2 8 5 9 1 9 5 8 9]
     [3 3 1 3 5 2 7 8 8 9 6 9 3 6 5 5 2 3 2 3 5 1 6 1 7 8 7 2 2 7 8 1]
     [4 4 6 4 6 5 1 2 7 8 3 2 9 9 7 7 7 1 2 7 2 1 5 2 1 3 2 1 3 3 2 9]
     [4 6 3 5 8 4 1 1 2 5 8 8 8 3 9 6 5 6 7 9 2 1 9 3 2 5 4 1 7 5 3 9]
     [7 2 3 4 9 5 6 3 4 5 4 7 4 1 9 4 2 1 7 4 9 2 4 5 4 5 8 7 2 2 8 3]
     [5 7 6 2 9 4 7 1 8 6 2 1 6 5 5 6 3 8 1 5 2 1 8 3 1 9 3 3 5 2 2 5]
     [4 7 5 9 9 6 7 3 1 9 2 6 5 2 6 7 1 7 6 9 3 7 6 1 3 9 2 4 1 9 4 8]
     [2 4 3 1 1 2 2 7 2 3 7 9 8 8 3 4 1 2 9 2 9 4 4 8 5 7 7 3 9 9 5 3]
     [3 1 1 6 1 8 3 3 6 3 4 4 3 8 2 1 1 1 6 5 8 8 5 8 5 1 2 2 1 3 7 4]
     [4 2 8 4 4 1 9 6 9 9 5 4 3 1 3 8 1 2 8 2 5 8 9 3 2 5 9 7 7 4 2 1]
     [2 6 7 1 3 9 9 9 6 4 5 8 1 3 7 3 8 7 3 4 8 6 9 6 8 9 4 4 7 6 1 4]
     [2 8 2 1 2 6 2 8 5 9 9 8 6 4 4 1 4 1 4 4 4 7 5 9 9 8 9 1 8 4 7 3]
     [3 6 2 5 1 2 9 2 6 7 4 5 9 6 5 9 7 9 5 5 6 7 4 7 7 6 3 6 5 2 8 3]
     [1 7 3 2 4 8 1 7 3 4 1 6 1 4 4 1 6 7 9 3 9 2 2 2 2 8 1 1 6 3 6 1]
     [4 3 9 5 2 2 1 8 5 8 9 2 4 3 2 1 8 6 6 2 9 2 9 3 9 5 3 7 9 7 6 2]
     [9 4 8 1 3 7 9 5 2 4 9 9 6 9 6 4 6 3 3 9 6 8 1 5 5 1 6 5 1 9 3 9]
     [2 5 2 1 8 9 9 8 1 6 1 1 9 8 3 5 6 4 2 1 3 7 8 9 6 6 1 9 1 7 6 8]
     [4 7 6 6 2 2 1 8 7 1 1 2 1 1 9 8 9 4 9 5 7 8 9 9 5 1 6 8 9 6 7 5]
     [1 1 6 9 9 3 7 6 5 6 5 1 5 5 3 7 6 7 4 8 8 2 2 5 7 8 8 2 9 1 5 1]
     [5 4 6 8 8 3 7 7 5 7 8 7 4 8 2 9 4 8 1 3 8 5 3 7 3 7 1 9 1 5 4 7]
     [6 3 1 2 8 3 2 6 8 2 8 4 1 9 4 7 5 1 7 5 5 1 1 1 2 8 1 7 9 8 5 4]
     [2 8 5 1 3 4 9 8 6 9 6 2 4 2 2 7 8 2 1 3 7 1 4 6 4 6 3 3 1 6 8 3]
     [5 1 5 5 9 7 9 2 1 4 7 8 1 9 8 1 2 4 3 9 9 6 7 9 1 5 1 9 2 5 6 9]
     [1 9 9 6 5 7 9 5 4 1 2 8 3 8 1 9 6 1 7 9 3 2 2 4 7 9 9 4 7 1 5 8]
     [3 2 2 5 9 3 6 9 2 4 4 8 4 2 6 1 2 8 8 8 9 7 7 1 9 6 5 8 3 3 3 4]
     [9 1 6 1 3 7 8 1 2 6 5 9 4 4 7 2 3 9 8 7 8 2 6 4 5 6 5 4 9 6 1 9]
     [4 3 2 7 8 1 7 2 9 7 7 4 2 8 2 5 6 9 5 1 3 9 8 2 4 8 4 7 4 1 3 7]]
    输入数据B矩阵:  
    [[3 5 9 6 2 9 3 6 5 9 5 5 3 8 5 2 5 1 5 7 5 4 2 2 4 8 1 1 3 3 7 2]
     [6 7 4 6 1 4 8 3 9 2 2 3 4 6 5 3 4 8 2 6 4 8 6 7 3 8 6 7 3 8 1 1]
     [6 7 8 6 1 9 9 3 9 9 2 1 3 3 3 3 7 2 4 7 5 8 9 2 1 7 9 6 8 7 1 3]
     [3 3 9 2 3 9 4 1 8 2 5 1 2 6 5 5 6 4 8 8 7 5 9 6 7 6 8 8 2 6 1 2]
     [4 2 3 8 6 1 1 1 7 9 5 2 2 5 7 6 4 5 9 5 6 8 1 2 1 9 2 7 8 6 6 1]
     [4 8 6 6 3 1 7 8 7 3 2 9 8 6 9 8 3 2 5 5 7 9 7 7 4 8 3 5 2 7 1 2]
     [3 8 2 8 9 5 1 5 7 4 1 3 4 1 4 6 9 5 2 2 4 6 3 3 7 1 9 6 8 6 4 7]
     [2 3 8 1 5 9 8 4 5 4 6 5 4 5 3 2 3 5 4 2 1 2 9 2 3 8 9 8 8 1 2 7]
     [1 4 5 1 3 8 2 5 9 9 5 5 5 6 4 2 7 6 7 7 6 9 1 3 8 1 9 8 8 5 1 6]
     [5 3 8 9 8 2 6 6 1 3 2 1 2 9 3 9 1 1 4 9 8 6 6 5 6 8 4 2 2 7 2 1]
     [8 1 3 5 8 7 5 7 4 6 7 4 8 2 2 3 5 8 6 8 1 8 6 8 3 9 1 1 3 8 3 2]
     [7 7 5 1 5 4 6 1 1 6 8 8 1 7 7 2 1 7 7 7 7 6 1 7 3 3 8 9 3 8 9 8]
     [4 9 5 6 9 6 8 9 1 1 6 5 1 4 3 5 4 1 8 9 6 5 5 7 8 9 8 2 7 5 5 3]
     [9 8 4 9 5 4 7 5 7 6 9 8 5 7 2 9 6 6 5 1 4 5 9 6 7 5 5 2 3 7 6 5]
     [5 2 5 7 9 2 2 3 2 3 1 4 6 5 3 1 5 1 9 3 2 4 1 6 7 7 4 9 8 8 6 1]
     [3 7 5 6 7 8 2 2 8 7 6 1 3 5 3 2 7 6 7 8 6 5 2 2 8 2 2 6 6 4 9 6]
     [4 8 4 7 6 4 1 5 1 7 2 4 1 1 5 5 3 5 2 2 7 5 4 7 5 8 2 4 6 2 8 9]
     [9 2 7 4 1 7 4 4 7 1 9 7 4 5 3 8 7 8 8 4 1 9 9 8 4 9 3 1 1 8 6 3]
     [4 9 2 7 3 9 5 2 6 8 8 7 1 5 6 1 9 4 1 6 1 6 2 1 3 5 2 6 6 8 1 9]
     [8 3 9 4 9 7 7 4 2 8 4 1 7 9 3 9 1 3 8 7 6 1 4 9 1 6 8 7 6 3 2 2]
     [2 3 4 5 4 9 9 3 4 4 7 3 8 7 9 7 7 5 8 5 8 4 1 8 1 9 5 8 8 3 9 5]
     [7 7 5 6 6 1 4 7 9 7 6 2 3 5 7 1 3 5 9 2 2 4 6 9 4 5 9 7 2 3 8 3]
     [2 9 2 4 1 4 7 2 5 4 8 8 2 3 3 3 1 3 5 9 5 8 3 8 6 8 4 1 1 6 1 7]
     [7 1 8 5 2 6 6 6 7 1 7 4 2 1 5 9 6 4 2 8 4 3 2 5 9 1 3 9 1 9 3 9]
     [9 4 4 9 4 9 4 5 4 1 3 2 6 5 6 1 8 2 4 1 7 5 9 3 5 7 9 3 9 4 1 4]
     [1 6 2 1 7 1 5 2 8 8 6 4 4 2 5 2 5 8 1 2 9 3 1 1 8 6 9 4 2 2 1 8]
     [9 1 8 3 8 7 1 6 2 3 8 1 4 8 6 7 4 8 5 9 3 7 4 1 3 8 4 3 3 3 2 4]
     [9 4 5 6 2 2 3 7 2 2 3 3 2 8 5 4 5 5 5 5 1 5 8 4 4 1 1 3 8 5 3 8]
     [6 3 6 7 9 9 4 5 9 2 6 6 4 9 9 2 8 9 4 7 4 7 4 4 6 8 9 6 2 7 3 6]
     [9 1 5 8 8 8 5 9 6 8 4 9 4 2 3 6 2 2 4 8 2 6 6 4 6 7 6 9 5 8 5 9]
     [5 5 5 9 2 4 6 3 1 5 2 2 8 6 3 2 6 2 7 8 7 9 6 2 6 6 1 5 1 3 4 7]
     [6 6 9 1 2 3 4 1 1 5 3 2 3 4 5 5 3 8 6 6 9 1 5 9 2 2 9 4 4 6 2 2]]
    输入数据量化Tensor:  
    [1065353216 1073741824 1065353216 1073741824 1065353216 1065353216
     1065353216 1073741824 1073741824 1073741824 1065353216 1065353216
     1065353216 1065353216 1065353216 1073741824 1073741824 1065353216
     1073741824 1065353216 1073741824 1073741824 1065353216 1065353216
     1073741824 1065353216 1073741824 1073741824 1065353216 1073741824
     1065353216 1073741824]
    输出数据C矩阵: 
    [[ 943. 1676.  932. 1962.  893.  941.  817. 1528. 1778. 1740.  823.  715.
       659.  915.  818. 1500. 1710.  794. 1824.  890. 1558. 1938.  846.  827.
      1596. 1066. 1916. 1842.  822. 1860.  724. 1702.]
     [ 889. 1638.  814. 1730.  757.  863.  772. 1326. 1454. 1592.  780.  620.
       582.  821.  720. 1326. 1430.  715. 1632.  930. 1534. 1790.  751.  762.
      1380.  921. 1736. 1546.  721. 1712.  564. 1524.]
     [ 855. 1614.  847. 1774.  805.  873.  817. 1442. 1548. 1544.  776.  690.
       638.  849.  744. 1416. 1486.  755. 1668.  927. 1472. 1798.  750.  853.
      1456.  984. 1682. 1630.  731. 1800.  596. 1530.]
     [1033. 1746. 1044. 2034.  940. 1044.  873. 1764. 1860. 1816.  931.  802.
       717.  951.  910. 1742. 1832.  857. 1934. 1053. 1770. 2082.  904.  883.
      1818. 1126. 1934. 1972.  867. 2074.  729. 1890.]
     [ 902. 1650.  872. 1874.  821.  897.  850. 1482. 1736. 1530.  846.  746.
       632.  897.  830. 1496. 1582.  793. 1814.  976. 1564. 1954.  770.  851.
      1546. 1058. 1686. 1766.  749. 1930.  715. 1588.]
     [ 886. 1578.  900. 1740.  799.  913.  756. 1410. 1630. 1492.  737.  643.
       666.  819.  749. 1458. 1612.  762. 1596.  893. 1574. 1878.  832.  759.
      1494.  979. 1866. 1572.  703. 1750.  503. 1498.]
     [ 753. 1364.  754. 1576.  802.  818.  702. 1262. 1416. 1494.  746.  617.
       612.  775.  655. 1254. 1380.  690. 1578.  845. 1496. 1734.  663.  659.
      1500.  908. 1638. 1544.  693. 1566.  569. 1492.]
     [ 677. 1428.  767. 1478.  708.  704.  662. 1154. 1298. 1428.  627.  533.
       502.  709.  580. 1288. 1192.  585. 1526.  810. 1478. 1478.  617.  716.
      1342.  833. 1472. 1348.  647. 1508.  521. 1106.]
     [ 851. 1560.  858. 1662.  837.  854.  766. 1264. 1496. 1588.  813.  677.
       589.  821.  730. 1388. 1402.  758. 1792.  994. 1588. 1796.  673.  863.
      1472. 1029. 1650. 1616.  687. 1884.  613. 1378.]
     [ 751. 1388.  793. 1644.  755.  802.  683. 1236. 1374. 1494.  723.  569.
       600.  811.  750. 1276. 1482.  652. 1674.  888. 1500. 1702.  591.  673.
      1378.  906. 1442. 1632.  739. 1614.  605. 1420.]
     [ 683. 1436.  740. 1504.  696.  720.  652. 1160. 1588. 1438.  681.  568.
       526.  711.  630. 1306. 1376.  683. 1508.  816. 1456. 1684.  607.  682.
      1422.  866. 1542. 1366.  643. 1590.  511. 1224.]
     [ 873. 1678.  919. 1798.  854.  850.  814. 1350. 1750. 1726.  784.  651.
       619.  864.  775. 1522. 1492.  748. 1870.  977. 1714. 1850.  789.  857.
      1558. 1029. 1886. 1812.  750. 1896.  632. 1446.]
     [ 854. 1464.  787. 1644.  810.  922.  822. 1400. 1542. 1450.  872.  707.
       599.  785.  745. 1294. 1520.  757. 1536.  902. 1398. 1682.  690.  730.
      1500.  946. 1704. 1658.  676. 1736.  611. 1680.]
     [ 657. 1252.  676. 1350.  557.  690.  661. 1132. 1282. 1196.  651.  539.
       538.  654.  614. 1168. 1210.  530. 1388.  705. 1246. 1370.  597.  674.
      1216.  711. 1338. 1362.  524. 1372.  470. 1212.]
     [ 761. 1524.  814. 1636.  805.  906.  706. 1358. 1718. 1606.  797.  590.
       549.  813.  730. 1230. 1568.  737. 1604.  945. 1396. 1830.  676.  670.
      1516.  895. 1726. 1626.  744. 1676.  560. 1574.]
     [ 912. 1756.  910. 1832.  874.  961.  873. 1544. 1906. 1696.  859.  785.
       715.  847.  875. 1508. 1694.  861. 1762.  916. 1704. 2014.  818.  901.
      1670. 1089. 2064. 1926.  836. 1946.  666. 1806.]
     [ 903. 1526.  879. 1748.  865.  887.  848. 1536. 1604. 1480.  834.  677.
       672.  853.  800. 1386. 1490.  792. 1634.  954. 1610. 1864.  768.  811.
      1610. 1047. 1858. 1710.  677. 1794.  566. 1592.]
     [ 908. 1756.  893. 1928.  866.  944.  805. 1522. 1728. 1538.  847.  664.
       653.  868.  779. 1504. 1772.  805. 1832.  954. 1686. 1930.  801.  870.
      1814.  986. 1836. 1724.  773. 1860.  711. 1700.]
     [ 610. 1272.  634. 1334.  578.  681.  674.  988. 1342. 1236.  636.  585.
       520.  666.  652. 1082. 1238.  615. 1248.  652. 1246. 1472.  570.  612.
      1110.  836. 1324. 1412.  551. 1374.  483. 1278.]
     [ 853. 1486.  856. 1790.  754.  997.  838. 1456. 1616. 1528.  807.  674.
       638.  819.  749. 1328. 1606.  731. 1614.  937. 1520. 1904.  841.  777.
      1492. 1082. 1710. 1552.  756. 1740.  560. 1640.]
     [1024. 1736.  989. 1946.  916.  966.  862. 1676. 1646. 1832.  833.  722.
       712.  886.  804. 1638. 1594.  783. 1904.  970. 1644. 1860.  852.  933.
      1534. 1041. 1912. 1826.  846. 1946.  753. 1588.]
     [ 853. 1726.  833. 1888.  777.  757.  798. 1534. 1634. 1460.  752.  692.
       594.  749.  748. 1548. 1490.  705. 1644.  850. 1588. 1772.  818.  816.
      1664.  945. 1706. 1618.  753. 1764.  625. 1636.]
     [ 903. 1646.  959. 1848.  781. 1035.  813. 1446. 1828. 1662.  849.  684.
       647.  892.  839. 1332. 1736.  803. 1822. 1004. 1540. 1914.  792.  840.
      1662. 1018. 1802. 1992.  818. 1854.  663. 1820.]
     [ 827. 1442.  887. 1760.  882.  972.  749. 1342. 1744. 1552.  826.  570.
       655.  850.  779. 1530. 1724.  791. 1758.  908. 1654. 1836.  766.  737.
      1568. 1034. 1812. 1700.  781. 1676.  603. 1512.]
     [ 915. 1642.  953. 1814.  825.  944.  842. 1466. 1836. 1736.  883.  674.
       656.  868.  787. 1622. 1698.  852. 1922.  973. 1722. 1918.  853.  875.
      1672.  999. 1836. 1810.  809. 1922.  733. 1656.]
     [ 742. 1342.  725. 1580.  765.  819.  656. 1236. 1544. 1652.  739.  639.
       592.  770.  681. 1164. 1454.  732. 1506.  794. 1358. 1612.  621.  641.
      1382.  857. 1456. 1548.  704. 1552.  585. 1500.]
     [ 699. 1408.  751. 1612.  729.  795.  720. 1298. 1438. 1414.  632.  540.
       590.  674.  633. 1310. 1380.  656. 1392.  826. 1484. 1658.  670.  675.
      1440.  871. 1522. 1530.  697. 1508.  541. 1466.]
     [ 932. 1604.  911. 1844.  817.  824.  835. 1416. 1644. 1710.  826.  701.
       693.  857.  806. 1668. 1560.  768. 1910.  937. 1660. 1810.  759.  924.
      1522.  963. 1734. 1828.  760. 1958.  697. 1582.]
     [ 909. 1844.  923. 1772.  851.  962.  825. 1330. 1844. 1736.  823.  639.
       662.  889.  841. 1492. 1742.  884. 1674.  940. 1800. 1892.  809.  782.
      1574.  966. 2034. 1866.  814. 1826.  592. 1686.]
     [ 861. 1508.  839. 1670.  806.  884.  777. 1308. 1542. 1538.  838.  650.
       627.  865.  799. 1362. 1530.  753. 1824.  848. 1496. 1744.  755.  811.
      1362. 1018. 1798. 1700.  809. 1690.  628. 1524.]
     [ 916. 1632.  918. 1792.  847.  948.  807. 1450. 1622. 1644.  848.  752.
       655.  883.  830. 1530. 1636.  784. 1750.  959. 1636. 1852.  725.  860.
      1498. 1032. 1818. 1660.  752. 1950.  662. 1574.]
     [ 822. 1602.  807. 1662.  757.  812.  678. 1306. 1734. 1624.  840.  633.
       568.  804.  737. 1366. 1586.  830. 1734.  860. 1544. 1862.  747.  801.
      1578.  921. 1696. 1490.  689. 1740.  622. 1506.]]
    ```

-   示例三：通路CO1-\>C1，使能tensor量化功能接口。输入A矩阵和输入B矩阵的数据类型为half，输出C矩阵为float，使能ND2DN搬运，不使能量化，将mmad计算出的结果由float量化成float。

    ```
    AscendC::LocalTensor<l1out_T> dst_l0c = outQueueCO1.DeQue<l1out_T>();
    AscendC::LocalTensor<uint64_t> cbufWorkspace = deqQueue.DeQue<uint64_t>();
    uint16_t deqDataSize = AscendC::DivCeil(deq_size * sizeof(uint64_t), 128) * 128;
    float tmp = 0.5;
    uint64_t val = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AscendC::AlignUp(m, AscendC::BLOCK_CUBE)), m};
    fixpipeParams.params = {1, 0, 0, 1};
    fixpipeParams.reluEn = 1;
    AscendC::Fixpipe<dst_T, l1out_T, AscendC::CFG_COLUMN_MAJOR>(output_gm, dst_l0c, fixpipeParams);
    outQueueCO1.FreeTensor(dst_l0c);
    deqQueue.FreeTensor(cbufWorkspace);
      
    ```

-   示例四：通路CO1-\>C1，使能tensor量化功能接口。输入A矩阵和输入B矩阵的数据类型为half，输出C矩阵为float，使能ND2DN搬运，使能量化F322F16，将mmad计算出的结果由float量化成half。

    ```
    AscendC::LocalTensor<l1out_T> dst_l0c = outQueueCO1.DeQue<l1out_T>();
    AscendC::LocalTensor<uint64_t> cbufWorkspace = deqQueue.DeQue<uint64_t>();
    uint16_t deqDataSize = AscendC::DivCeil(deq_size * sizeof(uint64_t), 128) * 128;
    float tmp = 0.5;
    uint64_t val = static_cast<uint64_t>(*reinterpret_cast<int32_t*>(&tmp));
    AscendC::FixpipeParamsC310<AscendC::CO2Layout::COLUMN_MAJOR> fixpipeParams = {n, m, static_cast<uint16_t>(AscendC::AlignUp(m, AscendC::BLOCK_CUBE)), m};
    fixpipeParams.params = {1, 0, 0, 1};
    fixpipeParams.reluEn = 1;
    fixpipeParams.quantPre = F322F16;
    AscendC::Fixpipe<dst_T, l1out_T, AscendC::CFG_COLUMN_MAJOR>(output_gm, dst_l0c, fixpipeParams);
    outQueueCO1.FreeTensor(dst_l0c);
    deqQueue.FreeTensor(cbufWorkspace);
     
    ```

