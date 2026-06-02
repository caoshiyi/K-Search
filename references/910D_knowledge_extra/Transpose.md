# Transpose<a name="ZH-CN_TOPIC_0000002523344192"></a>

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
<td class="cellrowborder" align="center" valign="top" width="42%" headers="mcps1.1.3.1.2 "><p id="p37256491200"><a name="p37256491200"></a><a name="p37256491200"></a>√</p>
</td>
</tr>
</tbody>
</table>

## 功能说明<a name="section618mcpsimp"></a>

用于实现16\*16的二维矩阵数据块转置或者\[N,C,H,W\]与\[N,H,W,C\]数据格式互相转换。

## 函数原型<a name="section620mcpsimp"></a>

-   普通转置，支持16\*16的二维矩阵数据块进行转置

    ```
    template <typename T>
    __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T>& src)
    ```

-   增强转置，支持16\*16的二维矩阵数据块转置，支持\[N,C,H,W\]与\[N,H,W,C\]互相转换

    ```
    template <typename T>
    __aicore__ inline void Transpose(const LocalTensor<T>& dst, const LocalTensor<T> &src, const LocalTensor<uint8_t> &sharedTmpBuffer, const TransposeParamsExt &transposeParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table4835205712588"></a>
<table><thead align="left"><tr id="row118356578583"><th class="cellrowborder" valign="top" width="16.72%" id="mcps1.2.3.1.1"><p id="p48354572582"><a name="p48354572582"></a><a name="p48354572582"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.28%" id="mcps1.2.3.1.2"><p id="p583535795817"><a name="p583535795817"></a><a name="p583535795817"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row1835857145817"><td class="cellrowborder" valign="top" width="16.72%" headers="mcps1.2.3.1.1 "><p id="p5835457165816"><a name="p5835457165816"></a><a name="p5835457165816"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.28%" headers="mcps1.2.3.1.2 "><p id="p168351657155818"><a name="p168351657155818"></a><a name="p168351657155818"></a>操作数的数据类型。</p>
<a name="ul112101938192815"></a><a name="ul112101938192815"></a><ul id="ul112101938192815"><li><strong id="b194746392520"><a name="b194746392520"></a><a name="b194746392520"></a>普通转置:</strong><p id="p3451329122213"><a name="p3451329122213"></a><a name="p3451329122213"></a><span id="ph845102992219"><a name="ph845102992219"></a><a name="ph845102992219"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/half</p>
</li></ul>
<a name="ul1487794116284"></a><a name="ul1487794116284"></a><ul id="ul1487794116284"><li><strong id="b253617524133"><a name="b253617524133"></a><a name="b253617524133"></a>增强转置:</strong><a name="ul186007042913"></a><a name="ul186007042913"></a><ul id="ul186007042913"><li>transposeType为TRANSPOSE_ND2ND_B16：<p id="p119785012222"><a name="p119785012222"></a><a name="p119785012222"></a><span id="ph897155016228"><a name="ph897155016228"></a><a name="ph897155016228"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：uint16_t/int16_t/half</p>
</li><li>transposeType为TRANSPOSE_NCHW2NHWC或TRANSPOSE_NHWC2NCHW：<p id="p179916212235"><a name="p179916212235"></a><a name="p179916212235"></a><span id="ph157991222316"><a name="ph157991222316"></a><a name="ph157991222316"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/fp4x2_e2m1_t/fp4x2_e1m2_t/hifloat8_t/fp8_e5m2_t/fp8_e4m3fn_t/fp8_e8m0_t//int4x2_t/int16_t/uint16_t/half/bfloat16_t/int32_t/uint32_t/float/complex32</p>
</li></ul>
</li></ul>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.92169216921692%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="10.65106510651065%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="72.42724272427242%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.92169216921692%" headers="mcps1.2.4.1.1 "><p id="p20583144710373"><a name="p20583144710373"></a><a name="p20583144710373"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="10.65106510651065%" headers="mcps1.2.4.1.2 "><p id="p736835513915"><a name="p736835513915"></a><a name="p736835513915"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="72.42724272427242%" headers="mcps1.2.4.1.3 "><p id="p12944655134717"><a name="p12944655134717"></a><a name="p12944655134717"></a>目的操作数。</p>
<p id="p16703131355116"><a name="p16703131355116"></a><a name="p16703131355116"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p632555516496"><a name="p632555516496"></a><a name="p632555516496"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.92169216921692%" headers="mcps1.2.4.1.1 "><p id="p1725974393716"><a name="p1725974393716"></a><a name="p1725974393716"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="10.65106510651065%" headers="mcps1.2.4.1.2 "><p id="p15369205520396"><a name="p15369205520396"></a><a name="p15369205520396"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.42724272427242%" headers="mcps1.2.4.1.3 "><p id="p1491311164811"><a name="p1491311164811"></a><a name="p1491311164811"></a>源操作数。</p>
<p id="p1750601284810"><a name="p1750601284810"></a><a name="p1750601284810"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p98282011175715"><a name="p98282011175715"></a><a name="p98282011175715"></a><span id="ph1082813111575"><a name="ph1082813111575"></a><a name="ph1082813111575"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
<p id="p19196125422"><a name="p19196125422"></a><a name="p19196125422"></a>数据类型需要与dst保持一致。</p>
</td>
</tr>
<tr id="row2958858103018"><td class="cellrowborder" valign="top" width="16.92169216921692%" headers="mcps1.2.4.1.1 "><p id="p295905883015"><a name="p295905883015"></a><a name="p295905883015"></a>sharedTmpBuffer</p>
</td>
<td class="cellrowborder" valign="top" width="10.65106510651065%" headers="mcps1.2.4.1.2 "><p id="p1714314380312"><a name="p1714314380312"></a><a name="p1714314380312"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.42724272427242%" headers="mcps1.2.4.1.3 "><p id="p14959125873013"><a name="p14959125873013"></a><a name="p14959125873013"></a>共享的临时Buffer，sharedTmpBuffer的大小参考<a href="#table499712165910">表4</a>。</p>
</td>
</tr>
<tr id="row2222029103114"><td class="cellrowborder" valign="top" width="16.92169216921692%" headers="mcps1.2.4.1.1 "><p id="p3222122933117"><a name="p3222122933117"></a><a name="p3222122933117"></a>transposeParams</p>
</td>
<td class="cellrowborder" valign="top" width="10.65106510651065%" headers="mcps1.2.4.1.2 "><p id="p14222132916314"><a name="p14222132916314"></a><a name="p14222132916314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="72.42724272427242%" headers="mcps1.2.4.1.3 "><p id="p12596185919348"><a name="p12596185919348"></a><a name="p12596185919348"></a>控制Transpose的数据结构。结构体内包含：输入的shape信息和transposeType参数。该数据结构的定义请参考<a href="#table15780447181917">表3</a>。</p>
<a name="screen8348103082311"></a><a name="screen8348103082311"></a><pre class="screen" codetype="Cpp" id="screen8348103082311">struct TransposeParamsExt {
    __aicore__ TransposeParamsExt() {}
    __aicore__ TransposeParamsExt(const uint16_t nSizeIn, const uint16_t cSizeIn, const uint16_t hSizeIn,
        const uint16_t wSizeIn, const TransposeType transposeTypeIn)
        : nSize(nSizeIn),
          cSize(cSizeIn),
          hSize(hSizeIn),
          wSize(wSizeIn),
          transposeType(transposeTypeIn)
    {}
    uint16_t nSize = 0;
    uint16_t cSize = 0;
    uint16_t hSize = 0;
    uint16_t wSize = 0;
    TransposeType transposeType = TransposeType::TRANSPOSE_ND2ND_B16;
};</pre>
</td>
</tr>
</tbody>
</table>

**表 3**  TransposeParamsExt结构体内参数说明

<a name="table15780447181917"></a>
<table><thead align="left"><tr id="row0780947111915"><th class="cellrowborder" valign="top" width="17.18%" id="mcps1.2.3.1.1"><p id="p1780124771913"><a name="p1780124771913"></a><a name="p1780124771913"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="82.82000000000001%" id="mcps1.2.3.1.2"><p id="p1578014718198"><a name="p1578014718198"></a><a name="p1578014718198"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row10780647151919"><td class="cellrowborder" valign="top" width="17.18%" headers="mcps1.2.3.1.1 "><p id="p6340835122118"><a name="p6340835122118"></a><a name="p6340835122118"></a>nSize</p>
</td>
<td class="cellrowborder" valign="top" width="82.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p12340173514212"><a name="p12340173514212"></a><a name="p12340173514212"></a>n轴长度。默认值为0。</p>
<a name="ul448095971815"></a><a name="ul448095971815"></a><ul id="ul448095971815"><li>二维矩阵数据块转置，无需传入，传入数值无效。</li><li>[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：nSize∈[0, 65535]。</li></ul>
</td>
</tr>
<tr id="row6780947191919"><td class="cellrowborder" valign="top" width="17.18%" headers="mcps1.2.3.1.1 "><p id="p1934033512213"><a name="p1934033512213"></a><a name="p1934033512213"></a>cSize</p>
</td>
<td class="cellrowborder" valign="top" width="82.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1634012352218"><a name="p1634012352218"></a><a name="p1634012352218"></a>c轴长度。默认值为0。</p>
<a name="ul15113217102012"></a><a name="ul15113217102012"></a><ul id="ul15113217102012"><li>二维矩阵数据块转置，无需传入，传入数值无效。</li></ul>
<a name="ul143612115208"></a><a name="ul143612115208"></a><ul id="ul143612115208"><li>[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：cSize∈[0, 4095]</li></ul>
</td>
</tr>
<tr id="row1078074711194"><td class="cellrowborder" valign="top" width="17.18%" headers="mcps1.2.3.1.1 "><p id="p334033518217"><a name="p334033518217"></a><a name="p334033518217"></a>hSize</p>
</td>
<td class="cellrowborder" valign="top" width="82.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p1734053517219"><a name="p1734053517219"></a><a name="p1734053517219"></a>h轴长度。默认值为0。</p>
<a name="ul36679373203"></a><a name="ul36679373203"></a><ul id="ul36679373203"><li>二维矩阵数据块转置，固定传入16。</li></ul>
<a name="ul4667193782017"></a><a name="ul4667193782017"></a><ul id="ul4667193782017"><li>[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：hSize * wSize ∈[0, 4095]，hSize * wSize * sizeof(T)需要保证32B对齐。</li></ul>
</td>
</tr>
<tr id="row1761285762117"><td class="cellrowborder" valign="top" width="17.18%" headers="mcps1.2.3.1.1 "><p id="p0306336224"><a name="p0306336224"></a><a name="p0306336224"></a>wSize</p>
</td>
<td class="cellrowborder" valign="top" width="82.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p19481233193414"><a name="p19481233193414"></a><a name="p19481233193414"></a>w轴长度。默认值为0。</p>
<a name="ul55481746202212"></a><a name="ul55481746202212"></a><ul id="ul55481746202212"><li>二维矩阵数据块转置，固定传入16。</li></ul>
<a name="ul15548146142219"></a><a name="ul15548146142219"></a><ul id="ul15548146142219"><li>[N,C,H,W]与[N,H,W,C]数据格式互相转换，取值范围：hSize * wSize ∈[0, 4095]，hSize * wSize * sizeof(T)需要保证32B对齐。</li></ul>
</td>
</tr>
<tr id="row1751545416214"><td class="cellrowborder" valign="top" width="17.18%" headers="mcps1.2.3.1.1 "><p id="p88841831173112"><a name="p88841831173112"></a><a name="p88841831173112"></a>transposeType</p>
</td>
<td class="cellrowborder" valign="top" width="82.82000000000001%" headers="mcps1.2.3.1.2 "><p id="p4983113185917"><a name="p4983113185917"></a><a name="p4983113185917"></a>数据排布及reshape的类型，类型为TransposeType枚举类。默认值为TRANSPOSE_ND2ND_B16。</p>
<a name="screen398333135915"></a><a name="screen398333135915"></a><pre class="screen" codetype="Cpp" id="screen398333135915">enum class TransposeType : uint8_t {
    TRANSPOSE_TYPE_NONE,           // API不做任何处理
    TRANSPOSE_NZ2ND_0213,          // 当前不支持
    TRANSPOSE_NZ2NZ_0213,          // 当前不支持
    TRANSPOSE_NZ2NZ_012_WITH_N,    // 当前不支持
    TRANSPOSE_NZ2ND_012_WITH_N,    // 当前不支持
    TRANSPOSE_NZ2ND_012_WITHOUT_N, // 当前不支持
    TRANSPOSE_NZ2NZ_012_WITHOUT_N, // 当前不支持
    TRANSPOSE_ND2ND_ONLY,          // 当前不支持
    TRANSPOSE_ND_UB_GM,            // 当前不支持
    TRANSPOSE_GRAD_ND_UB_GM,       // 当前不支持
    TRANSPOSE_ND2ND_B16,           // [16,16]二维矩阵转置
    TRANSPOSE_NCHW2NHWC,           // [N,C,H,W]-&gt;[N,H,W,C]，
    TRANSPOSE_NHWC2NCHW            // [N,H,W,C]-&gt;[N,C,H,W]
};</pre>
</td>
</tr>
</tbody>
</table>

**表 4**  增强转置接口sharedTmpBuffer所需的大小

<a name="table499712165910"></a>
<table><thead align="left"><tr id="row1499719161599"><th class="cellrowborder" valign="top" width="17.39%" id="mcps1.2.3.1.1"><p id="p1699771616910"><a name="p1699771616910"></a><a name="p1699771616910"></a>transposeType</p>
</th>
<th class="cellrowborder" valign="top" width="82.61%" id="mcps1.2.3.1.2"><p id="p139976161918"><a name="p139976161918"></a><a name="p139976161918"></a>sharedTmpBuffer所需的大小</p>
</th>
</tr>
</thead>
<tbody><tr id="row399710160914"><td class="cellrowborder" valign="top" width="17.39%" headers="mcps1.2.3.1.1 "><p id="p139979161694"><a name="p139979161694"></a><a name="p139979161694"></a>TRANSPOSE_ND2ND_B16</p>
</td>
<td class="cellrowborder" valign="top" width="82.61%" headers="mcps1.2.3.1.2 "><p id="p109970161694"><a name="p109970161694"></a><a name="p109970161694"></a>不需要临时Buffer。</p>
</td>
</tr>
<tr id="row129982161491"><td class="cellrowborder" valign="top" width="17.39%" headers="mcps1.2.3.1.1 "><p id="p119986161799"><a name="p119986161799"></a><a name="p119986161799"></a>TRANSPOSE_NCHW2NHWC</p>
</td>
<td class="cellrowborder" valign="top" width="82.61%" headers="mcps1.2.3.1.2 "><a name="ul11404105801114"></a><a name="ul11404105801114"></a>
<p id="p1286618272148"><a name="p1286618272148"></a><a name="p1286618272148"></a>针对以下型号：</p>
<a name="ul128181642111411"></a><a name="ul128181642111411"></a><ul id="ul128181642111411"><li><span id="ph1724348161314"><a name="ph1724348161314"></a><a name="ph1724348161314"></a>Ascend 950PR/Ascend 950DT</span></li></ul>
<p id="p104139415132"><a name="p104139415132"></a><a name="p104139415132"></a>临时Buffer的大小按照下述计算规则（伪代码）进行计算。</p>
<a name="screen397648171517"></a><a name="screen397648171517"></a><pre class="screen" codetype="Cpp" id="screen397648171517">auto h0 = 16; // 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16
auto w0 = 32 / sizeof(type);  // type代表数据类型
auto tmpBufferSize = (cSize + 2)  * h0 * w0 * sizeof(type);</pre>
</td>
</tr>
<tr id="row15998121618916"><td class="cellrowborder" valign="top" width="17.39%" headers="mcps1.2.3.1.1 "><p id="p1299881616915"><a name="p1299881616915"></a><a name="p1299881616915"></a>TRANSPOSE_NHWC2NCHW</p>
</td>
<td class="cellrowborder" valign="top" width="82.61%" headers="mcps1.2.3.1.2 "><a name="ul1551155015179"></a><a name="ul1551155015179"></a>
<p id="p17512185071711"><a name="p17512185071711"></a><a name="p17512185071711"></a>针对以下型号：</p>
<a name="ul1251255051718"></a><a name="ul1251255051718"></a><ul id="ul1251255051718"><li><span id="ph151285031720"><a name="ph151285031720"></a><a name="ph151285031720"></a>Ascend 950PR/Ascend 950DT</span></li></ul>
<p id="p2051255019175"><a name="p2051255019175"></a><a name="p2051255019175"></a>临时Buffer的大小按照下述计算规则（伪代码）进行计算。</p>
<a name="screen45128509174"></a><a name="screen45128509174"></a><pre class="screen" codetype="Cpp" id="screen45128509174">auto h0 = 16; // 当数据类型的位宽为8时，h0 = 32；其他情况下，h0 = 16
auto w0 = 32 / sizeof(type);  // type代表数据类型
auto tmpBufferSize = (cSize  * 2 + 1)  * h0 * w0 * sizeof(type);</pre>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   普通转置接口支持src和dst复用。
-   增强转置接口，transposeType为TRANSPOSE\_ND2ND\_B16时支持src和dst复用，transposeType为TRANSPOSE\_NCHW2NHWC、TRANSPOSE\_NHWC2NCHW时不支持src和dst复用。

## 调用示例<a name="section19372434133520"></a>

-   普通接口调用示例片段，完整片段请参考[transpose\_common样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/09_transpose/transpose_common)，该示例对\[16,16\]的half类型矩阵进行转置。

    ```
    // dstLocal：目的操作数tensor
    // srcLocal：源操作数tensor
    AscendC::Transpose<half>(dstLocal, srcLocal);
    ```

    ```
    #include "kernel_operator.h"
    
    class KernelTranspose {
    public:
        __aicore__ inline KernelTranspose() {}
        __aicore__ inline void Init(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
        {
            srcGlobal.SetGlobalBuffer((__gm__ half*)src);
            dstGlobal.SetGlobalBuffer((__gm__ half*)dstGm);
    
            pipe.InitBuffer(inQueueSrc, 1, srcDataSize * sizeof(half));
            pipe.InitBuffer(outQueueDst, 1, dstDataSize * sizeof(half));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.AllocTensor<half>();
            AscendC::DataCopy(srcLocal, srcGlobal, srcDataSize);
            inQueueSrc.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<half> srcLocal = inQueueSrc.DeQue<half>();
            AscendC::LocalTensor<half> dstLocal = outQueueDst.AllocTensor<half>();
    
            AscendC::Transpose<half>(dstLocal, srcLocal);
    
            outQueueDst.EnQue<half>(dstLocal);
            inQueueSrc.FreeTensor(srcLocal);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<half> dstLocal = outQueueDst.DeQue<half>();
            AscendC::DataCopy(dstGlobal, dstLocal, dstDataSize);
            outQueueDst.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrc;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueDst;
    
        AscendC::GlobalTensor<half> srcGlobal, dstGlobal;
        int srcDataSize = 256;
        int dstDataSize = 256;
    };
    
    extern "C" __global__ __aicore__ void transpose_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dstGm)
    {
        KernelTranspose op;
        op.Init(src, dstGm);
        op.Process();
    }
    ```

    ```
    输入数据src_gm:
    [[  0.   1.   2.   3.   4.   5.   6.   7.   8.   9.  10.  11.  12.  13.
       14.  15.]
     [ 16.  17.  18.  19.  20.  21.  22.  23.  24.  25.  26.  27.  28.  29.
       30.  31.]
     [ 32.  33.  34.  35.  36.  37.  38.  39.  40.  41.  42.  43.  44.  45.
       46.  47.]
     [ 48.  49.  50.  51.  52.  53.  54.  55.  56.  57.  58.  59.  60.  61.
       62.  63.]
     [ 64.  65.  66.  67.  68.  69.  70.  71.  72.  73.  74.  75.  76.  77.
       78.  79.]
     [ 80.  81.  82.  83.  84.  85.  86.  87.  88.  89.  90.  91.  92.  93.
       94.  95.]
     [ 96.  97.  98.  99. 100. 101. 102. 103. 104. 105. 106. 107. 108. 109.
      110. 111.]
     [112. 113. 114. 115. 116. 117. 118. 119. 120. 121. 122. 123. 124. 125.
      126. 127.]
     [128. 129. 130. 131. 132. 133. 134. 135. 136. 137. 138. 139. 140. 141.
      142. 143.]
     [144. 145. 146. 147. 148. 149. 150. 151. 152. 153. 154. 155. 156. 157.
      158. 159.]
     [160. 161. 162. 163. 164. 165. 166. 167. 168. 169. 170. 171. 172. 173.
      174. 175.]
     [176. 177. 178. 179. 180. 181. 182. 183. 184. 185. 186. 187. 188. 189.
      190. 191.]
     [192. 193. 194. 195. 196. 197. 198. 199. 200. 201. 202. 203. 204. 205.
      206. 207.]
     [208. 209. 210. 211. 212. 213. 214. 215. 216. 217. 218. 219. 220. 221.
      222. 223.]
     [224. 225. 226. 227. 228. 229. 230. 231. 232. 233. 234. 235. 236. 237.
      238. 239.]
     [240. 241. 242. 243. 244. 245. 246. 247. 248. 249. 250. 251. 252. 253.
      254. 255.]]
    
    输出数据dst_gm:
    [[  0.  16.  32.  48.  64.  80.  96. 112. 128. 144. 160. 176. 192. 208.
      224. 240.]
     [  1.  17.  33.  49.  65.  81.  97. 113. 129. 145. 161. 177. 193. 209.
      225. 241.]
     [  2.  18.  34.  50.  66.  82.  98. 114. 130. 146. 162. 178. 194. 210.
      226. 242.]
     [  3.  19.  35.  51.  67.  83.  99. 115. 131. 147. 163. 179. 195. 211.
      227. 243.]
     [  4.  20.  36.  52.  68.  84. 100. 116. 132. 148. 164. 180. 196. 212.
      228. 244.]
     [  5.  21.  37.  53.  69.  85. 101. 117. 133. 149. 165. 181. 197. 213.
      229. 245.]
     [  6.  22.  38.  54.  70.  86. 102. 118. 134. 150. 166. 182. 198. 214.
      230. 246.]
     [  7.  23.  39.  55.  71.  87. 103. 119. 135. 151. 167. 183. 199. 215.
      231. 247.]
     [  8.  24.  40.  56.  72.  88. 104. 120. 136. 152. 168. 184. 200. 216.
      232. 248.]
     [  9.  25.  41.  57.  73.  89. 105. 121. 137. 153. 169. 185. 201. 217.
      233. 249.]
     [ 10.  26.  42.  58.  74.  90. 106. 122. 138. 154. 170. 186. 202. 218.
      234. 250.]
     [ 11.  27.  43.  59.  75.  91. 107. 123. 139. 155. 171. 187. 203. 219.
      235. 251.]
     [ 12.  28.  44.  60.  76.  92. 108. 124. 140. 156. 172. 188. 204. 220.
      236. 252.]
     [ 13.  29.  45.  61.  77.  93. 109. 125. 141. 157. 173. 189. 205. 221.
      237. 253.]
     [ 14.  30.  46.  62.  78.  94. 110. 126. 142. 158. 174. 190. 206. 222.
      238. 254.]
     [ 15.  31.  47.  63.  79.  95. 111. 127. 143. 159. 175. 191. 207. 223.
      239. 255.]]
    ```

-   增强接口调用示例片段，完整片段请参考[transpose\_enhanced样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/09_transpose/transpose_enhanced)，完成half类型的\[N,C,H,W\]-\>\[N,H,W,C\]转置。

    ```
    AscendC::TransposeParamsExt transposeParams;
    transposeParams.nSize = N; // N轴长度
    transposeParams.cSize = C; // C轴长度
    transposeParams.hSize = H; // H轴长度
    transposeParams.wSize = W; // W轴长度
    transposeParams.transposeType = transposeType; 
    AscendC::Transpose(dstLocal, srcLocal, stackBuffer, transposeParams);
    ```

    ```
    #include "kernel_operator.h"
    
    template <typename T>
    class Kernel4dTrans {
    public:
        __aicore__ inline Kernel4dTrans() {}
        __aicore__ inline void Init(__gm__ uint8_t *srcGm, __gm__ uint8_t *dstGm)
        {
            inputSize = N * C * H * W;
            tmpBufferSize = (C + 2) * 16 * 16;
            srcGlobal.SetGlobalBuffer((__gm__ T *)srcGm);
            dstGlobal.SetGlobalBuffer((__gm__ T *)dstGm);
            pipe.InitBuffer(inQueueSrcVecIn, 1, inputSize*sizeof(T));
            pipe.InitBuffer(inQueueSrcVecOut, 1, inputSize*sizeof(T));
            pipe.InitBuffer(tmpQueue, 1, tmpBufferSize * sizeof(T));
        }
        __aicore__ inline void Process()
        {
            CopyIn();
            Compute();
            CopyOut();
        }
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.AllocTensor<T>();
            AscendC::DataCopy(srcLocal, srcGlobal, inputSize);
            inQueueSrcVecIn.EnQue(srcLocal);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<T> srcLocal = inQueueSrcVecIn.DeQue<T>();
            AscendC::LocalTensor<T> dstLocal = inQueueSrcVecOut.AllocTensor<T>();
            AscendC::LocalTensor<uint8_t> stackBuffer = tmpQueue.AllocTensor<uint8_t>();
    
            AscendC::TransposeParamsExt transposeParams;
            transposeParams.nSize = N;
            transposeParams.cSize = C;
            transposeParams.hSize = H;
            transposeParams.wSize = W;
            transposeParams.transposeType = transposeType;
            AscendC::Transpose(dstLocal, srcLocal, stackBuffer, transposeParams);
            inQueueSrcVecOut.EnQue<T>(dstLocal);
            inQueueSrcVecIn.FreeTensor(srcLocal);
            tmpQueue.FreeTensor(stackBuffer);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<T> dstLocal = inQueueSrcVecOut.DeQue<T>();
            AscendC::DataCopy(dstGlobal, dstLocal, inputSize);
            inQueueSrcVecOut.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe pipe;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueSrcVecIn;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> inQueueSrcVecOut;
        AscendC::TQue<AscendC::TPosition::VECCALC, 1> tmpQueue;
    
        AscendC::GlobalTensor<T> srcGlobal;
        AscendC::GlobalTensor<T> dstGlobal;
        uint32_t N = 3;
        uint32_t C = 3;
        uint32_t H = 2;
        uint32_t W = 8;
        uint32_t inputSize, tmpBufferSize;
        AscendC::TransposeType transposeType = AscendC::TransposeType::TRANSPOSE_NCHW2NHWC;
    };
    
    extern "C" __global__ __aicore__ void transpose_kernel(__gm__ uint8_t* srcGm, __gm__ uint8_t* dstGm)
    {
        Kernel4dTrans<half>op;
        op.Init(srcGm, dstGm);
        op.Process();
    }
    ```

