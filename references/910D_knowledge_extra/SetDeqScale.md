# SetDeqScale<a name="ZH-CN_TOPIC_0000002523303556"></a>

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

设置DEQSCALE寄存器的值。

## 函数原型<a name="section620mcpsimp"></a>

-   用于[AddDeqRelu](AddDeqRelu.md)/[Cast](Cast.md)/[CastDeq](CastDequant.md)的s322f16场景

    ```
    __aicore__ inline void SetDeqScale(half scale)
    ```

-   用于[CastDeq](CastDequant.md)（isVecDeq=false）的场景

    ```
    __aicore__ inline void SetDeqScale(float scale, int16_t offset, bool signMode)
    ```

-   用于[CastDeq](CastDequant.md)（isVecDeq=true）的场景

    ```
    template <typename T>
    __aicore__ inline void SetDeqScale(const LocalTensor<T>& vdeq, const VdeqInfo& vdeqInfo)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table1089016373553"></a>
<table><thead align="left"><tr id="row68901437105514"><th class="cellrowborder" valign="top" width="18.55%" id="mcps1.2.3.1.1"><p id="p18207657115516"><a name="p18207657115516"></a><a name="p18207657115516"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="81.45%" id="mcps1.2.3.1.2"><p id="p1636802562"><a name="p1636802562"></a><a name="p1636802562"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5891437155519"><td class="cellrowborder" valign="top" width="18.55%" headers="mcps1.2.3.1.1 "><p id="p12891173785519"><a name="p12891173785519"></a><a name="p12891173785519"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="81.45%" headers="mcps1.2.3.1.2 "><p id="p289153775520"><a name="p289153775520"></a><a name="p289153775520"></a>输入量化Tensor的数据类型。支持的数据类型为uint64_t。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  参数说明

<a name="zh-cn_topic_0235751031_table33761356"></a>
<table><thead align="left"><tr id="zh-cn_topic_0235751031_row27598891"><th class="cellrowborder" valign="top" width="18.52%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0235751031_p20917673"><a name="zh-cn_topic_0235751031_p20917673"></a><a name="zh-cn_topic_0235751031_p20917673"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="10.07%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0235751031_p16609919"><a name="zh-cn_topic_0235751031_p16609919"></a><a name="zh-cn_topic_0235751031_p16609919"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.41%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0235751031_p59995477"><a name="zh-cn_topic_0235751031_p59995477"></a><a name="zh-cn_topic_0235751031_p59995477"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row42461942101815"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p479605232211"><a name="p479605232211"></a><a name="p479605232211"></a>scale（half）</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p1579635215228"><a name="p1579635215228"></a><a name="p1579635215228"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1179555214221"><a name="p1179555214221"></a><a name="p1179555214221"></a>scale量化参数，half类型。</p>
<p id="p455695691814"><a name="p455695691814"></a><a name="p455695691814"></a><span id="ph8949105641813"><a name="ph8949105641813"></a><a name="ph8949105641813"></a>Ascend 950PR/Ascend 950DT</span>：用于<a href="AddDeqRelu.md">AddDeqRelu</a>/<a href="CastDequant.md">CastDeq</a>/<a href="Cast.md">Cast</a>的s322f16场景。</p>
</td>
</tr>
<tr id="row2137145181815"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p179035252218"><a name="p179035252218"></a><a name="p179035252218"></a>scale（float）</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p7789185214226"><a name="p7789185214226"></a><a name="p7789185214226"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p20789195232218"><a name="p20789195232218"></a><a name="p20789195232218"></a>scale量化参数，float类型。</p>
<p id="p1249811013143"><a name="p1249811013143"></a><a name="p1249811013143"></a>用于<a href="CastDequant.md">CastDeq</a>（isVecDeq=false）场景设置DEQSCALE寄存器的值。</p>
</td>
</tr>
<tr id="row137797127280"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p16783195216221"><a name="p16783195216221"></a><a name="p16783195216221"></a>offset</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p14782205213226"><a name="p14782205213226"></a><a name="p14782205213226"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p16781115219227"><a name="p16781115219227"></a><a name="p16781115219227"></a>offset量化参数，int16_t类型，只有前9位有效。</p>
<p id="p6760911151412"><a name="p6760911151412"></a><a name="p6760911151412"></a>用于<a href="CastDequant.md">CastDeq</a>（isVecDeq=false）的场景，设置offset。</p>
</td>
</tr>
<tr id="row19615183817191"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p6780452102214"><a name="p6780452102214"></a><a name="p6780452102214"></a>signMode</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p4780145232218"><a name="p4780145232218"></a><a name="p4780145232218"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p1779752192216"><a name="p1779752192216"></a><a name="p1779752192216"></a>bool类型，表示量化结果是否带符号。</p>
<p id="p13636582366"><a name="p13636582366"></a><a name="p13636582366"></a>用于<a href="CastDequant.md">CastDeq</a>（isVecDeq=false）的场景，设置signMode。</p>
</td>
</tr>
<tr id="row250163914199"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p167521752102219"><a name="p167521752102219"></a><a name="p167521752102219"></a>vdeq</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p157511152112217"><a name="p157511152112217"></a><a name="p157511152112217"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p19784265526"><a name="p19784265526"></a><a name="p19784265526"></a>用于<a href="CastDequant.md">CastDeq</a>（isVecDeq=true）的场景，输入量化tensor，大小为128Byte。</p>
<p id="p167301549113818"><a name="p167301549113818"></a><a name="p167301549113818"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p11840758205316"><a name="p11840758205316"></a><a name="p11840758205316"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row10671575013"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.4.1.1 "><p id="p7681716012"><a name="p7681716012"></a><a name="p7681716012"></a>vdeqInfo</p>
</td>
<td class="cellrowborder" valign="top" width="10.07%" headers="mcps1.2.4.1.2 "><p id="p14682711012"><a name="p14682711012"></a><a name="p14682711012"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.41%" headers="mcps1.2.4.1.3 "><p id="p868177704"><a name="p868177704"></a><a name="p868177704"></a>存储量化tensor信息的数据结构，结构体内包含量化tensor中的16组量化参数</p>
<a name="screen2023315320313"></a><a name="screen2023315320313"></a><pre class="screen" codetype="Cpp" id="screen2023315320313">const uint8_t VDEQ_TENSOR_SIZE = 16;

struct VdeqInfo {
    __aicore__ VdeqInfo() {}
    __aicore__ VdeqInfo(const float vdeqScaleIn[VDEQ_TENSOR_SIZE], const int16_t vdeqOffsetIn[VDEQ_TENSOR_SIZE],
        const bool vdeqSignModeIn[VDEQ_TENSOR_SIZE])
    {
        for (int32_t i = 0; i &lt; VDEQ_TENSOR_SIZE; ++i) {
            vdeqScale[i] = vdeqScaleIn[i];
            vdeqOffset[i] = vdeqOffsetIn[i];
            vdeqSignMode[i] = vdeqSignModeIn[i];
        }
    }

    float vdeqScale[VDEQ_TENSOR_SIZE] = { 0 };
    int16_t vdeqOffset[VDEQ_TENSOR_SIZE] = { 0 };
    bool vdeqSignMode[VDEQ_TENSOR_SIZE] = { 0 };
};</pre>
<a name="ul11740315281"></a><a name="ul11740315281"></a><ul id="ul11740315281"><li>vdeqScale：float类型的数组，用于存储量化tensor中的scale参数scale<sub id="sub11593111534520"><a name="sub11593111534520"></a><a name="sub11593111534520"></a>0</sub>-scale<sub id="sub759310155456"><a name="sub759310155456"></a><a name="sub759310155456"></a>15</sub>。</li><li>vdeqOffset：int16_t类型的数组，用于存储量化tensor中的offset参数offset<sub id="sub676414442457"><a name="sub676414442457"></a><a name="sub676414442457"></a>0</sub>-offset<sub id="sub476418440452"><a name="sub476418440452"></a><a name="sub476418440452"></a>15</sub>。</li><li>vdeqSignMode：bool类型的数组，用于存储量化tensor中的signMode参数signMode<sub id="sub1501201244618"><a name="sub1501201244618"></a><a name="sub1501201244618"></a>0</sub>-signMode<sub id="sub950117129469"><a name="sub950117129469"></a><a name="sub950117129469"></a>15</sub>。</li></ul>
</td>
</tr>
</tbody>
</table>

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

无

## 调用示例<a name="section837496171220"></a>

-   SetDeqScale\(half scale\)

    ```
    // 配合Cast的s322f16场景使用
    // dstLocal为half类型的LocalTensor，srcLocal为int32_t类型的LocalTensor
    uint32_t srcSize = 256; // 参与计算的元素个数
    half scale = 1.0; // 量化参数为1
    AscendC::SetDeqScale(scale);
    // dst = src
    AscendC::Cast(dstLocal, srcLocal, AscendC::RoundMode::CAST_NONE, srcSize);
    ```

-   SetDeqScale\(float scale, int16\_t offset, bool signMode\)

    ```
    // 配合CastDeq（isVecDeq=false）场景使用
    // dstLocal为int16_t类型的LocalTensor，srcLocal为int8_t类型的LocalTensor
    uint32_t srcSize = 256; // 参与计算的元素个数
    float scale = 1.0; // 量化参数为1
    int16_t offset = 0; // 不带偏移
    bool signMode = true; // dstLocal为int8_t类型，为有符号数
    AscendC::SetDeqScale(scale, offset, signMode);
    // dst = src
    AscendC::CastDeq<int8_t, int16_t, false, false>(dstLocal, srcLocal, srcSize);
    ```

-   SetDeqScale\(const LocalTensor<T\>& vdeq, const VdeqInfo& vdeqInfo\)

    ```
    // 配合CastDeq（isVecDeq=true）场景使用
    // dstLocal为int16_t类型的LocalTensor，srcLocal为int8_t类型的LocalTensor
    uint32_t srcSize = 256; // 参与计算的元素个数
    float vdeqScale[16] = { 0 };
    int16_t vdeqOffset[16] = { 0 };
    bool vdeqSignMode[16] = { 0 };
    for (int i = 0; i < 16; i++) {
        vdeqScale[i] = 1.0; // 量化参数为1
        vdeqOffset[i] = 0; // 不带偏移
        vdeqSignMode[i] = true; // dstLocal为int8_t类型，为有符号数
    }
    AscendC::VdeqInfo vdeqInfo(vdeqScale, vdeqOffset, vdeqSignMode);
    AscendC::SetDeqScale<uint64_t>(tmpBuffer, vdeqInfo);
    // dst = src
    AscendC::CastDeq<int8_t, int16_t, true, false>(dstLocal, srcLocal, srcSize);
    ```

