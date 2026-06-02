# CastDequant<a name="ZH-CN_TOPIC_0000002523344880"></a>

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

对输入做量化并进行精度转换。不同的数据类型，转换公式不同。

-   在输入类型为int16\_t的情况下，对int16\_t类型的输入做量化并进行精度转换，得到int8\_t/uint8\_t类型的数据。使用该接口前需要调用[SetDeqScale](SetDeqScale.md)设置scale、offset、signMode等量化参数。

    通过模板参数isVecDeq控制是否选择向量量化模式。

    -   当isVecDeq=false时，根据SetDeqScale设置的scale、offset、signMode，对输入做量化并进行精度转换。计算公式如下：

        <!-- img2text -->
$$
\text{dstLocal}_{n}=\operatorname{CastToHalf}\left(\operatorname{Round}\left(\text{srcLocal}_{n}\times \text{scale}\times \left(\operatorname{signMode}?2:1\right)+\text{offset}\right)\right)
$$

    -   当isVecDeq=true时，根据SetDeqScale设置的一段128B的UB上的16组量化参数scale<sub>0</sub>-scale<sub>15</sub>、offset<sub>0</sub>-offset<sub>15</sub>、signMode<sub>0</sub>-signMode<sub>15</sub>，以循环的方式对输入做量化并进行精度转换。计算公式如下：

        <!-- img2text -->
$$
\text{tmp} = x \times \text{scale}_i + \text{offset}_i,\ i = 0,1,\ldots,15
$$

$$
\text{dst} =
\begin{cases}
\operatorname{float2half}(\operatorname{round}(\text{tmp})), & \text{if signMode}_i = 1 \\
\operatorname{float2half}(\operatorname{cast}_{\text{uint16}}(\operatorname{round}(\text{tmp}))), & \text{if signMode}_i = 0
\end{cases}
$$

-   在输入类型为int32\_t的情况下，对int32\_t类型的输入做量化并进行精度转换，得到half类型的数据。使用该接口前需要调用[SetDeqScale](SetDeqScale.md)设置scale参数。

    .<!-- img2text -->
$$dst = src \times scale$$

## 函数原型<a name="section620mcpsimp"></a>

-   tensor前n个数据计算

    ```
    template <typename T, typename U, bool isVecDeq = true, bool halfBlock = true>
    __aicore__ inline void CastDequant(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint32_t count)
    ```

-   tensor高维切分计算
    -   mask逐bit模式

        ```
        template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
        __aicore__ inline void CastDequant(const LocalTensor<T>& dst, const LocalTensor<U>& src, const uint64_t mask[], uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

    -   mask连续模式

        ```
        template <typename T, typename U, bool isSetMask = true, bool isVecDeq = true, bool halfBlock = true>
        __aicore__ inline void CastDequant(const LocalTensor<T>& dst, const LocalTensor<U>& src, const int32_t mask, uint8_t repeatTime, const UnaryRepeatParams& repeatParams)
        ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table202061686114"></a>
<table><thead align="left"><tr id="row1469044745019"><th class="cellrowborder" valign="top" width="16.400000000000002%" id="mcps1.2.3.1.1"><p id="p46912472503"><a name="p46912472503"></a><a name="p46912472503"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="83.6%" id="mcps1.2.3.1.2"><p id="p1569124715020"><a name="p1569124715020"></a><a name="p1569124715020"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row162067818117"><td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.3.1.1 "><p id="p15502050112916"><a name="p15502050112916"></a><a name="p15502050112916"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.6%" headers="mcps1.2.3.1.2 "><p id="p1355065012291"><a name="p1355065012291"></a><a name="p1355065012291"></a>输出Tensor的数据类型。</p>
<p id="p7744172741311"><a name="p7744172741311"></a><a name="p7744172741311"></a><span id="ph1574417270132"><a name="ph1574417270132"></a><a name="ph1574417270132"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int8_t/uint8_t/half</p>
<p id="p21661930191919"><a name="p21661930191919"></a><a name="p21661930191919"></a>和SetDeqScale接口的signMode入参配合使用，当signMode=true时输出数据类型int8_t；signMode=false时输出数据类型uint8_t。</p>
</td>
</tr>
<tr id="row182061281112"><td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.3.1.1 "><p id="p12653546297"><a name="p12653546297"></a><a name="p12653546297"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.6%" headers="mcps1.2.3.1.2 "><p id="p15266125412910"><a name="p15266125412910"></a><a name="p15266125412910"></a>输入Tensor的数据类型。</p>
<p id="p184813145116"><a name="p184813145116"></a><a name="p184813145116"></a><span id="ph448111142119"><a name="ph448111142119"></a><a name="ph448111142119"></a>Ascend 950PR/Ascend 950DT</span>，支持的数据类型为：int16_t/int32_t</p>
</td>
</tr>
<tr id="row112061287110"><td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.3.1.1 "><p id="p132231010152718"><a name="p132231010152718"></a><a name="p132231010152718"></a>isSetMask</p>
</td>
<td class="cellrowborder" valign="top" width="83.6%" headers="mcps1.2.3.1.2 "><p id="zh-cn_topic_0000002554343931_p77520541653"><a name="zh-cn_topic_0000002554343931_p77520541653"></a><a name="zh-cn_topic_0000002554343931_p77520541653"></a>是否在接口内部设置mask。</p>
<a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><a name="zh-cn_topic_0000002554343931_ul1163765616511"></a><ul id="zh-cn_topic_0000002554343931_ul1163765616511"><li>true，表示在接口内部设置mask。</li><li>false，表示在接口外部设置mask，开发者需要使用<a href="SetVectorMask.md">SetVectorMask</a>接口设置mask值。这种模式下，本接口入参中的mask值必须设置为占位符MASK_PLACEHOLDER。</li></ul>
</td>
</tr>
<tr id="row1154215474018"><td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.3.1.1 "><p id="p254213477015"><a name="p254213477015"></a><a name="p254213477015"></a>isVecDeq</p>
</td>
<td class="cellrowborder" valign="top" width="83.6%" headers="mcps1.2.3.1.2 "><p id="p754217476010"><a name="p754217476010"></a><a name="p754217476010"></a>控制是否选择向量量化模式。和SetDeqScale(const LocalTensor&lt;T&gt;&amp; src)接口配合使用，当SetDeqScale接口传入Tensor时，isVecDeq必须为true。</p>
</td>
</tr>
<tr id="row11364173815014"><td class="cellrowborder" valign="top" width="16.400000000000002%" headers="mcps1.2.3.1.1 "><p id="p1636411381707"><a name="p1636411381707"></a><a name="p1636411381707"></a>halfBlock</p>
</td>
<td class="cellrowborder" valign="top" width="83.6%" headers="mcps1.2.3.1.2 "><p id="p236463810014"><a name="p236463810014"></a><a name="p236463810014"></a>对int16_t类型的输入做量化并进行精度转换得到int8_t/uint8_t类型的数据时，halfBlock参数用于指示输出元素存放在上半还是下半Block。halfBlock=true时，结果存放在下半Block；halfBlock=false时，结果存放在上半Block，如图<a href="#fig1084698268">图1</a>。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  接口参数说明

<a name="table1055216132132"></a>
<table><thead align="left"><tr id="row105531513121315"><th class="cellrowborder" valign="top" width="16.49%" id="mcps1.2.4.1.1"><p id="p5553171319138"><a name="p5553171319138"></a><a name="p5553171319138"></a>参数名</p>
</th>
<th class="cellrowborder" valign="top" width="11.92%" id="mcps1.2.4.1.2"><p id="p5553151313131"><a name="p5553151313131"></a><a name="p5553151313131"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.59%" id="mcps1.2.4.1.3"><p id="p655316136139"><a name="p655316136139"></a><a name="p655316136139"></a>描述</p>
</th>
</tr>
</thead>
<tbody><tr id="row5553201314135"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p8553813111314"><a name="p8553813111314"></a><a name="p8553813111314"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p755318134134"><a name="p755318134134"></a><a name="p755318134134"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p358015811379"><a name="p358015811379"></a><a name="p358015811379"></a>目的操作数。</p>
<p id="p5945720195112"><a name="p5945720195112"></a><a name="p5945720195112"></a><span id="zh-cn_topic_0000002523303824_ph173308471594"><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><a name="zh-cn_topic_0000002523303824_ph173308471594"></a><span id="zh-cn_topic_0000002523303824_ph9902231466"><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><a name="zh-cn_topic_0000002523303824_ph9902231466"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816"><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p6526472369"><a name="p6526472369"></a><a name="p6526472369"></a><span id="ph1479701815419"><a name="ph1479701815419"></a><a name="ph1479701815419"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row6553613191315"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p195531113161311"><a name="p195531113161311"></a><a name="p195531113161311"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p155310135134"><a name="p155310135134"></a><a name="p155310135134"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p1743571233810"><a name="p1743571233810"></a><a name="p1743571233810"></a>源操作数。</p>
<p id="p169251414143819"><a name="p169251414143819"></a><a name="p169251414143819"></a><span id="zh-cn_topic_0000002523303824_ph173308471594_1"><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><a name="zh-cn_topic_0000002523303824_ph173308471594_1"></a><span id="zh-cn_topic_0000002523303824_ph9902231466_1"><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><a name="zh-cn_topic_0000002523303824_ph9902231466_1"></a><span id="zh-cn_topic_0000002523303824_ph1782115034816_1"><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a><a name="zh-cn_topic_0000002523303824_ph1782115034816_1"></a>类型为<a href="LocalTensor.md">LocalTensor</a>，支持的TPosition为VECIN/VECCALC/VECOUT。</span></span></span></p>
<p id="p8146174915365"><a name="p8146174915365"></a><a name="p8146174915365"></a><span id="ph1959874963611"><a name="ph1959874963611"></a><a name="ph1959874963611"></a>LocalTensor的起始地址需要32字节对齐。</span></p>
</td>
</tr>
<tr id="row16554713131317"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p2554141321313"><a name="p2554141321313"></a><a name="p2554141321313"></a>mask/mask[]</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p755431341319"><a name="p755431341319"></a><a name="p755431341319"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001530181537_p0554313181312"><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><a name="zh-cn_topic_0000001530181537_p0554313181312"></a><span id="ph42341681148"><a name="ph42341681148"></a><a name="ph42341681148"></a><a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section4252658182">mask</a>用于控制每次迭代内参与计算的元素。</span></p>
<a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><a name="zh-cn_topic_0000001530181537_ul1255411133132"></a><ul id="zh-cn_topic_0000001530181537_ul1255411133132"><li>逐bit模式：可以按位控制哪些元素参与计算，bit位的值为1表示参与计算，0表示不参与。<p id="zh-cn_topic_0000002523303824_p121114581013"><a name="zh-cn_topic_0000002523303824_p121114581013"></a><a name="zh-cn_topic_0000002523303824_p121114581013"></a>mask为数组形式，数组长度和数组元素的取值范围和操作数的数据类型有关。当操作数为16位时，数组长度为2，mask[0]、mask[1]∈[0, 2<sup id="zh-cn_topic_0000002523303824_sup1411059101"><a name="zh-cn_topic_0000002523303824_sup1411059101"></a><a name="zh-cn_topic_0000002523303824_sup1411059101"></a>64</sup>-1]并且不同时为0；当操作数为32位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup1711155161017"><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a><a name="zh-cn_topic_0000002523303824_sup1711155161017"></a>64</sup>-1]；当操作数为64位时，数组长度为1，mask[0]∈(0, 2<sup id="zh-cn_topic_0000002523303824_sup181195111019"><a name="zh-cn_topic_0000002523303824_sup181195111019"></a><a name="zh-cn_topic_0000002523303824_sup181195111019"></a>32</sup>-1]。</p>
<p id="zh-cn_topic_0000002523303824_p711354105"><a name="zh-cn_topic_0000002523303824_p711354105"></a><a name="zh-cn_topic_0000002523303824_p711354105"></a>例如，mask=[8, 0]，8=0b1000，表示仅第4个元素参与计算。</p>
</li></ul>
<a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><a name="zh-cn_topic_0000001530181537_ul18554121313135"></a><ul id="zh-cn_topic_0000001530181537_ul18554121313135"><li>连续模式：表示前面连续的多少个元素参与计算。取值范围和操作数的数据类型有关，数据类型不同，每次迭代内能够处理的元素个数最大值不同。当操作数为16位时，mask∈[1, 128]；当操作数为32位时，mask∈[1, 64]；当操作数为64位时，mask∈[1, 32]。</li></ul>
<p id="p379345093914"><a name="p379345093914"></a><a name="p379345093914"></a>当源操作数和目的操作数位数不同时，以数据类型的字节较大的为准。例如，源操作数为int16_t类型，目的操作数为int8_t类型，计算mask时以int16_t为准。</p>
</td>
</tr>
<tr id="row185542138131"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p755471321311"><a name="p755471321311"></a><a name="p755471321311"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p135541313101314"><a name="p135541313101314"></a><a name="p135541313101314"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p10767143753811"><a name="p10767143753811"></a><a name="p10767143753811"></a>重复迭代次数。矢量计算单元，每次读取连续的256Bytes数据进行计算，为完成对输入数据的处理，必须通过多次迭代（repeat）才能完成所有数据的读取与计算。repeatTime表示迭代的次数。</p>
<p id="p17845145813432"><a name="p17845145813432"></a><a name="p17845145813432"></a>关于该参数的具体描述请参考<a href="高维切分API.md">高维切分API</a>。</p>
</td>
</tr>
<tr id="row195541813181310"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p15554121320132"><a name="p15554121320132"></a><a name="p15554121320132"></a>repeatParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p18554141331317"><a name="p18554141331317"></a><a name="p18554141331317"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000002523303824_p455461351319"><a name="zh-cn_topic_0000002523303824_p455461351319"></a><a name="zh-cn_topic_0000002523303824_p455461351319"></a>控制操作数地址步长的参数。<a href="UnaryRepeatParams.md">UnaryRepeatParams</a>类型，包含操作数相邻迭代间相同<span id="zh-cn_topic_0000002523303824_ph1256166185416"><a name="zh-cn_topic_0000002523303824_ph1256166185416"></a><a name="zh-cn_topic_0000002523303824_ph1256166185416"></a>DataBlock</span>的地址步长，操作数同一迭代内不同<span id="zh-cn_topic_0000002523303824_ph131833567170"><a name="zh-cn_topic_0000002523303824_ph131833567170"></a><a name="zh-cn_topic_0000002523303824_ph131833567170"></a>DataBlock</span>的地址步长等参数。</p>
<p id="zh-cn_topic_0000002523303824_p1156819418442"><a name="zh-cn_topic_0000002523303824_p1156819418442"></a><a name="zh-cn_topic_0000002523303824_p1156819418442"></a>相邻迭代间的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section139459347420">repeatStride</a>；同一迭代内DataBlock的地址步长参数说明请参考<a href="高维切分API.md#zh-cn_topic_0000002267504656_zh-cn_topic_0000001764162593_section2815124173416">dataBlockStride</a>。</p>
</td>
</tr>
<tr id="row978613973918"><td class="cellrowborder" valign="top" width="16.49%" headers="mcps1.2.4.1.1 "><p id="p658231133912"><a name="p658231133912"></a><a name="p658231133912"></a>count</p>
</td>
<td class="cellrowborder" valign="top" width="11.92%" headers="mcps1.2.4.1.2 "><p id="p058214114395"><a name="p058214114395"></a><a name="p058214114395"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.59%" headers="mcps1.2.4.1.3 "><p id="p7582121119397"><a name="p7582121119397"></a><a name="p7582121119397"></a>参与计算的元素个数。</p>
</td>
</tr>
</tbody>
</table>

**图 1**  halfBlock说明<a name="fig1084698268"></a>  
<!-- img2text -->
```text
src vector
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ BLOCK 0 │ │ BLOCK 1 │ │ BLOCK 2 │ │ BLOCK 3 │ │ BLOCK 4 │ │ BLOCK 5 │ │ BLOCK 6 │ │ BLOCK 7 │
└─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘

dst vector
halfBlock = false
┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐
│ BLOCK 0 │/////////│ │ BLOCK 1 │/////////│ │ BLOCK2  │/////////│ │ BLOCK 3 │/////////│
└─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘
┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐
│ BLOCK4  │/////////│ │ BLOCK5  │/////////│ │ BLOCK 6 │/////////│ │ BLOCK7  │/////////│
└─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘

dst vector
halfBlock = true
┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐
│/////////│ BLOCK 0 │ │/////////│ BLOCK1  │ │/////////│ BLOCK2  │ │/////////│ BLOCK 3 │
└─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘
┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐ ┌─────────┬─────────┐
│/////////│ BLOCK4  │ │/////////│ BLOCK5  │ │/////////│ BLOCK 6 │ │/////////│ BLOCK7  │
└─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘ └─────────┴─────────┘
```

## 返回值说明<a name="section640mcpsimp"></a>

无

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。
-   操作数地址重叠约束请参考[通用地址重叠约束](通用说明和约束.md#section668772811100)。

## 调用示例<a name="section642mcpsimp"></a>

完整的调用样例可参考[CastDequant样例](https://gitcode.com/cann/asc-devkit/tree/master/examples/03_libraries/04_quantization/cast_dequant)。

-   高维切分计算接口样例-mask连续模式

    ```
    int32_t mask = 256 / sizeof(int16_t);
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::CastDequant<uint8_t, int16_t, true, true, true>(dstLocal, srcLocal, mask, 2, { 1, 1, 8, 8 });
    ```

-   高维切分计算接口样例-mask逐bit模式

    ```
    uint64_t mask[2] = { UINT64_MAX, UINT64_MAX };
    // repeatTime = 2, 128 elements one repeat, 256 elements total
    // dstBlkStride, srcBlkStride = 1, no gap between blocks in one repeat
    // dstRepStride, srcRepStride = 8, no gap between repeats
    AscendC::CastDequant<uint8_t, int16_t, true, true, true>(dstLocal, srcLocal, mask, 2, { 1, 1, 8, 8 });
    ```

-   前n个数计算接口样例

    ```
    AscendC::CastDequant<uint8_t, int16_t, true, true>(dstLocal, srcLocal, 256);
    ```

结果示例如下：

```
输入数据srcLocal: 
[20 53 26 12 36  6 20 93 66 30 56 99 59 92  7 37 22 47 98 10 85 29 14 46
 17 34 45 17 25 45 82 17 66 94 68 23 67  8 89  8 92  6 10 80 87 20  9 81
 70 62 11 58 38 83 32 14 38 47 41 63 94 26 96 89 88 35 86 55 60 82 15 65
 92 67 83 23 63 25 85 93 50 91 75 60 80 10 55 20 71 14 67 23 31 63  7 93
 69 45 61 23 43 86 11 81 81 36 76 58 53 25 23 51 59 78 82 10 39 40 24 50
 68 49 79 40  4 53 22 38 45 17 29 54  9 66 98 47 12 47 47 20 98  0 59 77
  1 21 39 70 66 20 68  8 77 77 54  0  3 33 37 37 48 60 83 88 27 70 31 49
 75 21 59  3 99 84 92 84 14 44 26 56 72 56 37 52 39 11  2 59 59 65 71 64
 10 65 62 48 42 79 69 69 27 99  8 38 36 77 34 34 60 50 52 50 41 31 95 68
 27 16 42 64 19 47  0 10 36 36 33 62 98 64 32 81 49 53 27 70 35  9 63  7
 10 89  3 39 94 23 89 16 23 60 71 42 46 58 65 90]
输出数据dstLocal: 
[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 20 53 26 12 36  6 20 93
 66 30 56 99 59 92  7 37  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 22 47 98 10 85 29 14 46 17 34 45 17 25 45 82 17  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 66 94 68 23 67  8 89  8 92  6 10 80 87 20  9 81
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 70 62 11 58 38 83 32 14
 38 47 41 63 94 26 96 89  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 88 35 86 55 60 82 15 65 92 67 83 23 63 25 85 93  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 50 91 75 60 80 10 55 20 71 14 67 23 31 63  7 93
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 69 45 61 23 43 86 11 81
 81 36 76 58 53 25 23 51  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 59 78 82 10 39 40 24 50 68 49 79 40  4 53 22 38  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 45 17 29 54  9 66 98 47 12 47 47 20 98  0 59 77
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1 21 39 70 66 20 68  8
 77 77 54  0  3 33 37 37  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 48 60 83 88 27 70 31 49 75 21 59  3 99 84 92 84  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 14 44 26 56 72 56 37 52 39 11  2 59 59 65 71 64
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10 65 62 48 42 79 69 69
 27 99  8 38 36 77 34 34  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
 60 50 52 50 41 31 95 68 27 16 42 64 19 47  0 10  0  0  0  0  0  0  0  0
  0  0  0  0  0  0  0  0 36 36 33 62 98 64 32 81 49 53 27 70 35  9 63  7
  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10 89  3 39 94 23 89 16
 23 60 71 42 46 58 65 90]
```

