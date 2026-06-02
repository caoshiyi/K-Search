# Load3D<a name="ZH-CN_TOPIC_0000002554423549"></a>

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

Load3D用于完成image to column操作，将多维feature map转为二维矩阵。支持如下数据通路：A1-\>A2; B1-\>B2。

## 函数原型<a name="section620mcpsimp"></a>

-   Load3Dv1接口

    ```
    template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData3DParamsV1<U>& loadDataParams)
    ```

-   Load3Dv2接口

    ```
    template <typename T, const IsResetLoad3dConfig &defaultConfig = IS_RESER_LOAD3D_DEFAULT_CONFIG, typename U = PrimT<T>, typename Std::enable_if<Std::is_same<PrimT<T>, U>::value, bool>::type = true>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData3DParamsV2<U>& loadDataParams)
    ```

-   Load3Dv2Pro接口

    ```
    template <typename T>
    __aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src, const LoadData3DParamsV2Pro& loadDataParams)
    ```

## 参数说明<a name="section622mcpsimp"></a>

**表 1**  模板参数说明

<a name="table07381635103112"></a>
<table><thead align="left"><tr id="row117393350314"><th class="cellrowborder" valign="top" width="16.55%" id="mcps1.2.3.1.1"><p id="p14739335193119"><a name="p14739335193119"></a><a name="p14739335193119"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="83.45%" id="mcps1.2.3.1.2"><p id="p8739203514314"><a name="p8739203514314"></a><a name="p8739203514314"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row18739935193119"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p173953516310"><a name="p173953516310"></a><a name="p173953516310"></a>T</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p12739193516313"><a name="p12739193516313"></a><a name="p12739193516313"></a>源操作数和目的操作数的数据类型。</p>
<a name="ul989112372517"></a><a name="ul989112372517"></a><ul id="ul989112372517"><li><strong id="b1670916232277"><a name="b1670916232277"></a><a name="b1670916232277"></a>Load3Dv2接口：</strong><a name="ul156718306403"></a><a name="ul156718306403"></a>
<a name="ul14193394403"></a><a name="ul14193394403"></a>
<a name="ul8314334018"></a><a name="ul8314334018"></a>
<p id="p1892174715910"><a name="p1892174715910"></a><a name="p1892174715910"></a><span id="ph892347155911"><a name="ph892347155911"></a><a name="ph892347155911"></a>Ascend 950PR/Ascend 950DT</span>，支持数据类型为：uint8_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/half/bfloat16_t/uint32_t/int32_t/float</p>
</li><li><strong id="b1883346102717"><a name="b1883346102717"></a><a name="b1883346102717"></a>Load3Dv2Pro接口：</strong><p id="p1295201512711"><a name="p1295201512711"></a><a name="p1295201512711"></a><span id="ph1295615152713"><a name="ph1295615152713"></a><a name="ph1295615152713"></a>Ascend 950PR/Ascend 950DT</span>，uint8_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/half/bfloat16_t/uint32_t/int32_t/float。</p>
</li></ul>
</td>
</tr>
<tr id="row157395356314"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p67393352310"><a name="p67393352310"></a><a name="p67393352310"></a>defaultConfig</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p43749474325"><a name="p43749474325"></a><a name="p43749474325"></a>控制是否在Load3Dv1/Load3Dv2接口内部设置相关属性。 IsResetLoad3dConfig类型。IsResetLoad3dConfig结构定义如下：</p>
<a name="screen206151842173411"></a><a name="screen206151842173411"></a><pre class="screen" codetype="Cpp" id="screen206151842173411">struct IsResetLoad3dConfig {
   bool isSetFMatrix = true;
   bool isSetPadding = true;
}; </pre>
<p id="p165915548368"><a name="p165915548368"></a><a name="p165915548368"></a>isSetFMatrix配置为true，表示在接口内部设置FeatureMap的属性描述（包括l1H、l1W、padList，参数介绍参考<a href="#table679014222918">表3</a>、<a href="#table193501032193419">表4</a>）；设置为false，表示该接口传入的FeatureMap的属性描述不生效，开发者需要通过<a href="SetFmatrix.md">SetFmatrix</a>进行设置。</p>
<p id="p3104316173910"><a name="p3104316173910"></a><a name="p3104316173910"></a>isSetPadding配置为true，表示在接口内部设置Pad属性描述（即padValue参数，参数介绍参考<a href="#table679014222918">表3</a>、<a href="#table193501032193419">表4</a>）；设置为false，表示该接口传入的Pad属性不生效，开发者需要通过<a href="SetLoadDataPaddingValue.md">SetLoadDataPaddingValue</a>进行设置。可参考样例<a href="SetFmatrix.md#section642mcpsimp">调用示例</a>。</p>
<p id="p661153174917"><a name="p661153174917"></a><a name="p661153174917"></a>该参数的默认值如下：</p>
<a name="screen1862315313517"></a><a name="screen1862315313517"></a><pre class="screen" codetype="Cpp" id="screen1862315313517">constexpr IsResetLoad3dConfig IS_RESER_LOAD3D_DEFAULT_CONFIG = {true, true};</pre>
</td>
</tr>
<tr id="row14580104484717"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p127976994513"><a name="p127976994513"></a><a name="p127976994513"></a>U</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p19530141011266"><a name="p19530141011266"></a><a name="p19530141011266"></a>LoadData3DParamsV1/LoadData3DParamsV2中padValue的数据类型。</p>
<a name="ul16617163514483"></a><a name="ul16617163514483"></a><ul id="ul16617163514483"><li>当dst、src使用基础数据类型时， U和dst、src的数据类型T需保持一致，否则编译失败。</li><li>当dst 、src使用<a href="TensorTrait.md">TensorTrait</a>类型时，U和dst、src的数据类型T的LiteType需保持一致，否则编译失败。</li></ul>
<p id="p101791414114819"><a name="p101791414114819"></a><a name="p101791414114819"></a>最后一个模板参数仅用于上述数据类型检查，用户无需关注。</p>
</td>
</tr>
</tbody>
</table>

**表 2**  通用参数说明

<a name="table18368155193919"></a>
<table><thead align="left"><tr id="row1036805543911"><th class="cellrowborder" valign="top" width="16.89168916891689%" id="mcps1.2.4.1.1"><p id="p1836835511393"><a name="p1836835511393"></a><a name="p1836835511393"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="11.111111111111112%" id="mcps1.2.4.1.2"><p id="p10368255163915"><a name="p10368255163915"></a><a name="p10368255163915"></a>输入/输出</p>
</th>
<th class="cellrowborder" valign="top" width="71.99719971997199%" id="mcps1.2.4.1.3"><p id="p436875573911"><a name="p436875573911"></a><a name="p436875573911"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row1436825518395"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p9649151061720"><a name="p9649151061720"></a><a name="p9649151061720"></a>dst</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p1649121041718"><a name="p1649121041718"></a><a name="p1649121041718"></a>输出</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p729941117282"><a name="p729941117282"></a><a name="p729941117282"></a>目的操作数，类型为LocalTensor。</p>
<p id="p3610102994715"><a name="p3610102994715"></a><a name="p3610102994715"></a>数据连续排列顺序由目的操作数所在TPosition决定，具体约束如下：</p>
<a name="ul76107290479"></a><a name="ul76107290479"></a><ul id="ul76107290479"><li>A2：ZZ格式/NZ格式；</li><li>B2：ZN格式；</li><li>A1/B1：无格式要求，一般情况下为NZ格式。</li></ul>
</td>
</tr>
<tr id="row1836875519393"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p7650141019171"><a name="p7650141019171"></a><a name="p7650141019171"></a>src</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p4650610141715"><a name="p4650610141715"></a><a name="p4650610141715"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p192019400435"><a name="p192019400435"></a><a name="p192019400435"></a>源操作数，类型为LocalTensor或GlobalTensor。</p>
<p id="p96501710201711"><a name="p96501710201711"></a><a name="p96501710201711"></a>数据类型需要与dst保持一致。</p>
</td>
</tr>
<tr id="row1767431631917"><td class="cellrowborder" valign="top" width="16.89168916891689%" headers="mcps1.2.4.1.1 "><p id="p667418162198"><a name="p667418162198"></a><a name="p667418162198"></a>loadDataParams</p>
</td>
<td class="cellrowborder" valign="top" width="11.111111111111112%" headers="mcps1.2.4.1.2 "><p id="p11675191610195"><a name="p11675191610195"></a><a name="p11675191610195"></a>输入</p>
</td>
<td class="cellrowborder" valign="top" width="71.99719971997199%" headers="mcps1.2.4.1.3 "><p id="p1667541617193"><a name="p1667541617193"></a><a name="p1667541617193"></a>LoadData参数结构体，类型为：</p>
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>LoadData3DParamsV1，具体参考<a href="#table679014222918">表3</a>。</li><li>LoadData3DParamsV2，具体参考<a href="#table193501032193419">表4</a>。</li><li>LoadData3DParamsV2Pro，具体参考。<a href="#table118027314415">表5</a></li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  LoadData3DParamsV1结构体内参数说明

<a name="table679014222918"></a>
<table><thead align="left"><tr id="row67907213292"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p7790192122918"><a name="p7790192122918"></a><a name="p7790192122918"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p5790826291"><a name="p5790826291"></a><a name="p5790826291"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row15790223298"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1863293418331"><a name="p1863293418331"></a><a name="p1863293418331"></a>padList</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p6632183415336"><a name="p6632183415336"></a><a name="p6632183415336"></a>padding列表 [padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。</p>
</td>
</tr>
<tr id="row177900242919"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p86321534103316"><a name="p86321534103316"></a><a name="p86321534103316"></a>l1H</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p116321234123319"><a name="p116321234123319"></a><a name="p116321234123319"></a>源操作数 height，取值范围：l1H∈[1, 32767]。</p>
</td>
</tr>
<tr id="row179172192915"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1633153423316"><a name="p1633153423316"></a><a name="p1633153423316"></a>l1W</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p96331334153318"><a name="p96331334153318"></a><a name="p96331334153318"></a>源操作数 width，取值范围：l1W∈[1, 32767] 。</p>
</td>
</tr>
<tr id="row9791112182915"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p0633113433313"><a name="p0633113433313"></a><a name="p0633113433313"></a>c1Index</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p3633934113317"><a name="p3633934113317"></a><a name="p3633934113317"></a>该指令在源tensor C1维度的起点，取值范围：c1Index∈[0, 4095] 。默认为0。</p>
</td>
</tr>
<tr id="row1791423291"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1763363473317"><a name="p1763363473317"></a><a name="p1763363473317"></a>fetchFilterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p663314348333"><a name="p663314348333"></a><a name="p663314348333"></a>该指令在卷积核上w维度的起始位置，取值范围：fetchFilterW∈[0, 254] 。默认为0。</p>
</td>
</tr>
<tr id="row7791142192919"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p363333413336"><a name="p363333413336"></a><a name="p363333413336"></a>fetchFilterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p19633173416330"><a name="p19633173416330"></a><a name="p19633173416330"></a>该指令在filter上h维度的起始位置，取值范围：fetchFilterH∈[0, 254] 。默认为0。</p>
</td>
</tr>
<tr id="row37917262919"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p863373412331"><a name="p863373412331"></a><a name="p863373412331"></a>leftTopW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1063310344339"><a name="p1063310344339"></a><a name="p1063310344339"></a>该指令在源操作数上w维度的起点，取值范围：leftTopW∈[-255, 32767] 。默认为0。如果padding_left = a，leftTopW配置为-a。</p>
</td>
</tr>
<tr id="row447165373219"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p186331934123312"><a name="p186331934123312"></a><a name="p186331934123312"></a>leftTopH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p263303413331"><a name="p263303413331"></a><a name="p263303413331"></a>该指令在源操作数上h维度的起点，取值范围：leftTopH∈[-255, 32767] 。默认为0。如果padding_top = a，leftTopH配置为-a。</p>
</td>
</tr>
<tr id="row1833159113212"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p463353411339"><a name="p463353411339"></a><a name="p463353411339"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p6633634133315"><a name="p6633634133315"></a><a name="p6633634133315"></a>卷积核在源操作数w维度滑动的步长，取值范围：strideW∈[1, 63] 。</p>
</td>
</tr>
<tr id="row186410213318"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1763353411339"><a name="p1763353411339"></a><a name="p1763353411339"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p56331734173311"><a name="p56331734173311"></a><a name="p56331734173311"></a>卷积核在源操作数h维度滑动的步长，取值范围：strideH∈[1, 63] 。</p>
</td>
</tr>
<tr id="row58812019339"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1663383453319"><a name="p1663383453319"></a><a name="p1663383453319"></a>filterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p14634634143317"><a name="p14634634143317"></a><a name="p14634634143317"></a>卷积核width，取值范围：filterW∈[1, 255] 。</p>
</td>
</tr>
<tr id="row181139570321"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p13634734103311"><a name="p13634734103311"></a><a name="p13634734103311"></a>filterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p46341634163313"><a name="p46341634163313"></a><a name="p46341634163313"></a>卷积核height，取值范围：filterH∈[1, 255] 。</p>
</td>
</tr>
<tr id="row18113111173316"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p12634143418339"><a name="p12634143418339"></a><a name="p12634143418339"></a>dilationFilterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p66341234143316"><a name="p66341234143316"></a><a name="p66341234143316"></a>卷积核width膨胀系数，取值范围：dilationFilterW∈[1, 255] 。</p>
</td>
</tr>
<tr id="row5386142123314"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p863473413333"><a name="p863473413333"></a><a name="p863473413333"></a>dilationFilterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1363414344338"><a name="p1363414344338"></a><a name="p1363414344338"></a>卷积核height膨胀系数，取值范围：dilationFilterH∈[1, 255] 。</p>
</td>
</tr>
<tr id="row1788117124332"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1963473419334"><a name="p1963473419334"></a><a name="p1963473419334"></a>jumpStride</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1634173414334"><a name="p1634173414334"></a><a name="p1634173414334"></a>迭代之间，目的操作数首地址步长，取值范围：jumpStride∈[1, 127] 。</p>
</td>
</tr>
<tr id="row969718143338"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p10634734203320"><a name="p10634734203320"></a><a name="p10634734203320"></a>repeatMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><div class="p" id="p763423411332"><a name="p763423411332"></a><a name="p763423411332"></a>迭代模式。<a name="ul62531128144719"></a><a name="ul62531128144719"></a><ul id="ul62531128144719"><li>模式0：每次迭代，增加卷积核窗口中的点，对应在目的矩阵上往w维度方向增长。</li><li>模式1：每次迭代，增加滑动窗口左上坐标，对应在目的矩阵上往h维度方向增长。</li></ul>
</div>
<p id="p116341034153312"><a name="p116341034153312"></a><a name="p116341034153312"></a>取值范围：repeatMode∈[0, 1] 。默认为0。</p>
</td>
</tr>
<tr id="row125051163334"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p36341034123314"><a name="p36341034123314"></a><a name="p36341034123314"></a>repeatTime</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p186341734153314"><a name="p186341734153314"></a><a name="p186341734153314"></a>迭代次数，每一次源操作数和目的操作数的地址都会改变。取值范围：repeatTime∈[1，255] 。</p>
</td>
</tr>
<tr id="row156115193335"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p7634183473311"><a name="p7634183473311"></a><a name="p7634183473311"></a>cSize</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1663453412336"><a name="p1663453412336"></a><a name="p1663453412336"></a>配置是否开启cSize = 4(b16) / cSize = 8(b8)优化，取值范围：cSize∈[0, 1] 。默认为0。</p>
</td>
</tr>
<tr id="row613719185337"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p19634173453320"><a name="p19634173453320"></a><a name="p19634173453320"></a>padValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p4635143418336"><a name="p4635143418336"></a><a name="p4635143418336"></a>Pad填充值的数值，数据类型需要与src保持一致。默认为0。若不想使能padding，可将padList设为全0。</p>
</td>
</tr>
</tbody>
</table>

**表 4**  LoadData3DParamsV2结构体内参数说明

<a name="table193501032193419"></a>
<table><thead align="left"><tr id="row235043213342"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p18350163220344"><a name="p18350163220344"></a><a name="p18350163220344"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p10350163223414"><a name="p10350163223414"></a><a name="p10350163223414"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row9351132173415"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p2442124310351"><a name="p2442124310351"></a><a name="p2442124310351"></a>padList</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p944284318351"><a name="p944284318351"></a><a name="p944284318351"></a>padding 列表 [padding_left, padding_right, padding_top, padding_bottom]，每个元素取值范围：[0,255]。默认为{0, 0, 0, 0}。</p>
</td>
</tr>
<tr id="row435123218344"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1944294315353"><a name="p1944294315353"></a><a name="p1944294315353"></a>l1H</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p64421434357"><a name="p64421434357"></a><a name="p64421434357"></a>源操作数height，取值范围：l1H∈[1, 32767]。</p>
</td>
</tr>
<tr id="row3351332163419"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p174426433354"><a name="p174426433354"></a><a name="p174426433354"></a>l1W</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1844294311355"><a name="p1844294311355"></a><a name="p1844294311355"></a>源操作数weight，取值范围：l1W∈[1, 32767] 。</p>
</td>
</tr>
<tr id="row1435123217348"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1444220435352"><a name="p1444220435352"></a><a name="p1444220435352"></a>channelSize</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p244214373519"><a name="p244214373519"></a><a name="p244214373519"></a>源操作数的通道数，取值范围：channelSize∈[1, 63] 。</p>
<p id="p175715359111"><a name="p175715359111"></a><a name="p175715359111"></a>针对以下型号，channelSize的取值要求为：对于uint32_t/int32_t/float，channelSize可取值为4，N * 8，N * 8 + 4；对于half/bfloat16，channelSize可取值为4，8，N * 16，N * 16 + 4，N * 16 + 8；对于int8_t/uint8_t，channelSize可取值为4，8，16， 32 * N，N * 32 + 4，N * 32 + 8，N * 32 + 16；对于int4b_t，ChannelSize可取值为8，16，32，N * 64，N * 64 + 8，N * 64 + 16，N * 64 + 32。N为正整数。</p>
<p id="p4363105305613"><a name="p4363105305613"></a><a name="p4363105305613"></a><span id="ph2363165375615"><a name="ph2363165375615"></a><a name="ph2363165375615"></a>Ascend 950PR/Ascend 950DT</span></p>
</td>
</tr>
<tr id="row203511732113417"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p184429433356"><a name="p184429433356"></a><a name="p184429433356"></a>kExtension</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p2442134363516"><a name="p2442134363516"></a><a name="p2442134363516"></a>该指令在目的操作数width维度的传输长度，如果不覆盖最右侧的分形，对于half类型，应为16的倍数，对于int8_t/uint8_t应为32的倍数；覆盖的情况则无倍数要求。取值范围: kExtension∈[1, 65535] 。</p>
</td>
</tr>
<tr id="row1735113320341"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p8442164363512"><a name="p8442164363512"></a><a name="p8442164363512"></a>mExtension</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p5442124373512"><a name="p5442124373512"></a><a name="p5442124373512"></a>该指令在目的操作数height维度的传输长度，如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；覆盖的情况则无倍数要求。取值范围：mExtension∈[1, 65535] 。</p>
</td>
</tr>
<tr id="row203521632193418"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1044244393516"><a name="p1044244393516"></a><a name="p1044244393516"></a>kStartPt</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p244254319352"><a name="p244254319352"></a><a name="p244254319352"></a>该指令在目的操作数width维度的起点，对于half类型，应为16的倍数，对于int8_t/uint8_t应为32的倍数。取值范围[0, 65535] 。默认为0。</p>
</td>
</tr>
<tr id="row13527328348"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p9443843173519"><a name="p9443843173519"></a><a name="p9443843173519"></a>mStartPt</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p344318433357"><a name="p344318433357"></a><a name="p344318433357"></a>该指令在目的操作数height维度的起点，如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；覆盖的情况则无倍数要求。取值范围[0, 65535] 。默认为0。</p>
</td>
</tr>
<tr id="row0352113233410"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p84439432353"><a name="p84439432353"></a><a name="p84439432353"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p124431643123512"><a name="p124431643123512"></a><a name="p124431643123512"></a>卷积核在源操作数width维度滑动的步长，取值范围：strideW∈[1, 63] 。</p>
</td>
</tr>
<tr id="row1335211325349"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p944344320353"><a name="p944344320353"></a><a name="p944344320353"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p444314363517"><a name="p444314363517"></a><a name="p444314363517"></a>卷积核在源操作数height 维度滑动的步长，取值范围：strideH∈[1, 63] 。</p>
</td>
</tr>
<tr id="row1535219322342"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p104431843183514"><a name="p104431843183514"></a><a name="p104431843183514"></a>filterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p6443154363519"><a name="p6443154363519"></a><a name="p6443154363519"></a>卷积核width，取值范围：filterW∈[1, 255] 。</p>
</td>
</tr>
<tr id="row2352113223413"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p444316434353"><a name="p444316434353"></a><a name="p444316434353"></a>filterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1844316435351"><a name="p1844316435351"></a><a name="p1844316435351"></a>卷积核height，取值范围：filterH∈[1, 255] 。</p>
</td>
</tr>
<tr id="row1835283210341"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p114431643113513"><a name="p114431643113513"></a><a name="p114431643113513"></a>dilationFilterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1544320434358"><a name="p1544320434358"></a><a name="p1544320434358"></a>卷积核width膨胀系数，取值范围：dilationFilterW∈[1, 255] 。</p>
</td>
</tr>
<tr id="row18353133210348"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p2044394312356"><a name="p2044394312356"></a><a name="p2044394312356"></a>dilationFilterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p444314434358"><a name="p444314434358"></a><a name="p444314434358"></a>卷积核height膨胀系数，取值范围：dilationFilterH∈[1, 255] 。</p>
</td>
</tr>
<tr id="row12353032113420"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1443184314353"><a name="p1443184314353"></a><a name="p1443184314353"></a>enTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p10443543203516"><a name="p10443543203516"></a><a name="p10443543203516"></a>是否启用转置功能，对整个目标矩阵进行转置，支持数据类型为 bool，仅在目的TPosition为A2，且源操作数为half类型时有效。默认为false。</p>
<a name="ul15699410133617"></a><a name="ul15699410133617"></a><ul id="ul15699410133617"><li>true：启用</li><li>false：不启用</li></ul>
</td>
</tr>
<tr id="row2353632173414"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1544417435359"><a name="p1544417435359"></a><a name="p1544417435359"></a>enSmallK</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p124441543123520"><a name="p124441543123520"></a><a name="p124441543123520"></a>是否使能small k特性，每个分形矩阵大小为16*4，支持数据类型为 bool，默认为false。当前产品形态，该特性已不再支持。</p>
<a name="ul125422143611"></a><a name="ul125422143611"></a><ul id="ul125422143611"><li>true：使能</li><li>false：不使能</li></ul>
</td>
</tr>
<tr id="row13531232133416"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p5444643113512"><a name="p5444643113512"></a><a name="p5444643113512"></a>padValue</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p7444443193510"><a name="p7444443193510"></a><a name="p7444443193510"></a>Pad填充值的数值，数据类型需要与src保持一致。默认为0。若不想使能padding，可将padList设为全0。</p>
</td>
</tr>
<tr id="row367113177316"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1442152993114"><a name="p1442152993114"></a><a name="p1442152993114"></a>filterSizeW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p14421729173112"><a name="p14421729173112"></a><a name="p14421729173112"></a>是否在filterW的基础上将卷积核width增加256 个元素。true，增加；false，不增加。</p>
</td>
</tr>
<tr id="row5702152012314"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p442182918315"><a name="p442182918315"></a><a name="p442182918315"></a>filterSizeH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p184292953119"><a name="p184292953119"></a><a name="p184292953119"></a>是否在filterH的基础上将卷积核height增加256个元素。true，增加；false，不增加。</p>
</td>
</tr>
<tr id="row1156362463118"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p154362933114"><a name="p154362933114"></a><a name="p154362933114"></a>fMatrixCtrl</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p143162943112"><a name="p143162943112"></a><a name="p143162943112"></a>表示LoadData3DV2指令从左矩阵还是右矩阵获取FeatureMap的属性描述，与<a href="SetFmatrix.md">SetFmatrix</a>配合使用，当前只支持设置为false，默认值为false。</p>
<a name="ul4431529183117"></a><a name="ul4431529183117"></a><ul id="ul4431529183117"><li>true：从右矩阵中获取FeatureMap的属性描述；</li><li>false：从左矩阵中获取FeatureMap的属性描述。</li></ul>
</td>
</tr>
</tbody>
</table>

**表 5**  LoadData3DParamsV2Pro结构体内参数说明

<a name="table118027314415"></a>
<table><thead align="left"><tr id="row1080373114413"><th class="cellrowborder" valign="top" width="18.56%" id="mcps1.2.3.1.1"><p id="p6803631149"><a name="p6803631149"></a><a name="p6803631149"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.44%" id="mcps1.2.3.1.2"><p id="p178031131749"><a name="p178031131749"></a><a name="p178031131749"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row12803131842"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p180316314410"><a name="p180316314410"></a><a name="p180316314410"></a>channelSize</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p8804153113415"><a name="p8804153113415"></a><a name="p8804153113415"></a>源操作数的通道数，取值范围：channelSize∈[1, 63] 。</p>
<p id="p58041831944"><a name="p58041831944"></a><a name="p58041831944"></a>对于half，channelSize除16的余数应当为0，4或8。</p>
<p id="p280413313418"><a name="p280413313418"></a><a name="p280413313418"></a>对于int8_t和uint8_t，channelSize除32的余数应当为0，4，8或16。</p>
<p id="p19804163112418"><a name="p19804163112418"></a><a name="p19804163112418"></a>对于int4b_t，ChannelSize为8，16，32，N*64，N*64+8，N*64+16，N*64+32，N为正整数。</p>
</td>
</tr>
<tr id="row88051331649"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p18805143120412"><a name="p18805143120412"></a><a name="p18805143120412"></a>enTranspose</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p188051331347"><a name="p188051331347"></a><a name="p188051331347"></a>是否启用转置功能，对整个目标矩阵进行转置，支持数据类型为bool，仅在目的TPosition为A2，且源操作数为half类型时有效。默认为false。</p>
<a name="ul178050311942"></a><a name="ul178050311942"></a><ul id="ul178050311942"><li>true：启用</li><li>false：不启用</li></ul>
</td>
</tr>
<tr id="row198062311146"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p118062311942"><a name="p118062311942"></a><a name="p118062311942"></a>enSmallK</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p198064316413"><a name="p198064316413"></a><a name="p198064316413"></a>是否使能small k特性，每个分形矩阵大小为16*4，支持数据类型为bool，默认为false。当前产品形态，该特性已不再支持。</p>
<a name="ul16806173116414"></a><a name="ul16806173116414"></a><ul id="ul16806173116414"><li>true：使能</li><li>false：不使能</li></ul>
</td>
</tr>
<tr id="row678812152919"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p712818336200"><a name="p712818336200"></a><a name="p712818336200"></a>filterSizeW</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1812853392016"><a name="p1812853392016"></a><a name="p1812853392016"></a>是否在filterW的基础上将卷积核width增加256个元素。true，增加；false，不增加。</p>
</td>
</tr>
<tr id="row532971912915"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p4128133314209"><a name="p4128133314209"></a><a name="p4128133314209"></a>filterSizeH</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p1312863318200"><a name="p1312863318200"></a><a name="p1312863318200"></a>是否在filterH的基础上将卷积核height增加256个元素。true，增加；false，不增加。</p>
</td>
</tr>
<tr id="row156501445185512"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1612853320207"><a name="p1612853320207"></a><a name="p1612853320207"></a>fMatrixCtrl</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p812815338206"><a name="p812815338206"></a><a name="p812815338206"></a>表示LoadData3DV2指令从左矩阵还是右矩阵获取FeatureMap的属性描述，与<a href="SetFmatrix.md">SetFmatrix</a>配合使用，当前只支持设置为false，默认值为false。</p>
<a name="ul61281633122018"></a><a name="ul61281633122018"></a><ul id="ul61281633122018"><li>true：从右矩阵中获取FeatureMap的属性描述；</li><li>false：从左矩阵中获取FeatureMap的属性描述。</li></ul>
</td>
</tr>
<tr id="row3826237345"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p7826193713417"><a name="p7826193713417"></a><a name="p7826193713417"></a>extConfig</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p0826337141"><a name="p0826337141"></a><a name="p0826337141"></a>组合参数(uint64_t类型)，默认值为0；</p>
<p id="p53001713161513"><a name="p53001713161513"></a><a name="p53001713161513"></a>extConfig= ((uint64_t)mStartPt &lt;&lt; 48) | ((uint64_t)kStartPt &lt;&lt; 32) | ((uint64_t)mExtension &lt;&lt; 16) | (uint64_t)kExtension。</p>
</td>
</tr>
<tr id="row92374211416"><td class="cellrowborder" valign="top" width="18.56%" headers="mcps1.2.3.1.1 "><p id="p1323042940"><a name="p1323042940"></a><a name="p1323042940"></a>filterConfig</p>
</td>
<td class="cellrowborder" valign="top" width="81.44%" headers="mcps1.2.3.1.2 "><p id="p4231942148"><a name="p4231942148"></a><a name="p4231942148"></a>组合参数(uint64_t类型)，默认值为0X10101010101；</p>
<p id="p196571042171717"><a name="p196571042171717"></a><a name="p196571042171717"></a>filterConfig= ((uint64_t)dilationFilterH &lt;&lt; 40) | ((uint64_t)dilationFilterW &lt;&lt; 32) | ((uint64_t)filterH &lt;&lt; 24) | ((uint64_t)filterW &lt;&lt; 16) | ((uint64_t)strideH &lt;&lt; 8) | (uint64_t)strideW。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

-   LoadData3DParamsV1 cSize特性的开启，需要保证A1/B1中的feature map为 4 channel对齐。
-   调用Load3Dv2/Load3Dv2Pro指令需要通过[SetLoadDataRepeat](SetLoadDataRepeat.md)接口配置dstStride，若不使能repeat模式，接口内repeat相关参数使用默认构造值。

## Load3d数据格式说明<a name="section726316123184"></a>

要求输入的feature map和filter的格式是NC1HWC0，其中C0是最低维度而且C0是固定值为16（对于u8/s8类型为32），C1=C/C0。

为了简化场景，以下场景假设输入的feature map的channel 为4，即Ci=4。输入feature maps在A1中的形状为 \(Hi,Wi,Ci\)，经过load3dv1处理后在A2的数据形状为\(Wo\*Ho, Hk\*Wk\*Ci\)。其中Wo 和Ho是卷积后输出的shape，Hk和Wk是filter的shape。

直观的来看，img2col的过程就是filter在feature map上扫过，将对应feature map的数据展开成输出数据的每一行的过程。filter首先在W方向上滑动Wo步，然后在H方向上走一步然后重复以上过程，最终输出Wo\*Ho行数据。下图中红色和黄色的数据分别代表第一行和第二行。数字表示原始输入数据，filter和输出数据三者之间的关联关系。可以看到，load3dv1首先在输入数据的Ci维度搬运对应于00的4个数，然后搬运对应于01的四个数，最终这一行的大小为Hk\*Wk\*Ci即3\*3\*4=36个数。

对应的feature map格式如下图：

<!-- img2text -->
```
Feature Map                           Filter Window of Feature Map                    Feature Map Matrix after Img2col

          Ci                                          Ci                                              Ci
         ↗                                           ↗                                               ↓
    ┌───────────────────────────────┐           ┌───────────┐      ┌──────────────────────────────────────────────────────────────────────────────┐
   ╱│ 00  01  02                   ╱│          ╱│ 00  01  02│     │00 00 00 00 01 01 01 01 02 02 02 02 03 03 03 03 04 04 04 04 05 05 05 05 │
  ╱ │ 03  04  05                  ╱ │         ╱ │ 03  04  05│     │06 06 06 06 07 07 07 07 08 08 08 08                                      │
 ┌──┼─────────────────────────────┐  │        ┌──┼──────────┐│     ├──────────────────────────────────────────────────────────────────────────────┤
 │06│07  08                      │  │        │06│07  08    ││     │                                                                              │
 │  ├─────────────────────────────┼──┘        │  ├──────────┼┘     │                                                                              │
 │  │                             │ ╱         │  │          │      │                                                                              │
 │  │                             │╱          │  │          │      │                                                                              │
 │  │                             ├──── Wi    │  └──────────┘      │                                                                              │
 │  │                             │           │     Wk             │                                                                              │
 │  │                             │           └─ Hk                │                                                                              │
 └──┴─────────────────────────────┘                                  │                                                                              │
   ↑                                                                 │                                                                              │
   Hi                                                                │                                                                              │
                                                                     │                                                                              │
                                                                     │                                                                              │
                                                                     │                                                                              │
                                                                     └──────────────────────────────────────────────────────────────────────────────┘
                                                                      ↑
                                                                    Wo*Ho


          Ci                                          Ci
         ↗                                           ↗
    ┌───────────────────────────────┐           ┌───────────┐      ┌──────────────────────────────────────────────────────────────────────────────┐
   ╱│                               ╱│          ╱│          │      │                                                                              │
  ╱ │ 00  01  02                  ╱ │         ╱ │ 00  01  02│     ├──────────────────────────────────────────────────────────────────────────────┤
 ┌──┼ 03  04  05                 ┌──┼        ┌──┼ 03  04  05│     │00 00 00 00 01 01 01 01 02 02 02 02 03 03 03 03 04 04 04 04 05 05 05 05 │
 │  │ 06  07  08                 │  │        │  │ 06  07  08│     │06 06 06 06 07 07 07 07 08 08 08 08                                      │
 │  ├─────────────────────────────┼──┘        │  ├──────────┼┘     ├──────────────────────────────────────────────────────────────────────────────┤
 │  │                             │ ╱         │  │          │      │                                                                              │
 │  │                             │╱          │  │          │      │                                                                              │
 │  │                             ├──── Wi    │  └──────────┘      │                                                                              │
 │  │                             │           │     Wk             │                                                                              │
 │  │                             │           └─ Hk                │                                                                              │
 └──┴─────────────────────────────┘                                  │                                                                              │
   ↑                                                                 │                                                                              │
   Hi                                                                │                                                                              │
                                                                     │                                                                              │
                                                                     │                                                                              │
                                                                     │                                                                              │
                                                                     └──────────────────────────────────────────────────────────────────────────────┘
                                                                      ↑
                                                                    Wo*Ho

00~08:
┌────┬────┬────┐
│ 00 │ 01 │ 02 │
├────┼────┼────┤
│ 03 │ 04 │ 05 │
├────┼────┼────┤
│ 06 │ 07 │ 08 │
└────┴────┴────┘
```

对应的filter的格式如下图：

其中n为filter的个数，可以看出维度排布为 \(Hk,Wk,Ci,n\)，但是需要注意的是下图的格式还需要根据Mmad中B矩阵的格式转换。

<!-- img2text -->
```text
                Kernel 0                 Kernel 1                 Kernel 2                               Kernel n
             ┌───────────┐            ┌───────────┐            ┌───────────┐                          ┌───────────┐
          ╱╱ │╲ ╲ ╲ ╲ ╲  │         ╱╱ │╲ ╲ ╲ ╲ ╲  │         ╱╱ │╲ ╲ ╲ ╲ ╲  │                       ╱╱ │╲ ╲ ╲ ╲ ╲  │
       Ci   ┌┼───────────│      ┌┼───────────│      ┌┼───────────│                          ┌┼───────────│
        ╲   ││   │   │   │      ││   │   │   │      ││   │   │   │                          ││   │   │   │
         ╲  ├┼───┼───┼───┤      ├┼───┼───┼───┤      ├┼───┼───┼───┤                          ├┼───┼───┼───┤
          ╲ ││   │   │   │      ││   │   │   │      ││   │   │   │                          ││   │   │   │
           ╲├┼───┼───┼───┤      ├┼───┼───┼───┤      ├┼───┼───┼───┤           • • •          ├┼───┼───┼───┤
            ││   │   │   │      ││   │   │   │      ││   │   │   │                          ││   │   │   │
            └┴───┴───┴───┘      └┴───┴───┴───┘      └┴───┴───┴───┘                          └┴───┴───┴───┘
              ↘                      ↙                  ↙                                        ↙
               ↘                    ↙                 ↙                                        ↙
                ↘                  ↙                ↙                                        ↙
                 ▼                ▼                ▼                                        ▼

                             Ci
                          ╭─────╮
                          │     │──────────────────────────────────────────────────────────────┐
                          ╰─────╯                                                              │
                            │   │   │                                                          │
                            ├───┼───┼──────────────────────────────────────────────────────────┤ ← n
                            │   │   │                                                          │
                            │   │   │                                                          │
                            ├───┼───┼──────────────────────────────────────────────────────────┤
                            │   │   │                                                          │
                            │   │   │                                                          │
                            │ • │ • │ •                                      • • •            │
                            │ • │ • │ •                                                          │  Ci*Wk*Hk
                            │ • │ • │ •                                                          │
                            ├───┼───┼──────────────────────────────────────────────────────────┤
                            │   │   │                                                          │
                            │   │   │                                                          │
                            └───┴───┴──────────────────────────────────────────────────────────┘
                                                 Filter Matrix
```

实际操作中，由于存储空间或者计算能力限制，我们通常会将整个卷积计算分块，一次只搬运并计算一小块数据。

<!-- img2text -->
```
                 Wk*Hk*Ci                                      Co                                           Co
          ┌───────────────────────┐                 ┌───────────────────────┐                 ┌────────────────────────┐
          │┌───┐                  │                 │┌───┐                  │                 │┌───┐ Output partial result│
          ││ A │                  │                 ││ B │                  │                 ││ C │                     │
          │└───┘                  │                 │└───┘                  │                 │└───┘                     │
Wo*Ho     │  │                    │                 │                       │                 │  │                       │
          │  ↓                    │                 │                       │                 │  ↓                       │
          │                       │                 │                       │                 │                          │
          │     Input Matrix      │        X        │     Weight Matrix     │        =        │       Output Matrix      │
          │                       │                 │                       │                 │                          │
          │                       │                 │                       │                 │                          │
          │                       │                 │                 Wk*Hk*Ci                │ Wo*Ho                    │
          │                       │                 │                       │                 │                          │
          │                       │                 │                       │                 │                          │
          └───────────────────────┘                 └───────────────────────┘                 └────────────────────────┘
```

对于A2的feature map来说有两种方案，水平分块和垂直分块。分别对应参数中repeatMode的0和1。

注：下图中的分型矩阵大小为4x4，实际应该为16x16 \(对于u8/s8类型为16x32\)

repeatMode =0时，每次repeat会改变在filter窗口中读取数据点的位置，然后跳到下一个C0的位置。

<!-- img2text -->
```
第1次 repeat

左侧：feature map / filter窗口位置
                ┌────┬────┬────┬────┐
顶层高亮        │    │    │    │    │
                └────┴────┴────┴────┘
前侧读取        ┌────┬────┬────┬────┐
                │ 00 │ 01 │ 02 │ 03 │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                └────┴────┴────┴────┘

中间：分型矩阵
                ┌────┬────┬────┬────┐
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                └────┴────┴────┴────┘
                  ↑
                 读取点

右侧：结果布局
                ┌────┬────┬────┬────┬──────────────────────────────┐
                │ 00 │ 00 │ 00 │ 00 │                              │
                ├────┼────┼────┼────┼──────────────────────────────┤
                │ 01 │ 01 │ 01 │ 01 │                              │
                ├────┼────┼────┼────┼──────────────────────────────┤
                │ 02 │ 02 │ 02 │ 02 │                              │
                ├────┼────┼────┼────┼──────────────────────────────┤
                │ 03 │ 03 │ 03 │ 03 │                              │
                ├────┼────┼────┼────┼──────────────────────────────┤
                │                                               │
                └───────────────────────────────────────────────┘


第2次 repeat

左侧：feature map / filter窗口位置
                ┌────┬────┬────┬────┐
顶层高亮          │    │    │    │    │
                └────┴────┴────┴────┘
前侧读取        ┌────┬────┬────┬────┬────┐
                │ 00 │ 01 │ 02 │ 03 │ 04 │
                ├────┼────┼────┼────┼────┤
                │    │    │    │    │    │
                ├────┼────┼────┼────┼────┤
                │    │    │    │    │    │
                └────┴────┴────┴────┴────┘

中间：分型矩阵
                ┌────┬────┬────┬────┐
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                └────┴────┴────┴────┘
                      ↑
                   读取点右移1格

右侧：结果布局
                ┌────┬────┬────┬────┬────┬────┬────┬────┬──────────┐
                │ 00 │ 00 │ 00 │ 00 │ 01 │ 01 │ 01 │ 01 │          │
                ├────┼────┼────┼────┼────┼────┼────┼────┼──────────┤
                │ 01 │ 01 │ 01 │ 01 │ 02 │ 02 │ 02 │ 02 │          │
                ├────┼────┼────┼────┼────┼────┼────┼────┼──────────┤
                │ 02 │ 02 │ 02 │ 02 │ 03 │ 03 │ 03 │ 03 │          │
                ├────┼────┼────┼────┼────┼────┼────┼────┼──────────┤
                │ 03 │ 03 │ 03 │ 03 │ 04 │ 04 │ 04 │ 04 │          │
                ├────┼────┼────┼────┼────┼────┼────┼────┼──────────┤
                │                                                  │
                └──────────────────────────────────────────────────┘


第3次 repeat

左侧：feature map / filter窗口位置
                ┌────┬────┬────┬────┐
顶层高亮            │    │    │    │    │
                └────┴────┴────┴────┘
前侧读取        ┌────┬────┬────┬────┬────┬────┐
                │ 00 │ 01 │ 02 │ 03 │ 04 │ 05 │
                ├────┼────┼────┼────┼────┼────┤
                │    │    │    │    │    │    │
                ├────┼────┼────┼────┼────┼────┤
                │    │    │    │    │    │    │
                └────┴────┴────┴────┴────┴────┘

中间：分型矩阵
                ┌────┬────┬────┬────┐
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                ├────┼────┼────┼────┤
                │    │    │    │    │
                └────┴────┴────┴────┘
                          ↑
                       读取点继续右移

右侧：结果布局
                ┌────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┐
                │ 00 │ 00 │ 00 │ 00 │ 01 │ 01 │ 01 │ 01 │ 02 │ 02 │ 02 │ 02 │
                ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
                │ 01 │ 01 │ 01 │ 01 │ 02 │ 02 │ 02 │ 02 │ 03 │ 03 │ 03 │ 03 │
                ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
                │ 02 │ 02 │ 02 │ 02 │ 03 │ 03 │ 03 │ 03 │ 04 │ 04 │ 04 │ 04 │
                ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
                │ 03 │ 03 │ 03 │ 03 │ 04 │ 04 │ 04 │ 04 │ 05 │ 05 │ 05 │ 05 │
                ├────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┼────┤
                │                                                              │
                └──────────────────────────────────────────────────────────────┘
```

说明:
- 图中分3行，表示连续3次 repeat 的效果。
- 左侧大块表示 feature map 中 filter 窗口沿水平方向移动；前侧可见标注分别为：
  - 第1次：00 01 02 03
  - 第2次：00 01 02 03 04
  - 第3次：00 01 02 03 04 05
- 右侧结果矩阵中可见数字分布分别为：
  - 第1次：
    - 第1行：00 00 00 00
    - 第2行：01 01 01 01
    - 第3行：02 02 02 02
    - 第4行：03 03 03 03
  - 第2次：
    - 第1行：00 00 00 00 01 01 01 01
    - 第2行：01 01 01 01 02 02 02 02
    - 第3行：02 02 02 02 03 03 03 03
    - 第4行：03 03 03 03 04 04 04 04
  - 第3次：
    - 第1行：00 00 00 00 01 01 01 01 02 02 02 02
    - 第2行：01 01 01 01 02 02 02 02 03 03 03 03
    - 第3行：02 02 02 02 03 03 03 03 04 04 04 04
    - 第4行：03 03 03 03 04 04 04 04 05 05 05 05
- 中间小块表示分型矩阵中的当前位置；每次 repeat 改变在 filter 窗口中读取数据点的位置，然后跳到下一个 C0 位置。

repeatMode =1的时候filter窗口中读取数据的位置保持不变，每个repeat在feature map中前进C0个元素。

<!-- img2text -->
```
第1次 repeat
左侧：feature map 中当前位置                     中间：filter 窗口                  右侧：展开/写入结果

┌──────────────────────────────┐              ┌─────────┐              ┌──────────────────────────────────────────────┐
│ 顶层可见序列: 00 01 02 03    │              │ 00      │              │ 00 00 00 00                                │
│            [红色选中区域]    │              │ 01      │              │ 01 01 01 01                                │
│                              │              │ 02      │              │ 02 02 02 02                                │
│                              │              │ 03      │              │ 03 03 03 03                                │
└──────────────────────────────┘              └─────────┘              └──────────────────────────────────────────────┘


第2次 repeat
左侧：feature map 中当前位置                     中间：filter 窗口                  右侧：展开/写入结果

┌──────────────────────────────┐              ┌─────────┐              ┌──────────────────────────────────────────────┐
│ 顶层可见序列: 00 01 02 03    │              │ 04      │              │ 00 00 00 00                                │
│              04 05 06 07     │              │ 05      │              │ 01 01 01 01                                │
│            [红色选中区域]    │              │ 06      │              │ 02 02 02 02                                │
│                              │              │ 07      │              │ 03 03 03 03                                │
└──────────────────────────────┘              └─────────┘              │ 04 04 04 04                                │
                                                                        │ 05 05 05 05                                │
                                                                        │ 06 06 06 06                                │
                                                                        │ 07 07 07 07                                │
                                                                        └──────────────────────────────────────────────┘


第3次 repeat
左侧：feature map 中当前位置                     中间：filter 窗口                  右侧：展开/写入结果

┌──────────────────────────────┐              ┌─────────┐              ┌──────────────────────────────────────────────┐
│ 顶层可见序列: 00 01 02 03    │              │ 08      │              │ 00 00 00 00                                │
│              04 05 06 07     │              │ 09      │              │ 01 01 01 01                                │
│                          08   │              │ 10      │              │ 02 02 02 02                                │
│                          09   │              │ 11      │              │ 03 03 03 03                                │
│                          10   │              └─────────┘              │ 04 04 04 04                                │
│                          11   │                                       │ 05 05 05 05                                │
│            [红色选中区域]    │                                       │ 06 06 06 06                                │
└──────────────────────────────┘                                       │ 07 07 07 07                                │
                                                                        │ 08 08 08 08                                │
                                                                        │ 09 09 09 09                                │
                                                                        │ 10 10 10 10                                │
                                                                        │ 11 11 11 11                                │
                                                                        └──────────────────────────────────────────────┘
```

说明:
- 图中共有 3 行，对应连续 3 次 repeat。
- 每次 repeat 中间的 filter 窗口读取位置保持不变，为一列 4 个元素。
- 每个 repeat 在 feature map 中前进 C0 个元素，因此红色选中区域依次为：
  - 第1次: 00 01 02 03
  - 第2次: 04 05 06 07
  - 第3次: 08 09 10 11
- 右侧矩阵中，每个编号在一行内重复 4 次：
  - 00 00 00 00
  - 01 01 01 01
  - 02 02 02 02
  - 03 03 03 03
  - 04 04 04 04
  - 05 05 05 05
  - 06 06 06 06
  - 07 07 07 07
  - 08 08 08 08
  - 09 09 09 09
  - 10 10 10 10
  - 11 11 11 11

## 返回值说明<a name="section640mcpsimp"></a>

无

