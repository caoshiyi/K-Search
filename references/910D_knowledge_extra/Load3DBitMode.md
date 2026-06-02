# Load3DBitMode<a name="ZH-CN_TOPIC_0000002554343743"></a>

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

```
template <TPosition Dst, TPosition Src, typename T>
__aicore__ inline void LoadData(const LocalTensor<T>& dst, const LocalTensor<T>& src,const Load3DBitModeParam& loadDataParams)
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
<a name="ul989112372517"></a><a name="ul989112372517"></a><ul id="ul989112372517"><li><strong id="b20601132485012"><a name="b20601132485012"></a><a name="b20601132485012"></a>Load3DBitMode接口：</strong><p id="p1460102405016"><a name="p1460102405016"></a><a name="p1460102405016"></a><span id="ph1160113247505"><a name="ph1160113247505"></a><a name="ph1160113247505"></a>Ascend 950PR/Ascend 950DT</span>，支持数据类型为：uint8_t/int8_t/fp8_e4m3fn_t/fp8_e5m2_t/hifloat8_t/half/bfloat16_t/uint32_t/int32_t/float</p>
</li></ul>
</td>
</tr>
<tr id="row146301527557"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p13631162185514"><a name="p13631162185514"></a><a name="p13631162185514"></a>Src</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p1498517449447"><a name="p1498517449447"></a><a name="p1498517449447"></a>源操作数存储的逻辑位置（TPosition），仅Load3DBitMode接口使用。</p>
</td>
</tr>
<tr id="row1916738185515"><td class="cellrowborder" valign="top" width="16.55%" headers="mcps1.2.3.1.1 "><p id="p1716788165519"><a name="p1716788165519"></a><a name="p1716788165519"></a>Dst</p>
</td>
<td class="cellrowborder" valign="top" width="83.45%" headers="mcps1.2.3.1.2 "><p id="p3303153874616"><a name="p3303153874616"></a><a name="p3303153874616"></a>目的操作数存储的逻辑位置（TPosition），仅Load3DBitMode接口使用。</p>
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
<a name="ul207951119112217"></a><a name="ul207951119112217"></a><ul id="ul207951119112217"><li>Load3DBitModeParam，具体参考<a href="#table106611666584">表3</a></li></ul>
<p id="p21811725744"><a name="p21811725744"></a><a name="p21811725744"></a>上述结构体参数定义请参考<span id="ph10562197165916"><a name="ph10562197165916"></a><a name="ph10562197165916"></a>${INSTALL_DIR}</span>/include/ascendc/basic_api/interface/kernel_struct_mm.h，<span id="ph14322531015"><a name="ph14322531015"></a><a name="ph14322531015"></a>${INSTALL_DIR}</span>请替换为CANN软件安装后文件存储路径。</p>
</td>
</tr>
</tbody>
</table>

**表 3**  Load3DBitModeParam类参数说明

<a name="table106611666584"></a>
<table><thead align="left"><tr id="row766110612583"><th class="cellrowborder" valign="top" width="18.52%" id="mcps1.2.3.1.1"><p id="p0661166115810"><a name="p0661166115810"></a><a name="p0661166115810"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.47999999999999%" id="mcps1.2.3.1.2"><p id="p06618615811"><a name="p06618615811"></a><a name="p06618615811"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row666110635819"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.3.1.1 "><p id="p146611567588"><a name="p146611567588"></a><a name="p146611567588"></a>config0</p>
</td>
<td class="cellrowborder" valign="top" width="81.47999999999999%" headers="mcps1.2.3.1.2 "><p id="p1466214619585"><a name="p1466214619585"></a><a name="p1466214619585"></a>uint64_t类型，与Load3DBitModeConfig0位域（bit-field）结构体类型参数config0BitMode组成联合体（union），初始化为0，可以使用类对象的GetConfig0()函数获取其值。</p>
</td>
</tr>
<tr id="row26621063586"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.3.1.1 "><p id="p17662106175815"><a name="p17662106175815"></a><a name="p17662106175815"></a>config0BitMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.47999999999999%" headers="mcps1.2.3.1.2 "><p id="p1666215615583"><a name="p1666215615583"></a><a name="p1666215615583"></a>Load3DBitModeConfig0位域（bit-field）结构体类型，参数参考<a href="#table184321224173">表4</a>，与config0组成联合体（union）。</p>
</td>
</tr>
<tr id="row06625616587"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.3.1.1 "><p id="p46628616587"><a name="p46628616587"></a><a name="p46628616587"></a>config1</p>
</td>
<td class="cellrowborder" valign="top" width="81.47999999999999%" headers="mcps1.2.3.1.2 "><p id="p56625665813"><a name="p56625665813"></a><a name="p56625665813"></a>uint64_t类型，与Load3DBitModeConfig1位域（bit-field）结构体类型参数config1BitMode组成联合体（union），初始化为0，可以使用类对象的GetConfig1()函数获取其值。</p>
</td>
</tr>
<tr id="row18662116155811"><td class="cellrowborder" valign="top" width="18.52%" headers="mcps1.2.3.1.1 "><p id="p3662106155813"><a name="p3662106155813"></a><a name="p3662106155813"></a>config1BitMode</p>
</td>
<td class="cellrowborder" valign="top" width="81.47999999999999%" headers="mcps1.2.3.1.2 "><p id="p18662116135814"><a name="p18662116135814"></a><a name="p18662116135814"></a>Load3DBitModeConfig1位域（bit-field）结构体类型，参数参考<a href="#table87491086815">表5</a>，与config1组成联合体（union）。</p>
</td>
</tr>
</tbody>
</table>

Load3DBitModeParam类参数设计思想说明：

联合体（union）是一种特殊的数据结构，允许在相同的内存位置存储不同的数据类型。union的所有成员共享同一块内存空间，大小由最大成员决定，同一时间只能使用一个成员。

位域（bit-field）是一种特殊的类成员，允许精确控制结构体中成员变量所占用的内存位数。结构体中成员变量从上到下对应内存中从低位到高位。

Load2DBitModeParam类使用union与bit-field方法，采用bit位表达参数类型，使用bit-field结构体自动处理入参的bit位数，并利用union的特性实现多参数融合传递，仅需传递一个入参即可包含全部所需信息，对应底层接口仅需要接收一个参数。同时，当需要修改参数中某一bit位的值时，仅需要通过循环和位运算即可实现，不需要重新传入参数，减少了scalar计算，实现性能提升。

Load2DBitModeParam类可以直接使用LoadData2DParamsV2结构体类型对象初始化：

```
template <typename T>
__aicore__ inline Load3DBitModeParam(const LoadData3DParamsV2<T> &loadData3DParams_);
```

也可以使用各参数的Set函数修改参数值，并且由于使用了联合体，还可以对congfig0和config1直接进行逐bit位修改来修改参数。

**表 4**  Load3DBitModeConfig0结构体参数说明

<a name="table184321224173"></a>
<table><thead align="left"><tr id="row10432152419712"><th class="cellrowborder" valign="top" width="18.54%" id="mcps1.2.3.1.1"><p id="p1243210240717"><a name="p1243210240717"></a><a name="p1243210240717"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.46%" id="mcps1.2.3.1.2"><p id="p164326241176"><a name="p164326241176"></a><a name="p164326241176"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row124321424577"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.3.1.1 "><p id="p843219248713"><a name="p843219248713"></a><a name="p843219248713"></a>kStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.46%" headers="mcps1.2.3.1.2 "><p id="p2029162111396"><a name="p2029162111396"></a><a name="p2029162111396"></a>该指令在目的操作数width维度的传输长度，如果不覆盖最右侧的分形，对于half类型，应为16的倍数，对于int8_t/uint8_t应为32的倍数；覆盖的情况则无倍数要求。取值范围: kStep∈[1, 65535] 。</p>
<p id="p84324248713"><a name="p84324248713"></a><a name="p84324248713"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的kExtension含义相同）</p>
<p id="p15706183624720"><a name="p15706183624720"></a><a name="p15706183624720"></a>该参数是位域结构体的最低位参数，占用16bit，可以使用Load3DBitModeParam类对象的SetKExtension()函数设置其值，使用GetKExtension()函数获取其值。</p>
</td>
</tr>
<tr id="row1043312411716"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.3.1.1 "><p id="p343319241376"><a name="p343319241376"></a><a name="p343319241376"></a>mStep</p>
</td>
<td class="cellrowborder" valign="top" width="81.46%" headers="mcps1.2.3.1.2 "><p id="p1874614235397"><a name="p1874614235397"></a><a name="p1874614235397"></a>该指令在目的操作数height维度的传输长度，如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；覆盖的情况则无倍数要求。取值范围：mStep∈[1, 65535] 。</p>
<p id="p1643315248718"><a name="p1643315248718"></a><a name="p1643315248718"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的mExtension含义相同）</p>
<p id="p12651214493"><a name="p12651214493"></a><a name="p12651214493"></a>该参数是位域结构体的第二低位参数，占用16bit，可以使用Load3DBitModeParam类对象的SetMExtension()函数设置其值，使用GetMExtension()函数获取其值。</p>
</td>
</tr>
<tr id="row1443311243713"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.3.1.1 "><p id="p943352414715"><a name="p943352414715"></a><a name="p943352414715"></a>kPos</p>
</td>
<td class="cellrowborder" valign="top" width="81.46%" headers="mcps1.2.3.1.2 "><p id="p118121125143912"><a name="p118121125143912"></a><a name="p118121125143912"></a>该指令在目的操作数width维度的起点，对于half类型，应为16的倍数，对于int8_t/uint8_t应为32的倍数。取值范围[0, 65535] 。默认为0。</p>
<p id="p124331624373"><a name="p124331624373"></a><a name="p124331624373"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的kStartPt含义相同）</p>
<p id="p2632183114914"><a name="p2632183114914"></a><a name="p2632183114914"></a>该参数是位域结构体的第三低位参数，占用16bit，可以使用Load3DBitModeParam类对象的SetKStartPt()函数设置其值，使用GetKStartPt()函数获取其值。</p>
</td>
</tr>
<tr id="row12433172419720"><td class="cellrowborder" valign="top" width="18.54%" headers="mcps1.2.3.1.1 "><p id="p1943342413715"><a name="p1943342413715"></a><a name="p1943342413715"></a>mPos</p>
</td>
<td class="cellrowborder" valign="top" width="81.46%" headers="mcps1.2.3.1.2 "><p id="p84232027173917"><a name="p84232027173917"></a><a name="p84232027173917"></a>该指令在目的操作数height维度的起点，如果不覆盖最下侧的分形，对于half/int8_t/uint8_t，应为16的倍数；覆盖的情况则无倍数要求。取值范围[0, 65535] 。默认为0。</p>
<p id="p743342415711"><a name="p743342415711"></a><a name="p743342415711"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的mStartPt含义相同）</p>
<p id="p18104735124915"><a name="p18104735124915"></a><a name="p18104735124915"></a>该参数是位域结构体的最高位参数，占用16bit，可以使用Load3DBitModeParam类对象的SetMStartPt()函数设置其值，使用GetMStartPt()函数获取其值。</p>
</td>
</tr>
</tbody>
</table>

**表 5**  Load3DBitModeConfig1结构体参数说明

<a name="table87491086815"></a>
<table><thead align="left"><tr id="row57497818816"><th class="cellrowborder" valign="top" width="18.73%" id="mcps1.2.3.1.1"><p id="p167491985819"><a name="p167491985819"></a><a name="p167491985819"></a>参数名称</p>
</th>
<th class="cellrowborder" valign="top" width="81.27%" id="mcps1.2.3.1.2"><p id="p19749788813"><a name="p19749788813"></a><a name="p19749788813"></a>含义</p>
</th>
</tr>
</thead>
<tbody><tr id="row9749781482"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p12749118982"><a name="p12749118982"></a><a name="p12749118982"></a>strideW</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p24581283916"><a name="p24581283916"></a><a name="p24581283916"></a>卷积核在源操作数width维度滑动的步长，取值范围：strideW∈[1, 63] 。</p>
<p id="p47496810810"><a name="p47496810810"></a><a name="p47496810810"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的strideW含义相同）</p>
<p id="p757122945116"><a name="p757122945116"></a><a name="p757122945116"></a>该参数是位域结构体的最低位参数，占用6bit，可以使用Load3DBitModeParam类对象的SetStrideW()函数设置其值，使用GetStrideW()函数获取其值。</p>
</td>
</tr>
<tr id="row3749188582"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p77491082083"><a name="p77491082083"></a><a name="p77491082083"></a>strideH</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p203413173393"><a name="p203413173393"></a><a name="p203413173393"></a>卷积核在源操作数height 维度滑动的步长，取值范围：strideH∈[1, 63] 。</p>
<p id="p17749168584"><a name="p17749168584"></a><a name="p17749168584"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的strideH含义相同）</p>
<p id="p6581532155112"><a name="p6581532155112"></a><a name="p6581532155112"></a>该参数是位域结构体的第二低位参数，占用6bit，可以使用Load3DBitModeParam类对象的SetStrideH()函数设置其值，使用GetStrideH()函数获取其值。</p>
</td>
</tr>
<tr id="row157491581982"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p11749158481"><a name="p11749158481"></a><a name="p11749158481"></a>Wk</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p1788215316399"><a name="p1788215316399"></a><a name="p1788215316399"></a>卷积核width，取值范围：Wk∈[1, 255] 。</p>
<p id="p17493816813"><a name="p17493816813"></a><a name="p17493816813"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的filterW含义相同）</p>
<p id="p23141635205115"><a name="p23141635205115"></a><a name="p23141635205115"></a>该参数是位域结构体的第三低位参数，占用8bit，可以使用Load3DBitModeParam类对象的SetFilterW()函数设置其值，使用GetFilterW()函数获取其值。</p>
</td>
</tr>
<tr id="row13749381783"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p87501385811"><a name="p87501385811"></a><a name="p87501385811"></a>Hk</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p8493334123911"><a name="p8493334123911"></a><a name="p8493334123911"></a>卷积核height，取值范围：Hk∈[1, 255] 。</p>
<p id="p3750138085"><a name="p3750138085"></a><a name="p3750138085"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的filterH含义相同）</p>
<p id="p44881337125112"><a name="p44881337125112"></a><a name="p44881337125112"></a>该参数是位域结构体的第四低位参数，占用8bit，可以使用Load3DBitModeParam类对象的SetFilterH()函数设置其值，使用GetFilterH()函数获取其值。</p>
</td>
</tr>
<tr id="row77501581887"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p14750783816"><a name="p14750783816"></a><a name="p14750783816"></a>dilationW</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p1635133613916"><a name="p1635133613916"></a><a name="p1635133613916"></a>卷积核width膨胀系数，取值范围：dilationW∈[1, 255] 。</p>
<p id="p18750168489"><a name="p18750168489"></a><a name="p18750168489"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的dilationFilterW含义相同）</p>
<p id="p11947203917519"><a name="p11947203917519"></a><a name="p11947203917519"></a>该参数是位域结构体的第五低位参数，占用8bit，可以使用Load3DBitModeParam类对象的SetDilationFilterW()函数设置其值，使用GetDilationFilterW()函数获取其值。</p>
</td>
</tr>
<tr id="row875018810811"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p1175019815810"><a name="p1175019815810"></a><a name="p1175019815810"></a>dilationH</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p187042372391"><a name="p187042372391"></a><a name="p187042372391"></a>卷积核height膨胀系数，取值范围：dilationH∈[1, 255]。</p>
<p id="p117508812817"><a name="p117508812817"></a><a name="p117508812817"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的dilationFilterH含义相同）</p>
<p id="p523384235117"><a name="p523384235117"></a><a name="p523384235117"></a>该参数是位域结构体的第六低位参数，占用8bit，可以使用Load3DBitModeParam类对象的SetDilationFilterH()函数设置其值，使用GetDilationFilterH()函数获取其值。</p>
</td>
</tr>
<tr id="row1375011813817"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p1775028383"><a name="p1775028383"></a><a name="p1775028383"></a>filterW</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p0759203910397"><a name="p0759203910397"></a><a name="p0759203910397"></a>是否在filterW的基础上将卷积核width增加256 个元素。true，增加；false，不增加。</p>
<p id="p67502818813"><a name="p67502818813"></a><a name="p67502818813"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的filterSizeW含义相同）</p>
<p id="p863234415516"><a name="p863234415516"></a><a name="p863234415516"></a>该参数是位域结构体的第七低位参数，占用1bit，可以使用Load3DBitModeParam类对象的SetFilterSizeW()函数设置其值，使用GetFilterSizeW()函数获取其值。</p>
</td>
</tr>
<tr id="row187881626683"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p1278892614811"><a name="p1278892614811"></a><a name="p1278892614811"></a>filterH</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p964224023915"><a name="p964224023915"></a><a name="p964224023915"></a>是否在filterH的基础上将卷积核height增加256个元素。true，增加；false，不增加。</p>
<p id="p2078832612818"><a name="p2078832612818"></a><a name="p2078832612818"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的filterSizeH含义相同）</p>
<p id="p186571476517"><a name="p186571476517"></a><a name="p186571476517"></a>该参数是位域结构体的第八低位参数，占用1bit，可以使用Load3DBitModeParam类对象的SetFilterSizeH()函数设置其值，使用GetFilterSizeH()函数获取其值。</p>
</td>
</tr>
<tr id="row1673153119810"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p1741431584"><a name="p1741431584"></a><a name="p1741431584"></a>transpose</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p12388915112311"><a name="p12388915112311"></a><a name="p12388915112311"></a>是否启用转置功能，对整个目标矩阵进行转置，支持数据类型为 bool，仅在目的TPosition为A2，且源操作数为half类型时有效。默认为false。</p>
<a name="ul938831592314"></a><a name="ul938831592314"></a><ul id="ul938831592314"><li>true：启用</li><li>false：不启用</li></ul>
<p id="p744132517265"><a name="p744132517265"></a><a name="p744132517265"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的enTranspose含义相同）</p>
<p id="p72431350205113"><a name="p72431350205113"></a><a name="p72431350205113"></a>该参数是位域结构体的第九低位参数，占用1bit，可以使用Load3DBitModeParam类对象的SetTranspose()函数设置其值，使用GetTranspose()函数获取其值。</p>
</td>
</tr>
<tr id="row972813351481"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p207281035983"><a name="p207281035983"></a><a name="p207281035983"></a>fmatrixCtrl</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p1536115301231"><a name="p1536115301231"></a><a name="p1536115301231"></a>表示LoadData3DV2指令从左矩阵还是右矩阵获取FeatureMap的属性描述，与<a href="SetFmatrix.md">SetFmatrix</a>配合使用，当前只支持设置为false，默认值为false。</p>
<a name="ul10361163072314"></a><a name="ul10361163072314"></a><ul id="ul10361163072314"><li>true：从右矩阵中获取FeatureMap的属性描述；</li><li>false：从左矩阵中获取FeatureMap的属性描述。</li></ul>
<p id="p43901629122614"><a name="p43901629122614"></a><a name="p43901629122614"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的fMatrixCtrl含义相同）</p>
<p id="p0258155215115"><a name="p0258155215115"></a><a name="p0258155215115"></a>该参数是位域结构体的第十低位参数，占用1bit，可以使用Load3DBitModeParam类对象的SetFmatrixCtrl()函数设置其值，使用GetFmatrixCtrl()函数获取其值。</p>
</td>
</tr>
<tr id="row38706390816"><td class="cellrowborder" valign="top" width="18.73%" headers="mcps1.2.3.1.1 "><p id="p10870173912818"><a name="p10870173912818"></a><a name="p10870173912818"></a>sizeChannel</p>
</td>
<td class="cellrowborder" valign="top" width="81.27%" headers="mcps1.2.3.1.2 "><p id="p4202155119234"><a name="p4202155119234"></a><a name="p4202155119234"></a>源操作数的通道数，取值范围：channelSize∈[1, 63] 。</p>
<p id="p520205142317"><a name="p520205142317"></a><a name="p520205142317"></a>channelSize的取值要求为：对于uint32_t/int32_t/float，channelSize可取值为4，N * 8，N * 8 + 4；对于half/bfloat16，channelSize可取值为4，8，N * 16，N * 16 + 4，N * 16 + 8；对于int8_t/uint8_t，channelSize可取值为4，8，16， 32 * N，N * 32 + 4，N * 32 + 8，N * 32 + 16；对于int4b_t，ChannelSize可取值为8，16，32，N * 64，N * 64 + 8，N * 64 + 16，N * 64 + 32。N为正整数。</p>
<p id="p879503515265"><a name="p879503515265"></a><a name="p879503515265"></a>（与<a href="Load3D.md#table193501032193419">表4</a>中的channelSize含义相同）</p>
<p id="p22641854155118"><a name="p22641854155118"></a><a name="p22641854155118"></a>该参数是位域结构体的最高位参数，占用16bit，可以使用Load3DBitModeParam类对象的SetChannelSize()函数设置其值，使用GetChannelSize()函数获取其值。</p>
</td>
</tr>
</tbody>
</table>

## 约束说明<a name="section633mcpsimp"></a>

-   操作数地址对齐要求请参见[通用地址对齐约束](通用说明和约束.md#section796754519912)。

## 返回值说明<a name="section640mcpsimp"></a>

无

